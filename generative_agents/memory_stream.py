import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np

from global_methods import run_json_trials
from generative_agents.memory_utils import get_embedding

logging.basicConfig(level=logging.INFO)


# Helper to safely get embedding (handles numpy arrays)
def _safe_get_embedding(obj, key='embedding', default=None):
    """Safely retrieve embedding, returning default if None or empty (works with numpy arrays)."""
    val = obj.get(key) if isinstance(obj, dict) else None
    if val is None:
        return default if default is not None else []
    # Check if empty (works for list, tuple, numpy array)
    if hasattr(val, '__len__') and len(val) == 0:
        return default if default is not None else []
    return val


IMPORTANCE_PROMPT_EN = (
    "Rate the intrinsic importance of the following MEMORY for guiding the agent's future behavior and preferences.\n"
    "Return ONLY a JSON object: {{\"importance\": <1-10>}} (integer). No extra text.\n\n"
    "MEMORY: \n{mem}\n"
)

#重要度評価プロンプト（日本語版）
IMPORTANCE_PROMPT_JA = (
    "次のMEMORYが、エージェントの将来の行動や好みにどれだけ重要かを1〜10の整数で評価してください。\n"
    "出力は JSON オブジェクトのみ: {{\"importance\": <1-10>}}。余計な文は書かないでください。\n\n"
    "MEMORY:\n{mem}\n"
)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _parse_date(date_str: Optional[str]) -> datetime:
    if not date_str:
        return datetime.utcnow()
    # try several common formats
    for fmt in ("%d %B, %Y", "%d %B %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return datetime.utcnow()


class MemoryStore:
    """
    Minimal memory stream for an agent.
    - Entries are stored as dicts inside agent JSON under key 'memory_stream'.
    - Entry schema (new): see class-level comment above.
    - Retrieval uses combined score of cosine similarity, importance, and recency.
    """

    def __init__(self, agent: Dict[str, Any], lang: str = 'en') -> None:
        self.agent = agent
        self.lang = lang or 'en'
        self.entries = list(agent.get('memory_stream', []))

    # ---------- persistence ----------
    def save(self) -> None:
        self.agent['memory_stream'] = self.entries

    # ---------- add entries ----------
    #重要度計算
    def _score_importance(self, text: str) -> int:
        prompt = (IMPORTANCE_PROMPT_JA if self.lang == 'ja' else IMPORTANCE_PROMPT_EN).format(mem=text)
        try:
            # トークン数を増やし、リトライ回数を減らして高速化
            obj = run_json_trials(prompt, model='chatgpt', num_tokens_request=1000, max_retries=3)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    kk = str(k).strip().strip('"').strip("'").lower()
                    if kk == 'importance':
                        try:
                            iv = int(float(v))
                            return max(1, min(10, iv))
                        except Exception:
                            continue
            if isinstance(obj, (int, float, str)):
                try:
                    iv = int(float(obj))
                    return max(1, min(10, iv))
                except Exception:
                    pass
        except Exception as e:
            logging.warning(f"Importance scoring failed; falling back to default 5: {type(e).__name__}: {e}")
        return 5

    def _embed(self, text: str) -> List[float]:
        try:
            vec = get_embedding([text])[0]
            return list(map(float, vec))
        except Exception:
            return []

    def _next_id(self) -> str:
        return f"M_{len(self.entries)+1:07d}"

    def add_memory(self, text: str, created_at: Optional[str], source_type: str,
                   related_memory_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        # ターンインデックスを付与（順序スコア計算用）
        turn_index = len(self.entries)
        created = created_at or _now_iso()
        entry: Dict[str, Any] = {
            'id': self._next_id(),
            'text': text,
            'created_at': created,
            'last_accessed_at': created,
            'source_type': source_type,
            'importance': self._score_importance(text),
            'recency_weight': 1.0,
            'embedding': self._embed(text),
            'retrieval_score': 0.0,
            'related_memory_ids': list(related_memory_ids or []),
            'turn_index': turn_index,  # ターン順序スコア用
        }
        self.entries.append(entry)
        return entry

    def add_from_facts(self, speaker_name: str, facts: Dict[str, List], session_date: Optional[str], about: str = '') -> int:
        count = 0
        items = facts.get(speaker_name, []) if isinstance(facts, dict) else []
        for it in items:
            if isinstance(it, list) and len(it) >= 1:
                fact_text = it[0]
            elif isinstance(it, str):
                fact_text = it
            else:
                fact_text = str(it)
            text = (session_date + ", " if session_date else "") + fact_text
            self.add_memory(text=text, created_at=session_date, source_type='observation')
            count += 1
        return count

    def add_from_reflections(self, reflections: List[str], session_date: Optional[str], about: str) -> int:
        count = 0
        for s in reflections or []:
            if not s:
                continue
            text = (session_date + ", " if session_date else "") + str(s)
            self.add_memory(text=text, created_at=session_date, source_type='reflection')
            count += 1
        return count
    
    #
    # ---------- retrieval ----------
    #Relevance（関連性）
    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        # Safely check for empty embeddings (handles numpy arrays, lists, None)
        def _is_empty(x):
            if x is None:
                return True
            if hasattr(x, '__len__'):
                return len(x) == 0
            return False
        if _is_empty(a) or _is_empty(b):
            return 0.0
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    def retrieve(self, query_text: str, topk: int = 5, now_date: Optional[str] = None,
                 w_sim: float = 0.5, w_imp: float = 0.2, w_order: float = 0.3) -> List[Dict[str, Any]]:
        """
        メモリストリームから上位 Top-K 件を取得します（類似度・重要度・ターン順序の合成スコア）。
        Args:
            query_text (str):
                クエリとなるテキスト（例: 直前の相手の発話やペルソナ文）。
                空文字または None の場合は空リストを返します。
                
            topk (int, default=5):
                返却するメモリエントリの件数（最低1）。
                
            now_date (Optional[str]):
                未使用（後方互換性のため残存）。
                
            w_sim (float, default=0.5):
                類似度（クエリ埋め込みとメモリ埋め込みのコサイン類似度）の重み。
                値を上げると意味的な近さをより重視します。
                
            w_imp (float, default=0.2):
                重要度（1〜10 を 0.1〜1.0 に正規化）の重み。
                値を上げると「重要度の高い記憶」を優先します。
                
            w_order (float, default=0.3):
                ターン順序の重み。新しいターンほど高いスコア（0.95 ** (max_turn - turn_index)）。
                発話順序を反映するために使用します。

        Returns:
            List[Dict[str, Any]]: スコア順に上位 Top-K のメモリエントリ辞書を返します。

        備考:
            - タイムスタンプベースの recency は秒単位では感度が低いため廃止
            - ターン順序スコアで発話の新しさを反映
        """
        if not self.entries or not query_text:
            return []
        try:
            q = list(map(float, get_embedding([query_text])[0]))
        except Exception:
            q = []
        
        # 最大ターンインデックスを取得（順序スコア計算用）
        max_turn = max((e.get('turn_index', 0) for e in self.entries), default=0)
        
        scored: List[tuple] = []
        for e in self.entries:
            emb_val = _safe_get_embedding(e, 'embedding', [])
            sim = self._cosine(q, emb_val)
            imp = (float(e.get('importance', 5)) / 10.0)
            # ターン順序スコア: 新しいターンほど高い（0.95^差分）
            turn_idx = e.get('turn_index', 0)
            order_score = float(np.power(0.95, max_turn - turn_idx))
            # 合成スコア
            score = w_sim * sim + w_imp * imp + w_order * order_score
            e['order_score'] = order_score
            e['retrieval_score'] = score
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        # topk=0 の場合は空リストを返す
        if topk <= 0:
            return []
        top = [e for _, e in scored[:topk]]
        self.agent['memory_stream'] = self.entries
        return top

    # ---------- formatting ----------
    @staticmethod
    def format_snippet(entries: List[Dict[str, Any]], lang: str = 'en') -> str:
        if not entries:
            return ''
        header = (
            "Memory context (top {}):\n".format(len(entries))
            if lang != 'ja'
            else "メモリ文脈 (上位 {}):\n".format(len(entries))
        )
        lines: List[str] = []
        for e in entries:
            lines.append(f"- {e.get('text','').strip()}")
        return header + "\n".join(lines) + "\n"
