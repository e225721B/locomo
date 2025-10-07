import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

from global_methods import run_json_trials
from generative_agents.memory_utils import get_embedding

logging.basicConfig(level=logging.INFO)


IMPORTANCE_PROMPT_EN = (
    "Rate the intrinsic importance of the following MEMORY for guiding the agent's future behavior and preferences.\n"
    "Return ONLY a JSON object: {{\"importance\": <1-10>}} (integer). No extra text.\n\n"
    "MEMORY: \n{mem}\n"
)

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
    - Each entry: {
        id: str, text: str, created_at: iso str, source: 'fact'|'reflection',
        importance: int (1-10), embedding: List[float], refs: List[str]|None,
        about: str|'self'|'other'|''
      }
    - Retrieval uses combined score of cosine similarity, importance, and recency.
    """

    def __init__(self, agent: Dict[str, Any], lang: str = 'en'):
        self.agent = agent
        self.lang = lang or 'en'
        self.entries: List[Dict[str, Any]] = list(agent.get('memory_stream', []))

    # ---------- persistence ----------
    def save(self):
        self.agent['memory_stream'] = self.entries

    # ---------- add entries ----------
    def _score_importance(self, text: str) -> int:
        prompt = (IMPORTANCE_PROMPT_JA if self.lang == 'ja' else IMPORTANCE_PROMPT_EN).format(mem=text)
        try:
            obj = run_json_trials(prompt, model='chatgpt', num_tokens_request=80)
            # 1) 期待形: {"importance": <1-10>}
            if isinstance(obj, dict):
                # キー名のゆらぎ対策（"importance" などの余計な引用や大文字小文字差異）
                for k, v in obj.items():
                    kk = str(k).strip()
                    kk = kk.strip('"').strip("'").lower()
                    if kk == 'importance':
                        try:
                            iv = int(float(v))
                            return max(1, min(10, iv))
                        except Exception:
                            continue
            # 2) 単体の数値/数値文字列が返るケース
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
        return f"M{len(self.entries)+1}"

    def add_memory(self, text: str, created_at: Optional[str], source: str, refs: Optional[List[str]] = None, about: str = '') -> Dict[str, Any]:
        entry = {
            'id': self._next_id(),
            'text': text,
            'created_at': created_at or _now_iso(),
            'source': source,
            'refs': refs or [],
            'about': about,
        }
        entry['importance'] = self._score_importance(text)
        entry['embedding'] = self._embed(text)
        self.entries.append(entry)
        return entry

    def add_from_facts(self, speaker_name: str, facts: Dict[str, List], session_date: Optional[str], about: str = '') -> int:
        count = 0
        items = facts.get(speaker_name, []) if isinstance(facts, dict) else []
        for it in items:
            # formats: [fact, id...] or string
            if isinstance(it, list) and len(it) >= 1:
                fact_text = it[0]
                refs = list(map(str, it[1:]))
            elif isinstance(it, str):
                fact_text = it
                refs = []
            else:
                fact_text = str(it)
                refs = []
            text = (session_date + ", " if session_date else "") + fact_text
            self.add_memory(text=text, created_at=session_date, source='fact', refs=refs, about=about)
            count += 1
        return count

    def add_from_reflections(self, reflections: List[str], session_date: Optional[str], about: str) -> int:
        count = 0
        for s in reflections or []:
            if not s:
                continue
            text = (session_date + ", " if session_date else "") + str(s)
            self.add_memory(text=text, created_at=session_date, source='reflection', refs=None, about=about)
            count += 1
        return count

    # ---------- retrieval ----------
    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def retrieve(self, query_text: str, topk: int = 5, now_date: Optional[str] = None,
                 w_sim: float = 0.6, w_imp: float = 0.25, w_rec: float = 0.15, tau_days: float = 7.0) -> List[Dict[str, Any]]:
        if not self.entries or not query_text:
            return []
        try:
            q = list(map(float, get_embedding([query_text])[0]))
        except Exception:
            q = []
        now = _parse_date(now_date) if now_date else datetime.utcnow()
        scored = []
        for e in self.entries:
            sim = self._cosine(q, e.get('embedding') or [])
            imp = (float(e.get('importance', 5)) / 10.0)
            created = _parse_date(e.get('created_at'))
            days = max(0.0, (now - created).total_seconds() / 86400.0)
            rec = float(np.exp(-days / max(0.1, tau_days)))
            score = w_sim * sim + w_imp * imp + w_rec * rec
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:max(1, topk)]]

    # ---------- formatting ----------
    @staticmethod
    def format_snippet(entries: List[Dict[str, Any]], lang: str = 'en') -> str:
        if not entries:
            return ''
        header = "Memory context (top {}):\n".format(len(entries)) if lang != 'ja' else "メモリ文脈 (上位 {}):\n".format(len(entries))
        lines = []
        for e in entries:
            lines.append(f"- {e.get('text','').strip()}")
        return header + "\n".join(lines) + "\n"
