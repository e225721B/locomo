import os, json
import time
import logging
from datetime import datetime, timedelta
from global_methods import run_json_trials, get_openai_embedding
import numpy as np
import pickle as pkl
import random
import unicodedata
import difflib
from typing import List, Dict, Any, Tuple
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


# Reflection prompts (EN/JA)
REFLECTION_INIT_PROMPT_EN = "{}\n\nGiven the information above, what are the three most salient insights that {} has about {}? Give concise answers in the form of a json list where each entry is a string."

REFLECTION_CONTINUE_PROMPT_EN = "{} has the following insights about {} from previous interactions.{}\n\nTheir next conversation is as follows:\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about {} now? Give concise answers in the form of a json list where each entry is a string."

SELF_REFLECTION_INIT_PROMPT_EN = "{}\n\nGiven the information above, what are the three most salient insights that {} has about self? Give concise answers in the form of a json list where each entry is a string."

SELF_REFLECTION_CONTINUE_PROMPT_EN = "{} has the following insights about self.{}\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about self now? Give concise answers in the form of a json list where each entry is a string."

#内省プロンプト
REFLECTION_PROMPT_JA = (
    "{} はこれまでのやり取りから {} について次の洞察を持っています。{}\n\n"
    "次の会話は以下です:\n\n{}\n\n"
    "上の情報に基づき、{} が {} について今持っている最も重要な洞察を3つ挙げてください。"
    "各項目は短い日本語の文とし、JSON 配列（各要素は文字列）のみを返してください。"
)

# 高レベル質問生成用プロンプト（EN/JA）
HL_QUESTIONS_PROMPT_EN = (
    "From the following RECENT MEMORIES (including conversations and reflections), propose exactly three SHORT high-level questions "
    "to observe the relationship dimensions below:\n"
    "1. Power: Who has more influence/dominance in the relationship?\n"
    "2. Intimacy: How close/warm is the relationship?\n"
    "3. TaskOriented: How task-oriented is the dialogue? (vs social/casual chat)\n\n"
    "Each question should be CONCISE (max 20 words) and help assess one of these dimensions.\n"
    "Return ONLY a JSON array of exactly three strings. No extra text.\n\nRECENT MEMORIES:\n{mems}\n"
)

# 高レベル質問生成用プロンプト（日本語版）
HL_QUESTIONS_PROMPT_JA = (
    "次の直近のメモリ（会話・気づきを含む）のみを根拠として、以下の3つの関係性次元を観察するための高レベル質問を1つずつ作成してください:\n"
    "1. Power（力関係）: 主体が相手に対して主観的に認知している社会的・個人的な力関係を指す\n"
    "2. Intimacy（親密度）: どれだけ親しく温かい関係か？\n"
    "3. TaskOriented（タスク指向対話）: やり取りがどれだけタスク指向か？（雑談・社交的 vs 目的達成志向）\n\n"
    "各質問は【簡潔に（30文字程度）】、それぞれの次元を評価するのに役立つものにしてください。\n"
    "出力は JSON 配列（要素は文字列）『のみ』で、ちょうど3件返してください。余計な文章は書かないでください。\n\n直近メモリ:\n{mems}\n"
)


#自分自身への内省プロンプト
SELF_REFLECTION_PROMPT_JA = (
    "{}\n\n上の情報に基づき、{} が自分自身について持っている最も重要な洞察を3つ挙げてください。"
    "各項目は短い日本語の文とし、JSON 配列（各要素は文字列）のみを返してください。"
)

#相手への内省プロンプト
OTHER_REFLECTION_PROMPT_JA = (
    "{} は自分自身について次の洞察を持っています。{}\n\n{}\n\n"
    "上の情報に基づき、{} が自分自身について今持っている最も重要な洞察を3つ挙げてください。"
    "各項目は短い日本語の文とし、JSON 配列（各要素は文字列）のみを返してください。"
)


CONVERSATION2FACTS_PROMPT_EN = """
Write a concise and short list of all possible OBSERVATIONS about each speaker that can be gathered from the CONVERSATION. Each dialog in the conversation contains a dialogue id within square brackets. Each observation should contain a piece of information about the speaker, and also include the dialog id of the dialogs from which the information is taken. The OBSERVATIONS should be objective factual information about the speaker that can be used as a database about them. Avoid abstract observations about the dynamics between the two speakers such as 'speaker is supportive', 'speaker appreciates' etc. Do not leave out any information from the CONVERSATION. Escape all double-quote characters within string output with backslash.\n\nReturn ONLY a valid JSON object: {"<SpeakerName>": [["observation", "D1:1"], ...], "<SpeakerName2>": [...] }. No prose outside JSON.\n"""


CONVERSATION2FACTS_PROMPT_JA = """
以下のCONVERSATIONから各話者について得られる客観的な観察事実(OBSERVATIONS)を漏れなく簡潔に列挙してください。各発話には角括弧内にダイアログIDがあります。各観察は: ["観察内容", "D1:番号"(必要なら複数)] の形で、観察内容はその話者に関する客観的事実に限定し、感情的・抽象的評価(例: 支えている / 感謝している 等)は除外してください。引用符はバックスラッシュでエスケープしてください。\n\n重要: 観察内容の文章は必ず自然な日本語で記述してください（英語で書かないこと）。JSON のキー（話者名）は、CONVERSATION に現れる話者名をそのまま完全一致で使い、翻訳や敬称の付与・省略・変更をしないでください。ダイアログIDの表記も元のまま使用してください。\n\n出力は JSON オブジェクトのみ: {"<話者名>": [["観察", "D1:1"], ...], "<別の話者名>": [...] } 。JSON 以外の文章は書かないでください。\n"""


# --- Relationship assessment prompts (JA/EN) ---
RELATIONSHIP_ASSESS_PROMPT_EN = (
    "You are rating the relationship FROM {src} TO {dst} based on the following CONVERSATION only.\n"
    "Rate four dimensions as integers from 1 (very low) to 10 (very high):\n"
    "- intimacy: closeness/affection {src} feels toward {dst}.\n"
    "- power: perceived dominance/influence {src} feels they have over {dst}. Higher means {src} feels more powerful.\n"
    "- social_distance: perceived formality/distance. Higher means more distant.\n"
    "- trust: how much {src} trusts {dst}.\n\n"
    "Return ONLY a JSON object with exactly these keys and integer values, e.g.\n"
    "{{\"intimacy\":7,\"power\":4,\"social_distance\":3,\"trust\":6}}. No extra text.\n\n"
    "CONVERSATION (date included in first line):\n{conv}\n"
)

#関係値生成のためのプロンプト
RELATIONSHIP_ASSESS_PROMPT_JA = (
    "次のCONVERSATION（当該セッションのみ）から、{src} が {dst} に対して感じている関係性を4つの指標で評価してください。\n"
    "各値は 1（非常に低い）〜10（非常に高い）の整数。\n"
    "- intimacy: 親密度（{src} が {dst} に感じる近しさ/好意）\n"
    "- power: 力関係（{src} が {dst} に対して主観的に認知している社会的・個人的な力関係を指す）\n"
    "- social_distance: 社会的距離（形式ばった距離感。高いほど距離がある）\n"
    "- trust: 信頼（{src} が {dst} をどれだけ信頼しているか）\n\n"
    "出力は次のキーを持つJSONオブジェクトのみ（英語キー名を厳守）: \n"
    "{{\"intimacy\":<1-10>,\"power\":<1-10>,\"social_distance\":<1-10>,\"trust\":<1-10>}} \n"
    "JSON以外の文章は書かないでください。\n\n"
    "CONVERSATION（1行目は日付を含む）:\n{conv}\n"
)

# 各次元ごとに個別の質問とエビデンスを渡し、関係値を生成させるプロンプト（日本語）
RELN_REFLECT_PROMPT_SPLIT_JA = (
    "以下の3つの関係性次元について、それぞれに対応する質問と根拠に基づいて、{src} から {dst} への関係値を 7 段階（-3〜+3 の整数）で評価してください。\n\n"
    "【Power（力関係）】\n"
    "評価基準: {src} が {dst} に対して主観的に認知している社会的・個人的な力関係を指す。\n"
    "質問: {q_power}\n"
    "根拠:\n{evid_power}\n\n"
    "【Intimacy（親密度）】\n"
    "評価基準: {src} が {dst} に対して感じる近しさ・好意の度合い。高いほど親しい。\n"
    "質問: {q_intimacy}\n"
    "根拠:\n{evid_intimacy}\n\n"
    "【TaskOriented（タスク指向対話）】\n"
    "評価基準: やり取りがどれだけタスク指向か。高いほどタスク志向、低いほど雑談・社交的。\n"
    "質問: {q_task}\n"
    "根拠:\n{evid_task}\n\n"
    "出力は JSON オブジェクトのみ（英語キー名厳守）: {{\"Power\":-3..+3,\"Intimacy\":-3..+3,\"TaskOriented\":-3..+3}}。JSON 以外の文章は書かないでください。\n"
)

# 各次元ごとに個別の質問とエビデンスを渡すプロンプト（英語）
RELN_REFLECT_PROMPT_SPLIT_EN = (
    "Rate a 7-point relationship reflection FROM {src} TO {dst} using the dimension-specific questions and evidence below.\n\n"
    "PERSONAS:\n- {src}: {src_persona}\n- {dst}: {dst_persona}\n\n"
    "【Power】\n"
    "Criterion: Perceived dominance/influence {src} feels they have over {dst}. Higher means {src} feels more powerful.\n"
    "Question: {q_power}\n"
    "Evidence:\n{evid_power}\n\n"
    "【Intimacy】\n"
    "Criterion: Degree of closeness/affection {src} feels toward {dst}. Higher means closer.\n"
    "Question: {q_intimacy}\n"
    "Evidence:\n{evid_intimacy}\n\n"
    "【TaskOriented】\n"
    "Criterion: How task-oriented the dialogue is. Higher means more task-focused, lower means more social/casual.\n"
    "Question: {q_task}\n"
    "Evidence:\n{evid_task}\n\n"
    "Return ONLY a JSON object with exactly these keys and integer values in [-3,3], e.g. {{\"Power\":-1,\"Intimacy\":2,\"TaskOriented\":1}}. No extra text.\n"
)


RETRIEVAL_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "models/text-embedding-004")

def get_embedding(texts, model=None):
    return get_openai_embedding(texts, model or RETRIEVAL_MODEL)


def get_session_facts(args, agent_a, agent_b, session_idx, return_embeddings=True):

    # Step 1: get events
    task = json.load(open(os.path.join(args.prompt_dir, 'fact_generation_examples_new.json')))
    # 言語に応じてプロンプトを切り替え (args に lang が無いケースは後方互換で英語)
    lang = getattr(args, 'lang', 'en')
    query = CONVERSATION2FACTS_PROMPT_JA if lang == 'ja' else CONVERSATION2FACTS_PROMPT_EN
    examples = [[task['input_prefix'] + e["input"], json.dumps(e["output"], indent=2)] for e in task['examples']]
    # 日本語出力を強制するため、ja のときは追加指示を与え、英語例文の影響を避ける
    if lang == 'ja':
        query += "\n出力する各『観察内容』のテキストは必ず日本語で書いてください。JSON のキー（話者名）とID表記はそのままで構いません。"
        # 厳格化: JSONのキーは会話に登場する2名のみ・完全一致で固定
        a_name = agent_a.get('name')
        b_name = agent_b.get('name')
        if a_name and b_name:
            query += (
                f"\n厳格条件: JSONのキー名は必ず \"{a_name}\" と \"{b_name}\" の二つのみとし、"
                "これ以外のキーを出力してはいけません。表記は完全一致で出力してください。"
            )
        examples = None

    conversation = ""
    conversation += agent_a['session_%s_date_time' % session_idx] + '\n'
    for i, dialog in enumerate(agent_a['session_%s' % session_idx]):
        try:
            conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"'
        except KeyError:
            conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

        if 'blip_caption' in dialog:
            conversation += ' and shared ' + dialog['blip_caption']
        conversation += '\n'
    
    # print(conversation)
    
    input = task['input_prefix'] + conversation
    # Gemini 2.5 thinking対応で2000トークンに増加
    facts = run_json_trials(query, num_gen=1, num_tokens_request=2000, use_16k=False, examples=examples, input=input)

    # --- 安全対策: facts の話者キーを Agent 名に正規化して整合させる ---
    def _norm_name(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = unicodedata.normalize('NFKC', s).lower().strip()
        # 空白全削除（全角含む）
        s = ''.join(ch for ch in s if not ch.isspace())
        return s

    def _best_key(target: str, keys: list[str]) -> str | None:
        if not keys:
            return None
        t = _norm_name(target)
        # 完全一致（正規化後）
        for k in keys:
            if _norm_name(k) == t:
                return k
        # 近似一致（閾値はやや緩め）
        scored = [(k, difflib.SequenceMatcher(None, _norm_name(k), t).ratio()) for k in keys]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] >= 0.6:
            return scored[0][0]
        return None

    if isinstance(facts, dict):
        fact_keys = list(facts.keys())
        key_a = _best_key(agent_a['name'], fact_keys)
        # B 探索時は A 候補を除外
        rem_keys = [k for k in fact_keys if k != key_a]
        key_b = _best_key(agent_b['name'], rem_keys)
        mapped = {
            agent_a['name']: facts.get(key_a, []) if key_a else [],
            agent_b['name']: facts.get(key_b, []) if key_b else []
        }
        if not key_a or not key_b:
            logging.warning(f"Facts keys did not strictly match agent names. Mapped as: A={key_a}, B={key_b}")
        facts = mapped
    else:
        # 想定外構造のときも KeyError を避ける
        logging.warning("Facts JSON was not a dict; initializing empty facts for both agents.")
        facts = {agent_a['name']: [], agent_b['name']: []}

    if not return_embeddings:
        return facts

    def _fact_text_list(fact_items, speaker_name, sess_date):
        cleaned = []
        for item in fact_items:
            # 形式想定: [fact, id] または [fact, id1, id2, ...] / 文字列単体
            if isinstance(item, list) and len(item) >= 1:
                fact_text = item[0]
                ids = item[1:]
                if ids:
                    fact_text = f"{fact_text} (refs: {', '.join(ids)})"
            elif isinstance(item, str):
                fact_text = item
            else:
                fact_text = str(item)
            cleaned.append(sess_date + ', ' + fact_text)
        return cleaned

    agent_a_embeddings = get_embedding(_fact_text_list(facts[agent_a['name']], agent_a['name'], agent_a['session_%s_date_time' % session_idx]))
    agent_b_embeddings = get_embedding(_fact_text_list(facts[agent_b['name']], agent_b['name'], agent_b['session_%s_date_time' % session_idx]))

    if session_idx > 1:
        with open(args.emb_file, 'rb') as f:
            embs = pkl.load(f)
    
        embs[agent_a['name']] = np.concatenate([embs[agent_a['name']], agent_a_embeddings], axis=0)
        embs[agent_b['name']] = np.concatenate([embs[agent_b['name']], agent_b_embeddings], axis=0)
    else:
        embs = {}
        embs[agent_a['name']] = agent_a_embeddings
        embs[agent_b['name']] = agent_b_embeddings
    
    with open(args.emb_file, 'wb') as f:
        pkl.dump(embs, f)
    
    return facts

#内省を生成させるために情報を投げる関数
def get_session_reflection(args, agent_a, agent_b, session_idx, target: str = 'both'):
    # --- New: Evidence-driven reflection using recent memories ---
    lang = getattr(args, 'lang', 'en')

    def _collect_recent_memories(limit: int = 10) -> List[Dict[str, Any]]:
        """両エージェントのメモリストリーム（memory_stream）から直近のメモリエントリを収集する。
        source_type は 'conversation' と 'reflection' を優先する。フォールバックとして、
        現在のセッションのダイアログと直近のリフレクションからエントリを構築する。
        戻り値は次のフィールドを持つエントリのリスト：
        text、created_at（ISO 文字列）、source、speaker（任意）。"""
        entries: List[Dict[str, Any]] = []
        def _from_stream(agent):
            ms = agent.get('memory_stream') or []
            for e in ms:
                try:
                    st = (e.get('source_type') or e.get('source') or '')
                    if st not in ('conversation', 'reflection'):
                        continue
                    entries.append({
                        'text': e.get('text',''),
                        'created_at': e.get('created_at') or e.get('last_accessed_at') or datetime.utcnow().isoformat(),
                        'importance': int(e.get('importance', 5)),
                        'embedding': _safe_get_embedding(e, 'embedding', []),
                        'source': st,
                        'who': agent.get('name')
                    })
                except Exception:
                    continue
        # 反省する対象に応じて収集範囲を限定（'both' の場合は従来通り両者）
        if target == 'a':
            _from_stream(agent_a)
        elif target == 'b':
            _from_stream(agent_b)
        else:
            _from_stream(agent_a)
            _from_stream(agent_b)
        if not entries:
            # Fallback: use current session dialogues + previous reflections (if any)
            try:
                # フォールバック時も primary を target に応じて選ぶ
                primary = agent_a if target != 'b' else agent_b
                sess = primary.get('session_%s' % session_idx) or []
                sess_date = primary.get('session_%s_date_time' % session_idx) or datetime.utcnow().strftime('%d %B, %Y')
                for d in sess:
                    entries.append({
                        'text': d.get('clean_text') or d.get('text',''),
                        'created_at': sess_date,
                        'importance': 5,
                        'embedding': [],
                        'source': 'conversation',
                        'who': d.get('speaker')
                    })
                if session_idx > 1:
                    prev = primary.get('session_%s_reflection' % (session_idx-1)) or {}
                    for s in (prev.get('self') or []):
                        entries.append({'text': str(s), 'created_at': sess_date, 'importance': 6, 'embedding': [], 'source': 'reflection', 'who': primary.get('name')})
                    for s in (prev.get('other') or []):
                        entries.append({'text': str(s), 'created_at': sess_date, 'importance': 6, 'embedding': [], 'source': 'reflection', 'who': primary.get('name')})
            except Exception:
                pass
        # sort by created_at desc (best-effort)
        def _to_dt(s):
            try:
                return datetime.fromisoformat(s)
            except Exception:
                try:
                    # try formats used elsewhere
                    for fmt in ("%d %B, %Y", "%d %B %Y", "%Y-%m-%d"):
                        return datetime.strptime(s, fmt)
                except Exception:
                    return datetime.utcnow()
        entries.sort(key=lambda x: _to_dt(x.get('created_at') or ''), reverse=True)
        return entries[:limit]

    def _gen_questions(recent_texts: List[str]) -> List[str]:
        joined = '\n- '.join([t for t in recent_texts if t])
        prompt = (HL_QUESTIONS_PROMPT_JA if lang == 'ja' else HL_QUESTIONS_PROMPT_EN).format(mems='- ' + joined)
        qs = run_json_trials(prompt, model='chatgpt', num_tokens_request=3000)
        if isinstance(qs, dict):
            qs = list(qs.values())
        if not isinstance(qs, list):
            qs = []
        # keep 3
        out = []
        for q in qs:
            if isinstance(q, str) and q.strip():
                out.append(q.strip())
            if len(out) >= 3:
                break
        while len(out) < 3:
            out.append( ("質問" if lang=='ja' else "Question") + f" {len(out)+1}")
        return out[:3]

    def _embed_list(texts: List[str]) -> List[List[float]]:
        try:
            vecs = get_openai_embedding(texts, RETRIEVAL_MODEL)
            return [list(map(float, v)) for v in vecs]
        except Exception:
            return [[] for _ in texts]

    def _cos(a: List[float], b: List[float]) -> float:
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
        na = np.linalg.norm(va); nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def _minmax(xs: List[float]) -> List[float]:
        if not xs:
            return []
        mn = min(xs); mx = max(xs)
        if mx - mn < 1e-9:
            return [0.5 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]

    def _score_and_select(entries: List[Dict[str, Any]], questions: List[str], now_dt: datetime) -> Tuple[List[Dict[str, Any]], List[str]]:
        # ensure embeddings for entries
        for e in entries:
            emb = e.get('embedding')
            # Handle numpy array, empty list, or None
            is_empty = emb is None or (isinstance(emb, (list, tuple)) and len(emb) == 0) or (hasattr(emb, '__len__') and len(emb) == 0)
            if is_empty:
                try:
                    e['embedding'] = get_openai_embedding([e.get('text','')], RETRIEVAL_MODEL)[0]
                except Exception:
                    e['embedding'] = []
        q_embs = _embed_list(questions)
        # raw metrics
        relevances = []
        importances = []
        recencies = []
        raws = []
        for e in entries:
            # relevance: max cosine to any question
            sim = 0.0
            emb_val = _safe_get_embedding(e, 'embedding', [])
            emb_floats = list(map(float, emb_val)) if len(emb_val) > 0 else []
            for qe in q_embs:
                sim = max(sim, _cos(emb_floats, qe))
            relevances.append(sim)
            # importance: use 1-10 scale
            imp = float(e.get('importance', 5))
            importances.append(imp)
            # recency: exponential decay by hours since created
            created = e.get('created_at') or now_dt.isoformat()
            try:
                cdt = datetime.fromisoformat(created)
            except Exception:
                try:
                    for fmt in ("%d %B, %Y", "%d %B %Y", "%Y-%m-%d"):
                        cdt = datetime.strptime(created, fmt); break
                except Exception:
                    cdt = now_dt
            hours = max(0.0, (now_dt - cdt).total_seconds() / 3600.0)
            rec = float(np.power(0.995, hours))
            recencies.append(rec)
            raws.append(e)
        # normalize and combine (equal weights)
        rel_n = _minmax(relevances)
        imp_n = _minmax(importances)
        rec_n = _minmax(recencies)
        scored = []
        for i, e in enumerate(raws):
            score = rel_n[i] + imp_n[i] + rec_n[i]
            e['relevance_raw'] = relevances[i]
            e['recency_raw'] = recencies[i]
            e['importance_raw'] = importances[i]
            e['rel_n'] = rel_n[i]; e['imp_n'] = imp_n[i]; e['rec_n'] = rec_n[i]
            e['combined_score'] = score
            scored.append(e)
        scored.sort(key=lambda x: x.get('combined_score', 0.0), reverse=True)
        # format statements with a single numbering across both agents
        lines: List[str] = []
        selected: List[Dict[str, Any]] = []
        char_budget = 2200  # rough safety budget
        for e in scored:
            tag = e.get('who') or 'Agent'
            txt = (e.get('text') or '').strip()
            line = f"[{tag}] {txt}"
            # simple budget control
            if sum(len(l) for l in lines) + len(line) + 10 > char_budget:
                break
            selected.append(e)
            lines.append(line)
            if len(lines) >= 14:  # hard cap
                break
        # numbered strings
        numbered = [f"{idx+1}. {s}" for idx, s in enumerate(lines)]
        return selected, numbered

#候補となる証拠を収集し、スコアリングして選択する
    # Build evidence once per reflection (for both A and B)
    recent = _collect_recent_memories(limit=10)
    def _trim(s: str, n: int = 160) -> str:
        s = s or ''
        return s if len(s) <= n else (s[:n] + '...')
    try:
        # logging.info(f"[hlq] collected recent memories: {len(recent)} items")
        for idx, e in enumerate(recent[:20]):
            # logging.info(
            #     f"[hlq][recent] {idx+1}. who={e.get('who')} src={e.get('source')} date={e.get('created_at')} "
            #     f"imp={e.get('importance')} text={_trim(e.get('text'))}"
            # )
            pass
        if len(recent) > 20:
            # logging.info(f"[hlq][recent] ... ({len(recent)-20} more)")
            pass
    except Exception:
        pass
    recent_texts = [e.get('text','') for e in recent]
    questions = _gen_questions(recent_texts)
    try:
        # logging.info("[hlq] high-level questions (3): " + json.dumps(questions, ensure_ascii=False))
        for qi, q in enumerate(questions or [], start=1):
            try:
                # logging.info(f"[hlq][question] {qi}. {q}")
                pass
            except Exception:
                # logging.info("[hlq][question] %d. %s" % (qi, str(q)))
                pass
    except Exception:
        pass
    now_dt = datetime.utcnow()
    selected_entries, numbered_statements = _score_and_select(recent, questions, now_dt)
    try:
        # logging.info(f"[hlq] selected evidence lines: {len(numbered_statements)}")
        for i, (e, line) in enumerate(zip(selected_entries, numbered_statements), start=1):
            try:
                score = float(e.get('combined_score', 0.0))
                reln = float(e.get('rel_n', 0.0))
                impn = float(e.get('imp_n', 0.0))
                recn = float(e.get('rec_n', 0.0))
            except Exception:
                score = reln = impn = recn = 0.0
            # logging.info(f"[hlq][evidence] {i}. score={score:.3f} rel={reln:.2f} imp={impn:.2f} rec={recn:.2f} | {line}")
            pass
    except Exception:
        pass
    # 証拠ブロック組み立て
    if lang == 'ja':
        ctx_hdr_q = "高レベルの問い (この3つを検索クエリとして使用):\n- " + "\n- ".join(questions)
        ctx_hdr_s = "\n\n根拠となるステートメント（番号付き）:\n" + "\n".join(numbered_statements)
        evidence_block = ctx_hdr_q + ctx_hdr_s
        cite_note = "\n\n注: 洞察の各項目の末尾に (根拠: 1,5,3) のようにこの番号を必ず付けてください。"
    else:
        ctx_hdr_q = "High-level questions (used as retrieval queries):\n- " + "\n- ".join(questions)
        ctx_hdr_s = "\n\nEvidence statements (numbered):\n" + "\n".join(numbered_statements)
        evidence_block = ctx_hdr_q + ctx_hdr_s
        cite_note = "\n\nNote: Append evidence indices like (evidence: 1,5,3) to each insight."

    # Fallback original conversation text (kept in case we want to append)
    conversation = ""
    conversation += agent_a['session_%s_date_time' % session_idx] + '\n'
    for dialog in agent_a['session_%s' % session_idx]:
        conversation += dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"\n'


    # Generate reflections using HL_QUESTIONS + evidence for BOTH directions and SELF
    # Prepare previous "other" (optional continuity)
    prev_ab = ''
    prev_ba = ''
    if session_idx > 1:
        try:
            prev_ab = '\n'.join((agent_a.get('session_%s_reflection' % (session_idx-1), {}) or {}).get('other', []) or [])
        except Exception:
            prev_ab = ''
        try:
            prev_ba = '\n'.join((agent_b.get('session_%s_reflection' % (session_idx-1), {}) or {}).get('other', []) or [])
        except Exception:
            prev_ba = ''

    agent_a_on_b = []
    agent_b_on_a = []
    if lang == 'ja':
        if target in ('a', 'both'):
            prompt_ab = REFLECTION_PROMPT_JA.format(
                agent_a['name'], agent_b['name'], prev_ab, evidence_block + cite_note, agent_a['name'], agent_b['name']
            )
            agent_a_on_b = run_json_trials(prompt_ab, model='chatgpt', num_tokens_request=2000)
        if target in ('b', 'both'):
            prompt_ba = REFLECTION_PROMPT_JA.format(
                agent_b['name'], agent_a['name'], prev_ba, evidence_block + cite_note, agent_b['name'], agent_a['name']
            )
            agent_b_on_a = run_json_trials(prompt_ba, model='chatgpt', num_tokens_request=2000)
    else:
        # EN: keep behavior consistent (use INIT if no previous, otherwise CONTINUE)
        if target in ('a', 'both'):
            if prev_ab.strip():
                prompt_ab = REFLECTION_CONTINUE_PROMPT_EN.format(
                    agent_a['name'], agent_b['name'], prev_ab, evidence_block + cite_note, agent_a['name'], agent_b['name']
                )
            else:
                prompt_ab = REFLECTION_INIT_PROMPT_EN.format(
                    evidence_block + cite_note, agent_a['name'], agent_b['name']
                )
            agent_a_on_b = run_json_trials(prompt_ab, model='chatgpt', num_tokens_request=2000)
        if target in ('b', 'both'):
            if prev_ba.strip():
                prompt_ba = REFLECTION_CONTINUE_PROMPT_EN.format(
                    agent_b['name'], agent_a['name'], prev_ba, evidence_block + cite_note, agent_b['name'], agent_a['name']
                )
            else:
                prompt_ba = REFLECTION_INIT_PROMPT_EN.format(
                    evidence_block + cite_note, agent_b['name'], agent_a['name']
                )
            agent_b_on_a = run_json_trials(prompt_ba, model='chatgpt', num_tokens_request=2000)

    # --- Self reflections ---
    prev_a_self = ''
    prev_b_self = ''
    if session_idx > 1:
        try:
            prev_a_self = '\n'.join((agent_a.get('session_%s_reflection' % (session_idx-1), {}) or {}).get('self', []) or [])
        except Exception:
            prev_a_self = ''
        try:
            prev_b_self = '\n'.join((agent_b.get('session_%s_reflection' % (session_idx-1), {}) or {}).get('self', []) or [])
        except Exception:
            prev_b_self = ''

    if lang == 'ja':
        # 初回は SELF_REFLECTION_PROMPT_JA、継続時は OTHER_REFLECTION_PROMPT_JA（名称は紛らわしいが自己継続用）
        if prev_a_self.strip():
            prompt_a_self = OTHER_REFLECTION_PROMPT_JA.format(
                agent_a['name'], prev_a_self, evidence_block + cite_note, agent_a['name']
            )
        else:
            prompt_a_self = SELF_REFLECTION_PROMPT_JA.format(
                evidence_block + cite_note, agent_a['name']
            )
        if prev_b_self.strip():
            prompt_b_self = OTHER_REFLECTION_PROMPT_JA.format(
                agent_b['name'], prev_b_self, evidence_block + cite_note, agent_b['name']
            )
        else:
            prompt_b_self = SELF_REFLECTION_PROMPT_JA.format(
                evidence_block + cite_note, agent_b['name']
            )
    else:
        if prev_a_self.strip():
            prompt_a_self = SELF_REFLECTION_CONTINUE_PROMPT_EN.format(
                agent_a['name'], prev_a_self, evidence_block + cite_note, agent_a['name']
            )
        else:
            prompt_a_self = SELF_REFLECTION_INIT_PROMPT_EN.format(
                evidence_block + cite_note, agent_a['name']
            )
        if prev_b_self.strip():
            prompt_b_self = SELF_REFLECTION_CONTINUE_PROMPT_EN.format(
                agent_b['name'], prev_b_self, evidence_block + cite_note, agent_b['name']
            )
        else:
            prompt_b_self = SELF_REFLECTION_INIT_PROMPT_EN.format(
                evidence_block + cite_note, agent_b['name']
            )

    agent_a_self = []
    agent_b_self = []
    if target in ('a', 'both'):
        agent_a_self = run_json_trials(prompt_a_self, model='chatgpt', num_tokens_request=2000)
    if target in ('b', 'both'):
        agent_b_self = run_json_trials(prompt_b_self, model='chatgpt', num_tokens_request=2000)

    if type(agent_a_self) == dict:
        agent_a_self = list(agent_a_self.values())
    if type(agent_b_self) == dict:
        agent_b_self = list(agent_b_self.values())
    if type(agent_a_on_b) == dict:
        agent_a_on_b = list(agent_a_on_b.values())
    if type(agent_b_on_a) == dict:
        agent_b_on_a = list(agent_b_on_a.values())  

    reflections = {}
    # make sure to keep schema; fill empty lists for the non-target side
    reflections['a'] = {'self': agent_a_self if target in ('a', 'both') else [],
                        'other': agent_a_on_b if target in ('a', 'both') else []}
    reflections['b'] = {'self': agent_b_self if target in ('b', 'both') else [],
                        'other': agent_b_on_a if target in ('b', 'both') else []}

    return reflections


# ---------------- Relationship Reflection (7-point: -3..+3) -----------------
def _rr_coerce(v):
    try:
        x = int(round(float(v)))
    except Exception:
        x = 0
    return max(-3, min(3, x))

def _ensure_rr_schema(obj: dict) -> dict:
    """Ensure new Intimacy/Power/TaskOriented keys.
    Backward compatibility: map old keys if new ones absent."""
    if not isinstance(obj, dict):
        obj = {}
    # backward mapping from legacy dimension names
    if 'Intimacy' not in obj:
        if 'attentiveness' in obj:
            obj['Intimacy'] = obj.get('attentiveness')
        elif 'Politeness' in obj:
            obj['Intimacy'] = obj.get('Politeness')
    if 'Power' not in obj:
        if 'positivity' in obj:
            obj['Power'] = obj.get('positivity')
        elif 'Self-Disclosure' in obj:
            obj['Power'] = obj.get('Self-Disclosure')
    if 'TaskOriented' not in obj:
        if 'GoalOrientation' in obj:
            obj['TaskOriented'] = obj.get('GoalOrientation')
        elif 'engagement' in obj:
            obj['TaskOriented'] = obj.get('engagement')
        elif 'TaskFocus' in obj:
            obj['TaskOriented'] = obj.get('TaskFocus')
    return {
        'Power': _rr_coerce(obj.get('Power', 0)),
        'Intimacy': _rr_coerce(obj.get('Intimacy', 0)),
        'TaskOriented': _rr_coerce(obj.get('TaskOriented', 0)),
    }

def get_relationship_reflection(args, agent_a, agent_b, session_idx, target: str = 'both', session_dialog=None, min_turns_required: int = 3, evidence_topk: int = None):
    """Compute 7-point (-3..+3) relationship reflection vector using HLQ + evidence pipeline.
    Returns dict:
      {
        'a_to_b': {Power, Intimacy, TaskOriented},
        'b_to_a': {...},
        'by_speaker': { A_name: {toward: B_name, vector:{...}}, B_name: {...} }
      }
    target: 'a' | 'b' | 'both'
    session_dialog: optional list of dialog entries to use instead of/in addition to memory stream
    min_turns_required: minimum number of conversation turns required to compute; returns zeros if fewer
    evidence_topk: number of recent memory entries to use as evidence candidates (default: args.reflection_evidence_topk or 10)
    """
    lang = getattr(args, 'lang', 'en')
    
    # ゼロ値を返すヘルパー
    def _zero_result():
        return {
            'a_to_b': {'Power': 0, 'Intimacy': 0, 'TaskOriented': 0},
            'b_to_a': {'Power': 0, 'Intimacy': 0, 'TaskOriented': 0},
            'by_speaker': {
                agent_a['name']: {'toward': agent_b['name'], 'vector': {'Power': 0, 'Intimacy': 0, 'TaskOriented': 0}},
                agent_b['name']: {'toward': agent_a['name'], 'vector': {'Power': 0, 'Intimacy': 0, 'TaskOriented': 0}}
            }
        }

    # Collect recent memories (same policy as reflections)
    def _collect(limit: int = 10):
        entries: List[Dict[str, Any]] = []
        def _from(agent):
            ms = agent.get('memory_stream') or []
            for e in ms:
                st = (e.get('source_type') or e.get('source') or '')
                if st not in ('conversation', 'reflection'):
                    continue
                entries.append({
                    'text': e.get('text',''),
                    'created_at': e.get('created_at') or e.get('last_accessed_at') or datetime.utcnow().isoformat(),
                    'importance': int(e.get('importance',5)),
                    'embedding': _safe_get_embedding(e, 'embedding', []),
                    'source': st,
                    'who': agent.get('name')
                })
        if target == 'a':
            _from(agent_a)
        elif target == 'b':
            _from(agent_b)
        else:
            _from(agent_a); _from(agent_b)
        
        # session_dialog が渡されていれば、それも entries に追加
        # インデックス付きのタイムスタンプを生成して、最新の発話が優先されるようにする
        if session_dialog and isinstance(session_dialog, list):
            sess_date_str = agent_a.get(f'session_{session_idx}_date_time') or datetime.utcnow().strftime('%d %B, %Y')
            try:
                base_dt = datetime.strptime(sess_date_str, '%d %B, %Y')
            except Exception:
                base_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            for idx, d in enumerate(session_dialog):
                if isinstance(d, dict):
                    # 各発話に秒単位のオフセットを付与（後の発話ほど新しい）
                    offset_dt = base_dt + timedelta(seconds=idx)
                    entries.append({
                        'text': d.get('clean_text') or d.get('text',''),
                        'created_at': offset_dt.isoformat(),
                        'importance': 5,
                        'embedding': [],
                        'source': 'conversation',
                        'who': d.get('speaker')
                    })
        
        if not entries:
            primary = agent_a if target != 'b' else agent_b
            sess = primary.get(f'session_{session_idx}') or []
            sess_date_str = primary.get(f'session_{session_idx}_date_time') or datetime.utcnow().strftime('%d %B, %Y')
            try:
                base_dt = datetime.strptime(sess_date_str, '%d %B, %Y')
            except Exception:
                base_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            for idx, d in enumerate(sess):
                offset_dt = base_dt + timedelta(seconds=idx)
                entries.append({
                    'text': d.get('clean_text') or d.get('text',''),
                    'created_at': offset_dt.isoformat(),
                    'importance': 5,
                    'embedding': [],
                    'source': 'conversation',
                    'who': d.get('speaker')
                })
        def _to_dt(s):
            try:
                return datetime.fromisoformat(s)
            except Exception:
                for fmt in ("%d %B, %Y", "%d %B %Y", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(s, fmt)
                    except Exception:
                        pass
            return datetime.utcnow()
        entries.sort(key=lambda x: _to_dt(x.get('created_at') or ''), reverse=True)
        return entries[:limit]
    
    # 会話データが不十分な場合はゼロ値を返す
    # session_dialog があればそのターン数をチェック、なければ agent のセッションデータをチェック
    dialog_count = 0
    if session_dialog and isinstance(session_dialog, list):
        dialog_count = len(session_dialog)
    else:
        primary = agent_a if target != 'b' else agent_b
        sess = primary.get(f'session_{session_idx}') or []
        dialog_count = len(sess)
    
    if dialog_count < min_turns_required:
        logging.info(f"[rr] Skipping relationship_reflection: only {dialog_count} turns (min {min_turns_required} required)")
        return _zero_result()

    def _embed_list(texts: List[str]):
        try:
            vecs = get_openai_embedding(texts, RETRIEVAL_MODEL)
            return [list(map(float, v)) for v in vecs]
        except Exception:
            return [[] for _ in texts]

    def _cos(a, b):
        # Safely check for empty embeddings (handles numpy arrays, lists, None)
        def _is_empty(x):
            if x is None:
                return True
            if hasattr(x, '__len__'):
                return len(x) == 0
            return False
        if _is_empty(a) or _is_empty(b):
            return 0.0
        va = np.array(a, dtype=float); vb = np.array(b, dtype=float)
        na = np.linalg.norm(va); nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb)/(na*nb))

    def _minmax(xs):
        if not xs:
            return []
        mn, mx = min(xs), max(xs)
        if mx - mn < 1e-9:
            return [0.5 for _ in xs]
        return [(x-mn)/(mx-mn) for x in xs]

    # エビデンス候補数: 引数 > args設定 > デフォルト30
    _evidence_limit = evidence_topk or getattr(args, 'reflection_evidence_topk', 30)
    recent = _collect(limit=_evidence_limit)
    recent_texts = [e.get('text','') for e in recent]
    # HLQ generation (Gemini 2.5 thinking対応で3000トークンに増加)
    joined = '\n- '.join([t for t in recent_texts if t])
    hl_prompt = (HL_QUESTIONS_PROMPT_JA if lang=='ja' else HL_QUESTIONS_PROMPT_EN).format(mems='- ' + joined)
    qs = run_json_trials(hl_prompt, model='chatgpt', num_tokens_request=3000)
    if isinstance(qs, dict):
        qs = list(qs.values())
    if not isinstance(qs, list):
        qs = []
    hlq = []
    for q in qs:
        if isinstance(q, str) and q.strip():
            hlq.append(q.strip())
        if len(hlq) >= 3:
            break
    while len(hlq) < 3:
        hlq.append(('質問' if lang=='ja' else 'Question') + f' {len(hlq)+1}')
    # Evidence selection
    for e in recent:
        emb = e.get('embedding')
        # Handle numpy array, empty list, or None
        is_empty = emb is None or (isinstance(emb, (list, tuple)) and len(emb) == 0) or (hasattr(emb, '__len__') and len(emb) == 0)
        if is_empty:
            try:
                e['embedding'] = get_openai_embedding([e.get('text','')], RETRIEVAL_MODEL)[0]
            except Exception:
                e['embedding'] = []
    
    # --- 各次元ごとに個別にスコアリングしてエビデンスを選定 ---
    # hlq[0]=Power用, hlq[1]=Intimacy用, hlq[2]=TaskOriented用 と仮定
    q_embs = _embed_list(hlq)  # 3つの質問のembedding
    # 各メモリに対して基本スコア（importance, order）を計算
    # 最大ターンインデックスを取得
    max_turn = max((e.get('turn_index', i) for i, e in enumerate(recent)), default=0)
    
    base_scores = []
    for i, e in enumerate(recent):
        importance = float(e.get('importance', 5))
        # ターン順序スコア: 新しいターンほど高い
        turn_idx = e.get('turn_index', i)  # turn_indexがない場合はインデックスを使用
        order_score = float(np.power(0.95, max_turn - turn_idx))
        base_scores.append({'entry': e, 'importance': importance, 'order': order_score})
    
    def _score_for_question(q_emb, char_budget=700, max_items=5):
        """特定の質問に対してスコアリングし、上位エビデンスを返す"""
        scored = []
        relevances = []
        importances = []
        orders = []
        for bs in base_scores:
            e = bs['entry']
            emb_val = _safe_get_embedding(e, 'embedding', [])
            emb_floats = list(map(float, emb_val)) if len(emb_val) > 0 else []
            sim = _cos(emb_floats, q_emb) if emb_floats and q_emb else 0.0
            relevances.append(sim)
            importances.append(bs['importance'])
            orders.append(bs['order'])
            scored.append(e)
        
        # 正規化
        rel_n = _minmax(relevances)
        imp_n = _minmax(importances)
        ord_n = _minmax(orders)
        
        # 合成スコアでソート（重み: 類似度0.5, 重要度0.2, 順序0.3）
        ranked = []
        for i, e in enumerate(scored):
            combined = 0.5 * rel_n[i] + 0.2 * imp_n[i] + 0.3 * ord_n[i]
            ranked.append((combined, e))
        ranked.sort(key=lambda x: x[0], reverse=True)
        
        # 上位を選択
        lines = []
        for _, e in ranked:
            tag = e.get('who') or 'Agent'
            txt = (e.get('text') or '').strip()
            line = f'[{tag}] {txt}'
            if sum(len(l) for l in lines) + len(line) + 10 > char_budget:
                break
            lines.append(line)
            if len(lines) >= max_items:
                break
        return lines
    
    # 各次元ごとにエビデンスを選定
    evid_power_lines = _score_for_question(q_embs[0] if len(q_embs) > 0 else [])
    evid_intimacy_lines = _score_for_question(q_embs[1] if len(q_embs) > 1 else [])
    evid_task_lines = _score_for_question(q_embs[2] if len(q_embs) > 2 else [])
    
    # 番号付きテキストに変換
    evid_power_numbered = [f'{i+1}. {s}' for i, s in enumerate(evid_power_lines)]
    evid_intimacy_numbered = [f'{i+1}. {s}' for i, s in enumerate(evid_intimacy_lines)]
    evid_task_numbered = [f'{i+1}. {s}' for i, s in enumerate(evid_task_lines)]
    
    evid_power_block = '\n'.join(evid_power_numbered) if evid_power_numbered else '(なし)'
    evid_intimacy_block = '\n'.join(evid_intimacy_numbered) if evid_intimacy_numbered else '(なし)'
    evid_task_block = '\n'.join(evid_task_numbered) if evid_task_numbered else '(なし)'
    
    # 後方互換用の統合エビデンス（トレース用）
    all_lines = list(set(evid_power_lines + evid_intimacy_lines + evid_task_lines))
    numbered = [f'{i+1}. {s}' for i, s in enumerate(all_lines[:14])]
    hlq_block = '\n- ' + '\n- '.join(hlq)
    evid_block = '\n'.join(numbered)
    
    src_p = (agent_a.get('persona_summary') or '').strip()
    dst_p = (agent_b.get('persona_summary') or '').strip()
    
    # 新しい分割プロンプトを使用
    if lang == 'ja':
        p_ab = RELN_REFLECT_PROMPT_SPLIT_JA.format(
            src=agent_a['name'], dst=agent_b['name'],
            src_persona=src_p, dst_persona=dst_p,
            q_power=hlq[0], evid_power=evid_power_block,
            q_intimacy=hlq[1], evid_intimacy=evid_intimacy_block,
            q_task=hlq[2], evid_task=evid_task_block
        )
        p_ba = RELN_REFLECT_PROMPT_SPLIT_JA.format(
            src=agent_b['name'], dst=agent_a['name'],
            src_persona=dst_p, dst_persona=src_p,
            q_power=hlq[0], evid_power=evid_power_block,
            q_intimacy=hlq[1], evid_intimacy=evid_intimacy_block,
            q_task=hlq[2], evid_task=evid_task_block
        )
    else:
        p_ab = RELN_REFLECT_PROMPT_SPLIT_EN.format(
            src=agent_a['name'], dst=agent_b['name'],
            src_persona=src_p, dst_persona=dst_p,
            q_power=hlq[0], evid_power=evid_power_block,
            q_intimacy=hlq[1], evid_intimacy=evid_intimacy_block,
            q_task=hlq[2], evid_task=evid_task_block
        )
        p_ba = RELN_REFLECT_PROMPT_SPLIT_EN.format(
            src=agent_b['name'], dst=agent_a['name'],
            src_persona=dst_p, dst_persona=src_p,
            q_power=hlq[0], evid_power=evid_power_block,
            q_intimacy=hlq[1], evid_intimacy=evid_intimacy_block,
            q_task=hlq[2], evid_task=evid_task_block
        )

    # --- Trace: write prompt components to a dedicated JSONL file ---
    try:
        trace_path = os.path.join(getattr(args, 'out_dir', '.'), f'rr_prompt_trace_session_{session_idx}.jsonl')
        # common payload pieces (新形式: 各次元ごとのエビデンスも含む)
        base_payload = {
            'time': datetime.utcnow().isoformat(),
            'session_idx': session_idx,
            'language': lang,
            'hlq_list': hlq,
            'hlq_block': hlq_block.strip(),
            'evidence_numbered': numbered,
            'evidence_block': evid_block.strip(),
            # 各次元ごとの詳細
            'per_dimension': {
                'Power': {
                    'question': hlq[0] if len(hlq) > 0 else '',
                    'evidence': evid_power_numbered,
                },
                'Intimacy': {
                    'question': hlq[1] if len(hlq) > 1 else '',
                    'evidence': evid_intimacy_numbered,
                },
                'TaskOriented': {
                    'question': hlq[2] if len(hlq) > 2 else '',
                    'evidence': evid_task_numbered,
                },
            },
        }
        # A -> B
        if target in ('a', 'both'):
            payload_ab = {
                **base_payload,
                'direction': 'a_to_b',
                'src': agent_a['name'],
                'dst': agent_b['name'],
                'src_persona': src_p,
                'dst_persona': dst_p,
                'prompt': p_ab,
            }
            with open(trace_path, 'a', encoding='utf-8') as tf:
                tf.write(json.dumps(payload_ab, ensure_ascii=False) + "\n")
        # B -> A
        if target in ('b', 'both'):
            payload_ba = {
                **base_payload,
                'direction': 'b_to_a',
                'src': agent_b['name'],
                'dst': agent_a['name'],
                'src_persona': dst_p,
                'dst_persona': src_p,
                'prompt': p_ba,
            }
            with open(trace_path, 'a', encoding='utf-8') as tf:
                tf.write(json.dumps(payload_ba, ensure_ascii=False) + "\n")
    except Exception as _e:
        logging.debug(f"Failed to append rr prompt trace: {_e}")

    result = {'a_to_b': None, 'b_to_a': None, 'by_speaker': {}}
    if target in ('a','both'):
        try:
            r_ab = run_json_trials(p_ab, model='chatgpt', num_tokens_request=2000)
        except Exception as e:
            logging.debug(f"relationship_reflection A->B parsing failed: {e}")
            r_ab = {}
        if not isinstance(r_ab, dict):
            logging.debug(f"relationship_reflection A->B not dict, got {type(r_ab)}; defaulting to zeros")
            r_ab = {}
        result['a_to_b'] = _ensure_rr_schema(r_ab)
    result['by_speaker'][agent_a['name']] = {'toward': agent_b['name'], 'vector': result['a_to_b']}
    if target in ('b','both'):
        try:
            r_ba = run_json_trials(p_ba, model='chatgpt', num_tokens_request=2000)
        except Exception as e:
            logging.debug(f"relationship_reflection B->A parsing failed: {e}")
            r_ba = {}
        if not isinstance(r_ba, dict):
            logging.debug(f"relationship_reflection B->A not dict, got {type(r_ba)}; defaulting to zeros")
            r_ba = {}
        result['b_to_a'] = _ensure_rr_schema(r_ba)
    result['by_speaker'][agent_b['name']] = {'toward': agent_a['name'], 'vector': result['b_to_a']}

    # --- プロンプトと生成結果を別ファイルに出力 ---
    try:
        result_trace_path = os.path.join(getattr(args, 'out_dir', '.'), 'relationship_reflection_details.jsonl')
        if target in ('a', 'both'):
            detail_ab = {
                'time': datetime.utcnow().isoformat(),
                'session_idx': session_idx,
                'direction': 'a_to_b',
                'src': agent_a['name'],
                'dst': agent_b['name'],
                'prompt': p_ab,
                'generated_values': result['a_to_b'],
            }
            with open(result_trace_path, 'a', encoding='utf-8') as rf:
                rf.write(json.dumps(detail_ab, ensure_ascii=False) + "\n")
        if target in ('b', 'both'):
            detail_ba = {
                'time': datetime.utcnow().isoformat(),
                'session_idx': session_idx,
                'direction': 'b_to_a',
                'src': agent_b['name'],
                'dst': agent_a['name'],
                'prompt': p_ba,
                'generated_values': result['b_to_a'],
            }
            with open(result_trace_path, 'a', encoding='utf-8') as rf:
                rf.write(json.dumps(detail_ba, ensure_ascii=False) + "\n")
    except Exception as _e:
        logging.debug(f"Failed to append relationship_reflection details: {_e}")

    if result['a_to_b'] is None:
        result['a_to_b'] = _ensure_rr_schema({})
    if result['b_to_a'] is None:
        result['b_to_a'] = _ensure_rr_schema({})
    return result


def _coerce_score(x, name):
    """Coerce a value to an int in [1,10]."""
    try:
        v = int(round(float(x)))
    except Exception:
        logging.warning(f"Relationship score for {name} not an int: {x}. Defaulting to 5")
        v = 5
    return max(1, min(10, v))


def _ensure_relationship_schema(obj: dict) -> dict:
    """Ensure required keys exist and are ints in [1,10]."""
    keys = ["intimacy", "power", "social_distance", "trust"]
    out = {}
    for k in keys:
        out[k] = _coerce_score(obj.get(k, 5), k)
    return out


def get_session_relationships(args, agent_a, agent_b, session_idx, session_dialog=None, session_date=None):
    """
    Evaluate relationships after a session. Returns a dict:
    {
      'a_to_b': { 'intimacy':int, 'power':int, 'social_distance':int, 'trust':int },
      'b_to_a': { ... }
    }
    """
    # Build conversation text (same style as reflections)
    conversation = ""
    # If session_dialog provided, use it (mid-session partial evaluation)
    if session_dialog is not None:
        # use provided session_date if available, otherwise try to read from agent_a
        if session_date:
            conversation += session_date + '\n'
        else:
            conversation += agent_a.get('session_%s_date_time' % session_idx, '') + '\n'
        for dialog in session_dialog:
            conversation += dialog.get('speaker', 'Unknown') + ' said, "' + dialog.get('clean_text', dialog.get('text', '')) + '"\n'
    else:
        conversation += agent_a.get('session_%s_date_time' % session_idx, '') + '\n'
        for dialog in agent_a.get('session_%s' % session_idx, []):
            conversation += dialog.get('speaker', 'Unknown') + ' said, "' + dialog.get('clean_text', dialog.get('text', '')) + '"\n'

    lang = getattr(args, 'lang', 'en')
    if lang == 'ja':
        prompt_ab = RELATIONSHIP_ASSESS_PROMPT_JA.format(src=agent_a['name'], dst=agent_b['name'], conv=conversation)
        prompt_ba = RELATIONSHIP_ASSESS_PROMPT_JA.format(src=agent_b['name'], dst=agent_a['name'], conv=conversation)
    else:
        prompt_ab = RELATIONSHIP_ASSESS_PROMPT_EN.format(src=agent_a['name'], dst=agent_b['name'], conv=conversation)
        prompt_ba = RELATIONSHIP_ASSESS_PROMPT_EN.format(src=agent_b['name'], dst=agent_a['name'], conv=conversation)

    # Force strict JSON (Gemini 2.5 thinking対応で2000トークンに増加)
    rel_ab = run_json_trials(prompt_ab, model='chatgpt', num_tokens_request=2000)
    rel_ba = run_json_trials(prompt_ba, model='chatgpt', num_tokens_request=2000)

    if not isinstance(rel_ab, dict):
        logging.warning(f"Relationship A->B not a dict: {type(rel_ab)}. Using defaults.")
        rel_ab = {}
    if not isinstance(rel_ba, dict):
        logging.warning(f"Relationship B->A not a dict: {type(rel_ba)}. Using defaults.")
        rel_ba = {}

    rels = {
        'a_to_b': _ensure_relationship_schema(rel_ab),
        'b_to_a': _ensure_relationship_schema(rel_ba),
        'by_speaker': {
            agent_a['name']: {
                'toward': agent_b['name'],
                'scores': _ensure_relationship_schema(rel_ab)
            },
            agent_b['name']: {
                'toward': agent_a['name'],
                'scores': _ensure_relationship_schema(rel_ba)
            }
        }
    }
    return rels


def get_recent_context(agent_a, agent_b, sess_id, context_length=2, reflection=False):

    speaker_1_facts = []
    for i in range(1, sess_id):
        for item in agent_a['session_%s_facts' % i][agent_a["name"]]:
            if isinstance(item, list) and len(item) > 0:
                fact_txt = item[0]
            elif isinstance(item, str):
                fact_txt = item
            else:
                fact_txt = str(item)
            speaker_1_facts.append(agent_a['session_%s_date_time' % i] + ': ' + fact_txt)
    speaker_2_facts = []
    for i in range(1, sess_id):
        for item in agent_a['session_%s_facts' % i][agent_b["name"]]:
            if isinstance(item, list) and len(item) > 0:
                fact_txt = item[0]
            elif isinstance(item, str):
                fact_txt = item
            else:
                fact_txt = str(item)
            speaker_2_facts.append(agent_a['session_%s_date_time' % i] + ': ' + fact_txt)
    
    if reflection:
        print(speaker_1_facts[-context_length:])
        print(agent_a['session_%s_reflection' % (sess_id-1)]['self'])
        return speaker_1_facts[-context_length:] + agent_a['session_%s_reflection' % (sess_id-1)]['self'], speaker_2_facts[-context_length:] + agent_a['session_%s_reflection' % (sess_id-1)]['other']
    else:
        return speaker_1_facts[-context_length:], speaker_2_facts[-context_length:]


def get_relevant_context(agent_a, agent_b, input_dialogue, embeddings, sess_id, context_length=2, reflection=False):

    logging.info("Getting relevant context for response to %s (session %s)" % (input_dialogue, sess_id))
    contexts_a, context_b = get_recent_context(agent_a, agent_b, sess_id, 10)
    # embeddings = pkl.load(open(emb_file, 'rb'))
    input_embedding = get_embedding([input_dialogue])
    sims_with_context_a = np.dot(embeddings[agent_a['name']], input_embedding[0])
    sims_with_context_b = np.dot(embeddings[agent_b['name']], input_embedding[0])
    top_k_sims_a = np.argsort(sims_with_context_a)[::-1][:context_length]
    top_k_sims_b = np.argsort(sims_with_context_b)[::-1][:context_length]
    # print(sims_with_context_a, sims_with_context_b)
    if reflection:
        print([contexts_a[idx] for idx in top_k_sims_a])
        print( agent_a['session_%s_reflection' % (sess_id-1)]['self'])
        return [contexts_a[idx] for idx in top_k_sims_a] + random.sample(agent_a['session_%s_reflection' % (sess_id-1)]['self'], k=context_length//2), [context_b[idx] for idx in top_k_sims_b] + random.sample(agent_a['session_%s_reflection' % (sess_id-1)]['other'], k=context_length//2)
    else:
        return [contexts_a[idx] for idx in top_k_sims_a], [context_b[idx] for idx in top_k_sims_b]
