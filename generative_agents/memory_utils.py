import os, json
import time
import logging
from datetime import datetime
from global_methods import run_json_trials, get_openai_embedding
import numpy as np
import pickle as pkl
import random
import unicodedata
import difflib
logging.basicConfig(level=logging.INFO)


# Reflection prompts (EN/JA)
REFLECTION_INIT_PROMPT_EN = "{}\n\nGiven the information above, what are the three most salient insights that {} has about {}? Give concise answers in the form of a json list where each entry is a string."

REFLECTION_CONTINUE_PROMPT_EN = "{} has the following insights about {} from previous interactions.{}\n\nTheir next conversation is as follows:\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about {} now? Give concise answers in the form of a json list where each entry is a string."

SELF_REFLECTION_INIT_PROMPT_EN = "{}\n\nGiven the information above, what are the three most salient insights that {} has about self? Give concise answers in the form of a json list where each entry is a string."

SELF_REFLECTION_CONTINUE_PROMPT_EN = "{} has the following insights about self.{}\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about self now? Give concise answers in the form of a json list where each entry is a string."

#内省時プロンプト
REFLECTION_INIT_PROMPT_JA = (
    "{}\n\n上の情報に基づき、{} が {} について持っている最も重要な洞察を3つ挙げてください。"
    "各項目は短い日本語の文とし、JSON 配列（各要素は文字列）だけを返してください。"
)

REFLECTION_CONTINUE_PROMPT_JA = (
    "{} はこれまでのやり取りから {} について次の洞察を持っています。{}\n\n"
    "次の会話は以下です:\n\n{}\n\n"
    "上の情報に基づき、{} が {} について今持っている最も重要な洞察を3つ挙げてください。"
    "各項目は短い日本語の文とし、JSON 配列（各要素は文字列）のみを返してください。"
)

SELF_REFLECTION_INIT_PROMPT_JA = (
    "{}\n\n上の情報に基づき、{} が自分自身について持っている最も重要な洞察を3つ挙げてください。"
    "各項目は短い日本語の文とし、JSON 配列（各要素は文字列）のみを返してください。"
)

SELF_REFLECTION_CONTINUE_PROMPT_JA = (
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
    "- power: 力関係（{src} が {dst} に対して自分がより優位/影響力があると感じる度合い。高いほど {src} が優位と感じる）\n"
    "- social_distance: 社会的距離（形式ばった距離感。高いほど距離がある）\n"
    "- trust: 信頼（{src} が {dst} をどれだけ信頼しているか）\n\n"
    "出力は次のキーを持つJSONオブジェクトのみ（英語キー名を厳守）: \n"
    "{{\"intimacy\":<1-10>,\"power\":<1-10>,\"social_distance\":<1-10>,\"trust\":<1-10>}} \n"
    "JSON以外の文章は書かないでください。\n\n"
    "CONVERSATION（1行目は日付を含む）:\n{conv}\n"
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
    # 以前は 500 トークンだったが JSON が途中で途切れる事例が多発したため余裕を持って拡張
    facts = run_json_trials(query, num_gen=1, num_tokens_request=900, use_16k=False, examples=examples, input=input)

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
def get_session_reflection(args, agent_a, agent_b, session_idx):


    # Step 1: get conversation
    conversation = ""
    conversation += agent_a['session_%s_date_time' % session_idx] + '\n'
    for dialog in agent_a['session_%s' % session_idx]:
        # if 'clean_text' in dialog:
        #     writer.write(dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"\n')
        # else:
        conversation += dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"\n'


    # Step 2: Self-reflections
    lang = getattr(args, 'lang', 'en')
    if session_idx == 1:
        if lang == 'ja':
            prompt_a = SELF_REFLECTION_INIT_PROMPT_JA.format(conversation, agent_a['name'])
            prompt_b = SELF_REFLECTION_INIT_PROMPT_JA.format(conversation, agent_b['name'])
        else:
            prompt_a = SELF_REFLECTION_INIT_PROMPT_EN.format(conversation, agent_a['name'])
            prompt_b = SELF_REFLECTION_INIT_PROMPT_EN.format(conversation, agent_b['name'])
        agent_a_self = run_json_trials(prompt_a, model='chatgpt', num_tokens_request=300)
        agent_b_self = run_json_trials(prompt_b, model='chatgpt', num_tokens_request=300)

    else:
        if lang == 'ja':
            prompt_a = SELF_REFLECTION_CONTINUE_PROMPT_JA.format(
                agent_a['name'], '\n'.join(agent_a['session_%s_reflection' % (session_idx-1)]['self']), conversation, agent_a['name']
            )
            prompt_b = SELF_REFLECTION_CONTINUE_PROMPT_JA.format(
                agent_b['name'], '\n'.join(agent_b['session_%s_reflection' % (session_idx-1)]['self']), conversation, agent_b['name']
            )
        else:
            prompt_a = SELF_REFLECTION_CONTINUE_PROMPT_EN.format(
                agent_a['name'], '\n'.join(agent_a['session_%s_reflection' % (session_idx-1)]['self']), conversation, agent_a['name']
            )
            prompt_b = SELF_REFLECTION_CONTINUE_PROMPT_EN.format(
                agent_b['name'], '\n'.join(agent_b['session_%s_reflection' % (session_idx-1)]['self']), conversation, agent_b['name']
            )
        agent_a_self = run_json_trials(prompt_a, model='chatgpt', num_tokens_request=300)
        agent_b_self = run_json_trials(prompt_b, model='chatgpt', num_tokens_request=300)

    # Step 3: Reflection about other speaker
    if session_idx == 1:
        if lang == 'ja':
            prompt_ab = REFLECTION_INIT_PROMPT_JA.format(conversation, agent_a['name'], agent_b['name'])
            prompt_ba = REFLECTION_INIT_PROMPT_JA.format(conversation, agent_b['name'], agent_a['name'])
        else:
            prompt_ab = REFLECTION_INIT_PROMPT_EN.format(conversation, agent_a['name'], agent_b['name'])
            prompt_ba = REFLECTION_INIT_PROMPT_EN.format(conversation, agent_b['name'], agent_a['name'])
        agent_a_on_b = run_json_trials(prompt_ab, model='chatgpt', num_tokens_request=300)
        agent_b_on_a = run_json_trials(prompt_ba, model='chatgpt', num_tokens_request=300)

    else:
        if lang == 'ja':
            prompt_ab = REFLECTION_CONTINUE_PROMPT_JA.format(
                agent_a['name'], agent_b['name'], '\n'.join(agent_a['session_%s_reflection' % (session_idx-1)]['other']), conversation, agent_a['name'], agent_b['name']
            )
            prompt_ba = REFLECTION_CONTINUE_PROMPT_JA.format(
                agent_b['name'], agent_a['name'], '\n'.join(agent_b['session_%s_reflection' % (session_idx-1)]['other']), conversation, agent_b['name'], agent_a['name']
            )
        else:
            prompt_ab = REFLECTION_CONTINUE_PROMPT_EN.format(
                agent_a['name'], agent_b['name'], '\n'.join(agent_a['session_%s_reflection' % (session_idx-1)]['other']), conversation, agent_a['name'], agent_b['name']
            )
            prompt_ba = REFLECTION_CONTINUE_PROMPT_EN.format(
                agent_b['name'], agent_a['name'], '\n'.join(agent_b['session_%s_reflection' % (session_idx-1)]['other']), conversation, agent_b['name'], agent_a['name']
            )
        agent_a_on_b = run_json_trials(prompt_ab, model='chatgpt', num_tokens_request=300)
        agent_b_on_a = run_json_trials(prompt_ba, model='chatgpt', num_tokens_request=300)

    if type(agent_a_self) == dict:
        agent_a_self = list(agent_a_self.values())
    if type(agent_b_self) == dict:
        agent_b_self = list(agent_b_self.values())
    if type(agent_a_on_b) == dict:
        agent_a_on_b = list(agent_a_on_b.values())
    if type(agent_b_on_a) == dict:
        agent_b_on_a = list(agent_b_on_a.values())  

    reflections = {}
    reflections['a'] = {'self': agent_a_self, 'other': agent_a_on_b}
    reflections['b'] = {'self': agent_b_self, 'other': agent_b_on_a}

    return reflections


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

    # Force strict JSON
    rel_ab = run_json_trials(prompt_ab, model='chatgpt', num_tokens_request=180)
    rel_ba = run_json_trials(prompt_ba, model='chatgpt', num_tokens_request=180)

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
    contexts_a, context_b = get_recent_context(agent_a, agent_b, sess_id, 10000)
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
