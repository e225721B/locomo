import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import logging
import argparse
import os, json, sys
import random
import pickle as pkl
from datetime import date, timedelta, datetime
from generative_agents.conversation_utils import *
from generative_agents.html_utils import convert_to_chat_html
from generative_agents.memory_utils import *
from generative_agents.memory_stream import MemoryStore
# イベントグラフ機能削除: event_utils 依存を除去
# from generative_agents.event_utils import *
# 画像系（BLIP, PIL）使用を停止（オプションで完全削除）
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image
from global_methods import run_chatgpt, run_chatgpt_with_examples, set_openai_key
# import torch  # 画像キャプション無効化のため未使用

logging.basicConfig(level=logging.INFO)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--out-dir', required=True, type=str, help="Path to directory containing agent files and downloaded images for a conversation")  # 出力ディレクトリ（エージェントJSON/HTML等の書き出し先）
    parser.add_argument('--prompt-dir', required=True, type=str, help="Path to the dirctory containing in-context examples")  # プロンプト例（in-context）を置くディレクトリ
    
    parser.add_argument('--start-session', type=int, default=1, help="Start iterating from this index; first session is 1")  # どのセッション番号から開始するか（1始まり）
    parser.add_argument('--num-sessions', type=int, default=20, help="Maximum number of sessions in the conversation")  # 生成するセッション数（最大）
    parser.add_argument('--num-days', type=int, default=240, help="Desired temporal span of the multi-session conversation")  # 会話の期間（日数・目安）
    parser.add_argument('--num-events', type=int, default=15, help="Total number of events to generate for each agent; 1 per session works best")  # 互換: 各エージェントのイベント総数（現在は未使用）
    parser.add_argument('--max-turns-per-session', type=int, default=20, help="Maximum number of total turns in each session")  # セッションあたりの最大ターン数
    parser.add_argument('--num-events-per-session', type=int, default=50, help="Total number of events to be assigned to each agent per session; 1-2 works best")  # 互換: セッションごとのイベント割当数（現在は未使用）

    parser.add_argument('--persona', action="store_true", help="Set flag to sample a new persona from MSC and generate details (ignored if --persona-a/--persona-b are provided)")  # MSCから新規ペルソナを生成（--persona-a/-b があれば無視）
    parser.add_argument('--persona-a', type=str, default=None, help="Path to user-provided persona for Agent A (json/txt). If set, uses this instead of MSC")  # ユーザー指定のペルソナA（json/txt）
    parser.add_argument('--persona-b', type=str, default=None, help="Path to user-provided persona for Agent B (json/txt). If set, uses this instead of MSC")  # ユーザー指定のペルソナB（json/txt）
    parser.add_argument('--session', action="store_true", help="Set flag to generate sessions based on the generated/existing personas")  # ペルソナに基づき会話セッションを生成
    # イベント生成および画像キャプション関連フラグを無効化（後方互換のため受け取るが無視）
    parser.add_argument('--events', action="store_true", help="(Disabled) Event graph generation removed")  # 無効: 旧イベント生成の互換フラグ
    parser.add_argument('--blip-caption', action="store_true", help="(Disabled) BLIP captioning removed")  # 無効: 旧BLIPキャプション
    parser.add_argument('--overwrite-persona', action='store_true', help="Overwrite existing persona summaries saved in the agent files")  # 既存ペルソナを上書き再生成
    parser.add_argument('--overwrite-events', action='store_true', help="Overwrite existing events saved in the agent files")  # 既存イベントを上書き（機能自体は無効）
    parser.add_argument('--overwrite-session', action='store_true', help="Overwrite existing sessions saved in the agent files")  # 既存セッションを上書き再生成
    parser.add_argument('--summary', action="store_true", help="Set flag to generate and use summaries in the conversation generation prompt")  # セッション要約を生成・利用
    parser.add_argument('--facts', action='store_true', help='(Optional) Extract per-session facts for each agent and save/use them. 既定は無効')  # セッション毎のFacts抽出（既定無効）

    parser.add_argument('--emb-file', type=str, default='embeddings.pkl', help="Name of the file used to save embeddings for the fine-grained retrieval-based memory module")  # 埋め込み保存ファイル名
    parser.add_argument('--reflection', action="store_true", help="Set flag to use reflection module at the end of each session and include in the conversation generation prompt for context")  # セッション末に内省を実行
    parser.add_argument('--reflection-importance-threshold', type=int, default=150, help='(Experimental) Sum of recent importance scores above which an early reflection is triggered during a session. 0/負値で無効')  # 早期内省の重要度合計閾値（<=0で無効）
    parser.add_argument('--reflection-every-turn', action='store_true', help='(Experimental) If set with --reflection, perform reflection after every turn (overwrites previous reflection). Disables importance threshold early reflection logic.')  # 毎ターン内省（閾値ロジック無効）
    parser.add_argument('--relationships', action='store_true', help='Evaluate relationship scores (intimacy, power, social_distance, trust) each session for both directions')  # セッションごとに関係スコアを評価
    parser.add_argument('--intra-relationships', action='store_true', help='Evaluate relationship scores every N turns during a session and pass them into the agent prompt')  # セッション中に一定間隔で関係スコアを評価
    parser.add_argument('--intra-frequency', type=int, default=5, help='Number of turns between intra-session relationship evaluations (default 5)')  # 上記の評価間隔（ターン数）
    # Memory stream options
    parser.add_argument('--memory-stream', action='store_true', help='Enable memory stream: store facts/reflections and retrieve top-K memories each turn')  # 記憶を保存し各ターンで上位K件をリトリーブ
    parser.add_argument('--memory-topk', type=int, default=5, help='Top-K memories to retrieve per turn when memory stream is enabled')  # リトリーブ件数K
    parser.add_argument('--min-turns-before-stop', type=int, default=6, help='Do not include stop instruction until at least this turn index (0-based)')  # 終了指示[END]を入れ始める最小ターン
    parser.add_argument('--exact-turns-per-session', type=int, default=None, help='この値が指定された場合、セッションはちょうどNターン生成される。N-1ターンまでは[END]を出力させない（ストップ指示無効化）')  # ちょうどNターン生成
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda','mps'], help='Device for BLIP model (auto: prefer cuda > mps > cpu)')  # 無効: 画像系モデルのデバイス
    parser.add_argument('--image-search', action='store_true', help='(Disabled) image retrieval removed')  # 無効: 旧画像検索
    parser.add_argument('--lang', type=str, default='en', choices=['en','ja'], help='Conversation language (default en)')  # 会話の出力言語

    args = parser.parse_args()
    return args


# 画像キャプション機能は削除
def get_blip_caption(*args, **kwargs):
    raise RuntimeError("BLIP captioning disabled")

def _load_persona_from_file(path: str, default_name: str):
    """ユーザー提供ファイルから persona を読み込む。
    - JSON の場合: {"name": ..., "persona_summary": ...} または {"name": ..., "persona": ...}
    - テキストの場合: 先頭行に 'name: xxx' があれば採用。残り全文を persona_summary とする。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Persona file not found: {path}")
    if p.suffix.lower() == '.json':
        data = json.load(open(p))
        name = data.get('name', default_name)
        persona_summary = data.get('persona_summary') or data.get('persona') or ''
        if not persona_summary:
            # 他キーに入っている場合のゆるい対応
            for k, v in data.items():
                if isinstance(v, str) and len(v) > 10 and 'persona' in k.lower():
                    persona_summary = v
                    break
        if not persona_summary:
            raise ValueError(f"JSON does not contain 'persona' or 'persona_summary': {path}")
        return { 'name': name, 'persona_summary': persona_summary }
    else:
        # テキスト系
        txt = open(p, 'r', encoding='utf-8').read().strip()
        name = default_name
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if lines:
            head = lines[0]
            if head.lower().startswith('name:'):
                name = head.split(':', 1)[1].strip() or default_name
                persona_summary = '\n'.join(lines[1:]).strip()
            else:
                persona_summary = txt
        else:
            persona_summary = ''
        if not persona_summary:
            raise ValueError(f"Empty persona text in: {path}")
        return { 'name': name, 'persona_summary': persona_summary }

def get_user_persona(args):
    """ユーザー指定のファイルから Agent A/B のペルソナを構築する。
    どちらか一方のみ指定された場合、もう一方は既存ファイルを尊重（なければ MSC 生成にフォールバック）。
    戻り値: (agent_a or None, agent_b or None)
    """
    agent_a = None
    agent_b = None
    if args.persona_a:
        agent_a = _load_persona_from_file(args.persona_a, default_name='Agent A')
    if args.persona_b:
        agent_b = _load_persona_from_file(args.persona_b, default_name='Agent B')

    # 追加: デフォルトのペルソナ配置ディレクトリからの自動読込
    # 優先順位: 明示指定 > 既定ディレクトリ（LOCOMO_PERSONA_DIR or 固定パス）
    default_dir = os.environ.get('LOCOMO_PERSONA_DIR', '/Users/tomokai/locomo/locomo/personas')
    if (agent_a is None or agent_b is None) and default_dir and os.path.isdir(default_dir):
        def _try_resolve(side: str):
            # 探索候補（上から順に優先）
            candidates = [
                f'agent_{side}.json', f'agent_{side}.txt', f'{side}.json', f'{side}.txt',
            ]
            for name in candidates:
                p = Path(default_dir) / name
                if p.exists():
                    return str(p)
            return None
        if agent_a is None:
            fp = _try_resolve('a')
            if fp:
                agent_a = _load_persona_from_file(fp, default_name='Agent A')
        if agent_b is None:
            fp = _try_resolve('b')
            if fp:
                agent_b = _load_persona_from_file(fp, default_name='Agent B')
    return agent_a, agent_b


def save_agents(agents, args):

    agent_a, agent_b = agents
    with open(args.agent_a_file, 'w') as f:
        json.dump(agent_a, f, indent=2)
    with open(args.agent_b_file, 'w') as f:
        json.dump(agent_b, f, indent=2)


def load_agents(args):

    agent_a = json.load(open(args.agent_a_file))
    agent_b = json.load(open(args.agent_b_file))
    return agent_a, agent_b


def get_random_time():

    start_time = timedelta(hours=9, minutes=0, seconds=0)
    end_time = timedelta(hours=21, minutes=59, seconds=59)
    random_seconds = random.randint(start_time.total_seconds(), end_time.total_seconds())
    hours = random_seconds//3600
    minutes = (random_seconds - (hours*3600))//60
    return timedelta(hours=hours, minutes=minutes, seconds=0)


## 時刻付きフォーマットは不要になったため削除。互換のためシンプルな日付文字列のみ利用。
def datetimeStr2Obj(dateStr):
    """旧フォーマット互換用: 日付だけを '%d %B, %Y' / '%d %B %Y' として解釈"""
    try:
        return datetime.strptime(dateStr, "%d %B, %Y")
    except:
        return datetime.strptime(dateStr, "%d %B %Y")

def datetimeObj2Str(datetimeObj):
    """date/datetime から日付のみの文字列を返す"""
    if isinstance(datetimeObj, date) and not isinstance(datetimeObj, datetime):
        return datetimeObj.strftime("%d %B, %Y")
    return datetimeObj.strftime("%d %B, %Y")


def dateObj2Str(dateObj):
    return dateObj.strftime("%d") + ' ' + dateObj.strftime("%B") + ', ' + dateObj.strftime("%Y")


def get_random_date():

    # initializing dates ranges
    test_date1, test_date2 = date(2022, 1, 1), date(2023, 6, 1)
    # getting days between dates
    dates_bet = test_date2 - test_date1
    total_days = dates_bet.days
    delta_days = random.choice(range(1, total_days))
    random_date = test_date1 + timedelta(days=int(delta_days))
    return random_date


def get_session_summary(session, speaker_1, speaker_2, curr_date, previous_summary=""):

    session_query = ''
    for c in session:
        session_query += "%s: %s\n" % (c["speaker"], c["text"])
        if "image" in c:
            session_query += "[%s shares %s]\n" % (c["speaker"], c["image"])

    if previous_summary:

        query = SESSION_SUMMARY_PROMPT % (speaker_1['name'], speaker_2['name'], previous_summary, curr_date,
                                               speaker_1['name'], speaker_2['name'], session_query, speaker_1['name'], speaker_2['name'])
    else:
        query = SESSION_SUMMARY_INIT_PROMPT % (speaker_1['name'], speaker_2['name'], curr_date, session_query)

    query += '\n\n'
    # should summarize persona, previous conversations with respect to speaker.
    if getattr(speaker_1, 'lang', None):
        pass  # placeholder (speaker dict does not hold lang)
    # 言語指定は run 時の args に存在するので後段 main から渡せないため、簡易判定: 日本語指示文を末尾に追加できるようフラグ環境利用も可
    if os.environ.get('LOCOMO_LANG') == 'ja':
        query += '\n出力は日本語で 150 語以内で要約してください。'
    output = run_chatgpt(query, 1, 150, 'chatgpt')
    output = output.strip()
    return output


def get_image_queries(events):

    images = [e["image"] for e in events]
    input_query = "\nInput: ".join(images)

    output = run_chatgpt(EVENT2QUERY_PROMPT % input_query, 1, 200, 'chatgpt')
    output = output.strip()
    print(output)
    json_output = clean_json_output(output)

    assert len(events) == len(json_output), [events, json_output]

    for i in range(len(events)):
        events[i]["query"] = json_output[i]
    return events


def get_all_session_summary(speaker, curr_sess_id):

    summary = "\n"
    for sess_id in range(1, curr_sess_id):
        sess_date = speaker['session_%s_date_time' % sess_id]
        sess_date = sess_date[2] + ' ' + sess_date[1] + ', ' + sess_date[0]
        # サマリーが無効化されている場合（--summary 未指定）でも安全にスキップ
        if ("session_%s_summary" % sess_id) in speaker:
            summary += sess_date + ': ' + speaker["session_%s_summary" % sess_id] + '\n'
    return summary


def catch_date(date_str):
    date_format1 = '%d %B, %Y'
    date_format2 = '%d %B %Y'
    try:
        return datetime.strptime(date_str, date_format1)
    except:
        return datetime.strptime(date_str, date_format2)


## イベント関連関数は削除（イベントグラフ非使用）


def get_event_string(*args, **kwargs):
    return ""  # 互換用ダミー


def remove_context(args, curr_dialog, prev_dialog, caption=None):

    prompt_data = json.load(open(os.path.join(args.prompt_dir, 'remove_context_examples.json')))
    if caption:
        query = prompt_data["input_format_w_image"].format(prev_dialog, curr_dialog, caption)
    else:
        query = prompt_data["input_format"].format(prev_dialog, curr_dialog)
    output = run_chatgpt_with_examples(prompt_data["prompt"], 
                              [[prompt_data["input_format"].format(*example["input"]) if len(example["input"]) == 2 else prompt_data["input_format_w_image"].format(*example["input"]), example["output"]] for example in prompt_data['examples']], 
                              query, num_gen=1, num_tokens_request=128, use_16k=False)
    return output

#毎ターンごとにエージェントに渡す情報
def get_agent_query(speaker_1, speaker_2, curr_sess_id=0,
                    prev_sess_date_time='', curr_sess_date_time='',
                    use_events=False, instruct_stop=False, dialog_id=0, last_dialog='', embeddings=None, reflection=False, language='en', relationships=None, memory_snippet=None):

    stop_instruction = "To end the conversation, write [END] at the end of the dialog."
    if instruct_stop:
        print("**** #[END]を渡すので会話が終了しやすくなるよ ****")

    if curr_sess_id == 1:
        
        if use_events:
            events = get_event_string(speaker_1['events_session_%s' % curr_sess_id], speaker_1['graph'])
            query = AGENT_CONV_PROMPT_SESS_1_W_EVENTS % (speaker_1['persona_summary'],
                    speaker_1['name'], speaker_2['name'], 
                    curr_sess_date_time, speaker_1['name'],  events, speaker_1['name'], speaker_2['name'], stop_instruction if instruct_stop else '')
        else:
            query = AGENT_CONV_PROMPT_SESS_1 % (speaker_1['persona_summary'],
                                speaker_1['name'], speaker_2['name'], 
                                curr_sess_date_time, speaker_1['name'],  speaker_2['name'], speaker_1['name'])
    
    else:
        if use_events:
            events = get_event_string(speaker_1['events_session_%s' % curr_sess_id], speaker_1['graph'])
            if dialog_id == 0:
                # if a new session is starting, get information about the topics discussed in last session
                context_from_1, context_from_2 = get_recent_context(speaker_1, speaker_2, curr_sess_id, reflection=reflection)
                recent_context = '\n'.join(context_from_1) + '\n' +  '\n'.join(context_from_2) # with reflection
                query = AGENT_CONV_PROMPT_W_EVENTS_V2_INIT % (speaker_1['persona_summary'],
                            speaker_1['name'], speaker_2['name'], prev_sess_date_time,
                            curr_sess_date_time, speaker_1['name'],  speaker_1['session_%s_summary' % (curr_sess_id-1)], events, stop_instruction if instruct_stop else '', speaker_2['name'])
                
            else:
                # during an ongoing session, get fine-grained information from a previous session using retriever modules
                past_context = get_relevant_context(speaker_1, speaker_2, last_dialog, embeddings, curr_sess_id, reflection=reflection)
                query = AGENT_CONV_PROMPT_W_EVENTS_V2 % (speaker_1['persona_summary'],
                            speaker_1['name'], speaker_2['name'], prev_sess_date_time,
                            curr_sess_date_time, speaker_1['name'], speaker_1['session_%s_summary' % (curr_sess_id-1)], events, past_context, stop_instruction if instruct_stop else '', speaker_2['name'])
        else:
            summary = get_all_session_summary(speaker_1, curr_sess_id)
            query = AGENT_CONV_PROMPT % (speaker_1['persona_summary'],
                                        speaker_1['name'], speaker_2['name'], prev_sess_date_time, summary,
                                        curr_sess_date_time, speaker_1['name'],  speaker_2['name'], speaker_1['name']) 
    
    if language == 'ja':
        # 会話を日本語で行う指示を最後に追加（人物名やイベント文字列は原文保持）
        query += "\n\n出力は必ず自然でカジュアルな日本語で、1発話のみ。括弧内の [END] 指示があればそのトークンも含めて出力の末尾に付与してください。英語は使わないでください。"
        # メタ応答防止: 「〜と聞かれたことに対する返信」のような説明文を禁止
        query += "\nメタな説明（例:『〜と聞かれたことに対する返信』）は書かず、自然な発話の本文だけを出力してください。"
    # If memory snippet is provided (memory stream retrieval), append it
    if memory_snippet:
        try:
            query += "\n\n" + memory_snippet
        except Exception:
            pass
    # If a relationships snapshot is provided, append a short human-readable summary to the prompt
    if relationships:
        try:
            # prefer by_speaker mapping when available
            a_to_b = None
            b_to_a = None
            if isinstance(relationships, dict) and 'by_speaker' in relationships and speaker_1['name'] in relationships['by_speaker']:
                a_to_b = relationships['by_speaker'].get(speaker_1['name'], {}).get('scores')
                b_to_a = relationships['by_speaker'].get(speaker_2['name'], {}).get('scores')
            else:
                a_to_b = relationships.get('a_to_b') if isinstance(relationships, dict) else None
                b_to_a = relationships.get('b_to_a') if isinstance(relationships, dict) else None

            def _fmt_scores(scores):
                if not scores:
                    return 'intimacy=?, power=?, social_distance=?, trust=?'
                return 'intimacy=%s, power=%s, social_distance=%s, trust=%s' % (
                    scores.get('intimacy', '?'), scores.get('power', '?'), scores.get('social_distance', '?'), scores.get('trust', '?'))

            rel_snip = '\n\nRelationship snapshot:\n'
            rel_snip += f"{speaker_1['name']} -> {speaker_2['name']}: " + _fmt_scores(a_to_b) + '\n'
            rel_snip += f"{speaker_2['name']} -> {speaker_1['name']}: " + _fmt_scores(b_to_a) + '\n'
            rel_snip += 'Use these values to guide tone and formality (higher intimacy -> more informal; higher power -> more commanding).\n'
            query += rel_snip
        except Exception:
            pass
    return query


def get_session(agent_a, agent_b, args, prev_date_time_string='', curr_date_time_string='', curr_sess_id=0, captioner=None, img_processor=None, reflection=False):
    
    # load embeddings for retrieveing relevat observations from previous conversations
    if curr_sess_id == 1:
        embeddings = None
    else:
        embeddings = pkl.load(open(args.emb_file, 'rb'))

    # select one of the speakers to start the session at random
    curr_speaker = -1
    if random.random() < 0.5:
        conv_so_far = agent_a['name'] + ': '
        curr_speaker = 0
    else:
        conv_so_far = agent_b['name'] + ': '
        curr_speaker = 1

    session = []
    # --- Dynamic reflection trigger state (per-agent) ---
    early_reflection_done = False
    RECENT_WINDOW_SIZE = 30  # per-agent recent window size
    # Load or initialize per-agent importance stats (persist across sessions)
    stats_a = agent_a.get('importance_stats', {
        'cumulative_total': 0,
        'recent_window': [],
        'early_reflections': 0
    })
    stats_b = agent_b.get('importance_stats', {
        'cumulative_total': 0,
        'recent_window': [],
        'early_reflections': 0
    })
    # current in-session relationship snapshot (updated every N turns if enabled)
    current_relationships = None
    # memory stores (loaded once per session)
    mem_a = MemoryStore(agent_a, lang=args.lang) if args.memory_stream else None
    mem_b = MemoryStore(agent_b, lang=args.lang) if args.memory_stream else None
    # ログ: 毎ターン内省モードが有効な場合はセッション開始時に告知
    if args.reflection and args.reflection_every_turn:
        logging.info(f"[reflection] 毎ターン内省モードを有効化 (session={curr_sess_id})")

    # メモリリトリーブの可視化用ヘルパ
    def _log_retrieval(speaker_name: str, turn_idx: int, entries):
        try:
            for rank, e in enumerate(entries or [], 1):
                txt = e.get('text', '') if isinstance(e, dict) else ''
                if isinstance(txt, str) and len(txt) > 80:
                    txt = txt[:77] + '...'
                src = (e.get('source_type') if isinstance(e, dict) else None) or (e.get('source') if isinstance(e, dict) else None)
                score = 0.0
                try:
                    score = float((e.get('retrieval_score') if isinstance(e, dict) else 0.0) or 0.0)
                except Exception:
                    score = 0.0
                logging.info(
                    f"[memory-topk] turn={turn_idx} speaker={speaker_name} "
                    f"rank={rank} src={src} "
                    f"imp={e.get('importance') if isinstance(e, dict) else None} score={round(score,3)} text={txt}"
                )
        except Exception as _e:
            logging.debug(f"memory-topk logging failed: {_e}")
    
    # Choose a turn index to start including stop instruction.
    # exact-turns-per-session が指定されたら、その最終ターンのみ終了許可。
    if args.exact_turns_per_session is not None:
        stop_dialog_count = max(args.exact_turns_per_session - 1, 1)
    else:
        stop_dialog_count = args.max_turns_per_session if args.max_turns_per_session <= 10 else random.choice(list(range(10, args.max_turns_per_session)))
    break_at_next_a = False
    break_at_next_b = False
    for i in range(args.max_turns_per_session):
        if break_at_next_a and break_at_next_b:
            break

        # 各ターンでのリトリーブ結果（テキストのみ）を収集するための一時領域
        retrieved_texts_for_turn = []

        # Before generating a turn, if intra-relationships is enabled and it's time, evaluate provisional relationships
        if args.intra_relationships and i > 0 and (i % args.intra_frequency) == 0:
            try:
                # pass the partial session so far for mid-session evaluation
                rels = get_session_relationships(args, agent_a, agent_b, curr_sess_id, session_dialog=session, session_date=agent_a.get('session_%s_date_time' % curr_sess_id))
                # save provisional relationships into agent records so exports will include intermediate snapshots
                agent_a['session_%s_relationships' % curr_sess_id] = rels
                agent_b['session_%s_relationships' % curr_sess_id] = rels
                current_relationships = rels
                save_agents([agent_a, agent_b], args)
                logging.info(f"[intra-relationships] Session {curr_sess_id} turn {i}: provisional relationships updated")
            except Exception as e:
                logging.warning(f"Failed to compute intra-session relationships at turn {i}: {e}")

        # --- Pre-turn reflection: the listener of the previous utterance (i>0) reflects before replying ---
        if args.reflection and args.reflection_every_turn and i > 0:
            try:
                # Save partial session so far for reflection context
                agent_a[f'session_{curr_sess_id}'] = session
                agent_b[f'session_{curr_sess_id}'] = session
                refl_all = get_session_reflection(args, agent_a, agent_b, curr_sess_id)
                if curr_speaker == 0:
                    # Agent A is about to speak; reflect for A only
                    agent_a[f'session_{curr_sess_id}_reflection'] = refl_all['a']
                    # store into A's memory stream for retrieval
                    if args.memory_stream and mem_a is not None:
                        sess_date = agent_a.get('session_%s_date_time' % curr_sess_id)
                        mem_a.add_from_reflections(refl_all['a'].get('self', []) or [], session_date=sess_date, about='self')
                        mem_a.add_from_reflections(refl_all['a'].get('other', []) or [], session_date=sess_date, about='other')
                        mem_a.save(); save_agents([agent_a, agent_b], args)
                    a_self = (refl_all['a'].get('self') or [])
                    a_other = (refl_all['a'].get('other') or [])
                    logging.info(f"（内省者：）{agent_a['name']} reflects -> self:{len(a_self)}, other:{len(a_other)}")
                else:
                    # Agent B is about to speak; reflect for B only
                    agent_b[f'session_{curr_sess_id}_reflection'] = refl_all['b']
                    if args.memory_stream and mem_b is not None:
                        sess_date = agent_b.get('session_%s_date_time' % curr_sess_id)
                        mem_b.add_from_reflections(refl_all['b'].get('self', []) or [], session_date=sess_date, about='self')
                        mem_b.add_from_reflections(refl_all['b'].get('other', []) or [], session_date=sess_date, about='other')
                        mem_b.save(); save_agents([agent_a, agent_b], args)
                    b_self = (refl_all['b'].get('self') or [])
                    b_other = (refl_all['b'].get('other') or [])
                    logging.info(f"（内省者：）{agent_b['name']} reflects -> self:{len(b_self)}, other:{len(b_other)}")
            except Exception as e:
                logging.warning(f"Pre-turn reflection failed at turn {i}: {e}")

        if curr_speaker == 0:
            # Memory retrieval for Agent A's turn
            mem_snip = None
            if args.memory_stream and mem_a is not None:
                #直前の相手の発話をクエリとする
                qtext = '' if i == 0 else (session[-1]['speaker'] + ' says, ' + session[-1]['clean_text'])
                #直前の発話に対してretrieve（上位k件抽出）を行なっている
                top = mem_a.retrieve(qtext or agent_a.get('persona_summary',''), topk=args.memory_topk, now_date=curr_date_time_string)
                _log_retrieval(agent_a.get('name','A'), i, top)
                mem_snip = MemoryStore.format_snippet(top, lang=args.lang)
                # エクスポート用にテキストのみ保持
                try:
                    retrieved_texts_for_turn = [e.get('text') for e in (top or []) if isinstance(e, dict) and e.get('text')]
                except Exception:
                    retrieved_texts_for_turn = []
            agent_query = get_agent_query(
                agent_a, agent_b,
                prev_sess_date_time=prev_date_time_string,
                curr_sess_date_time=curr_date_time_string,
                curr_sess_id=curr_sess_id,
                use_events=False,
                instruct_stop=(i >= stop_dialog_count and i >= args.min_turns_before_stop),
                dialog_id=i,
                last_dialog='' if i == 0 else session[-1]['speaker'] + ' says, ' + session[-1]['clean_text'],
                embeddings=embeddings,
                reflection=reflection,
                language=args.lang,
                relationships=current_relationships,
                memory_snippet=mem_snip
            )
        else:
            mem_snip = None
            if args.memory_stream and mem_b is not None:
                qtext = '' if i == 0 else (session[-1]['speaker'] + ' says, ' + session[-1]['clean_text'])
                top = mem_b.retrieve(qtext or agent_b.get('persona_summary',''), topk=args.memory_topk, now_date=curr_date_time_string)
                _log_retrieval(agent_b.get('name','B'), i, top)
                mem_snip = MemoryStore.format_snippet(top, lang=args.lang)
                # エクスポート用にテキストのみ保持
                try:
                    retrieved_texts_for_turn = [e.get('text') for e in (top or []) if isinstance(e, dict) and e.get('text')]
                except Exception:
                    retrieved_texts_for_turn = []
            agent_query = get_agent_query(
                agent_b, agent_a,
                prev_sess_date_time=prev_date_time_string,
                curr_sess_date_time=curr_date_time_string,
                curr_sess_id=curr_sess_id,
                use_events=False,
                instruct_stop=(i >= stop_dialog_count and i >= args.min_turns_before_stop),
                dialog_id=i,
                last_dialog='' if i == 0 else session[-1]['speaker'] + ' says, ' + session[-1]['clean_text'],
                embeddings=embeddings,
                reflection=reflection,
                language=args.lang,
                relationships=current_relationships,
                memory_snippet=mem_snip
            )

        # 画像関連機能無効化 (placeholder)
        # if args.image_search ...

        # トークン上限を拡大し、複数行が返っても1発話として取り込み（改行→スペース）
        raw = run_chatgpt(agent_query + conv_so_far, 1, 200, 'chatgpt', temperature=1.2)
        raw = ' '.join(raw.strip().splitlines())
        cleaned = clean_dialog(raw, agent_a['name'] if curr_speaker == 0 else agent_b['name'])
        output = {"text": cleaned, "raw_text": cleaned}

        output["speaker"] = agent_a["name"] if curr_speaker == 0 else agent_b['name']
        text_replaced_caption = replace_captions(output["text"], args)
        if not text_replaced_caption.isspace():
            if '[END]' in output["text"]:
                output["clean_text"] = text_replaced_caption
            else:
                if args.lang == 'ja':
                    output["clean_text"] = text_replaced_caption
                else:
                    output["clean_text"] = run_chatgpt(CASUAL_DIALOG_PROMPT % text_replaced_caption, 1, 100, 'chatgpt').strip()
        else:
            output["clean_text"] = ""

        output["dia_id"] = f'D{curr_sess_id}:{i+1}'
        # 各ターンのリトリーブ結果（テキスト配列）を保存（エクスポート用）
        output["retrieved"] = retrieved_texts_for_turn or []
        session.append(output)

        # もし、メモリストリームが有効なら、発話を記憶として追加し、各エージェントの importance を更新
        if args.memory_stream:
            try:
                target_stats = None
                if curr_speaker == 0 and mem_a is not None:
                    m_entry = mem_a.add_memory(text=output["clean_text"], created_at=curr_date_time_string, source_type='conversation')
                    target_stats = stats_a
                elif curr_speaker == 1 and mem_b is not None:
                    m_entry = mem_b.add_memory(text=output["clean_text"], created_at=curr_date_time_string, source_type='conversation')
                    target_stats = stats_b
                else:
                    m_entry = None
                if m_entry and target_stats is not None:
                    imp = int(m_entry.get('importance', 0))
                    target_stats['cumulative_total'] += imp
                    rw = target_stats['recent_window']
                    rw.append(imp)
                    if len(rw) > RECENT_WINDOW_SIZE:
                        rw.pop(0)
                    # ログ: 両エージェントの合計と直近窓サマリ
                    recent_sum_a = sum(stats_a['recent_window'])
                    recent_sum_b = sum(stats_b['recent_window'])
            except Exception as e:
                logging.warning(f"Failed to add dialog memory for dynamic reflection trigger: {e}")

        # Per-turn reflection is handled pre-turn (as listener), so skip here
        if args.reflection and args.reflection_every_turn:
            pass
        else:
            # Early reflection trigger (per-agent recent sums) if per-turn not active
            if (not early_reflection_done and args.reflection and args.reflection_importance_threshold > 0 and i >= 2):
                recent_sum_a = sum(stats_a['recent_window'])
                recent_sum_b = sum(stats_b['recent_window'])
                trigger_agent = None
                if recent_sum_a >= args.reflection_importance_threshold:
                    trigger_agent = 'A'
                elif recent_sum_b >= args.reflection_importance_threshold:
                    trigger_agent = 'B'
                if trigger_agent:
                    try:
                        agent_a[f'session_{curr_sess_id}'] = session
                        agent_b[f'session_{curr_sess_id}'] = session
                        refl = get_session_reflection(args, agent_a, agent_b, curr_sess_id)
                        agent_a[f'session_{curr_sess_id}_reflection'] = refl['a']
                        agent_b[f'session_{curr_sess_id}_reflection'] = refl['b']
                        # 詳細ログ＋トレース
                        a_self = (refl['a'].get('self') or [])
                        a_other = (refl['a'].get('other') or [])
                        b_self = (refl['b'].get('self') or [])
                        b_other = (refl['b'].get('other') or [])
                        logging.info(
                            f"（内省を行います：早期） turn={i} A(self:{len(a_self)}, other:{len(a_other)}) B(self:{len(b_self)}, other:{len(b_other)})"
                        )
                        turns_key = f'session_{curr_sess_id}_reflection_turns'
                        entry = {'turn': i, 'a': refl['a'], 'b': refl['b'], 'type': 'early', 'time': datetime.utcnow().isoformat()}
                        agent_a.setdefault(turns_key, []).append(entry)
                        agent_b.setdefault(turns_key, []).append(entry)
                        try:
                            trace_path = os.path.join(args.out_dir, f'reflection_trace_session_{curr_sess_id}.jsonl')
                            with open(trace_path, 'a', encoding='utf-8') as tf:
                                tf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        except Exception as _e:
                            logging.debug(f"Failed to append reflection trace: {_e}")
                        stats_a['early_reflections'] += 1
                        stats_b['early_reflections'] += 1
                        save_agents([agent_a, agent_b], args)
                    except Exception as e:
                        logging.warning(f"Early reflection failed at turn {i}: {e}")
                    early_reflection_done = True
                    stats_a['recent_window'].clear()
                    stats_b['recent_window'].clear()
                # 永続化のため stats をエージェントに戻す
        # Persist updated stats back into agent dict each turn (lightweight in-memory)
        agent_a['importance_stats'] = stats_a
        agent_b['importance_stats'] = stats_b

        print("############ ", output["speaker"], ': ', output["clean_text"])

        conv_so_far = conv_so_far + output["clean_text"] + '\n'

        # If exact-turns is requested, ignore [END] until the last turn
        if args.exact_turns_per_session is not None and i < args.exact_turns_per_session - 1:
            # ストップトークンはプロンプト継続のため削除
            output['clean_text'] = output['clean_text'].replace('[END]', '').strip()

        # Break 判定は clean_text を基準に、exact-turns 指定時は最終ターンのみ許可
        break_check_text = output['clean_text'] if args.exact_turns_per_session is not None else output['text']
        allow_break_now = True
        if args.exact_turns_per_session is not None:
            allow_break_now = (i >= args.exact_turns_per_session - 1)
        if break_check_text.endswith('[END]') and allow_break_now:
            if curr_speaker == 0:
                break_at_next_a = True
            else:
                break_at_next_b = True

        conv_so_far += f"\n{agent_b['name']}: " if curr_speaker == 0 else f"\n{agent_a['name']}: "
        curr_speaker = int(not curr_speaker)

        # 強制的に exact-turns で終了
        if args.exact_turns_per_session is not None and (i+1) >= args.exact_turns_per_session:
            break

    return session


def main():

    # get arguments
    args = parse_args()

    set_openai_key()
    # lang を環境変数経由で下位関数へ (簡易共有)
    os.environ['LOCOMO_LANG'] = args.lang

    args.emb_file = os.path.join(args.out_dir, args.emb_file)

    # create dataset directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logging.info("Dataset directory: %s" % args.out_dir)

    args.agent_a_file = os.path.join(args.out_dir, 'agent_a.json')
    args.agent_b_file = os.path.join(args.out_dir, 'agent_b.json')

    
    # Step 1: Get personalities for the agents
    # 優先度: ユーザー提供 (--persona-a/--persona-b) > 既存ファイル > MSC 自動生成 (--persona 指定時)
    ua, ub = get_user_persona(args)
    if ua or ub:
        # 既存があれば読み込み、片方のみ更新も可能に
        existing_a, existing_b = None, None
        if os.path.exists(args.agent_a_file) and os.path.exists(args.agent_b_file) and not args.overwrite_persona:
            existing_a, existing_b = load_agents(args)
        agent_a = ua or existing_a
        agent_b = ub or existing_b
        # 足りない側が未定義なら、--persona があれば MSC で補完、なければエラー
        if agent_a is None or agent_b is None:
            if args.persona:
                gen_a, gen_b = get_msc_persona(args)
                agent_a = agent_a or gen_a
                agent_b = agent_b or gen_b
            else:
                missing = 'A' if agent_a is None else 'B'
                raise RuntimeError(f"Persona for Agent {missing} is missing. Provide --persona-{missing.lower()} or use --persona to auto-generate.")
        save_agents([agent_a, agent_b], args)
    elif args.persona:
        # 互換: MSC から自動生成（従来挙動）
        agent_a, agent_b = get_msc_persona(args)
        if agent_a is not None and agent_b is not None:
            save_agents([agent_a, agent_b], args)


    # Step 2: check if events exist; if not, generate event graphs for each of the agents 
    # イベント生成ステップ削除: 既存エージェント JSON に graph があっても無視

    # Step 3: 
    if args.session:

        agent_a, agent_b = load_agents(args)

    # BLIP 初期化削除
    img_processor = None
    captioner = None

        # default start index is 1; if resuming conversation from a leter session, indicate in script arguments using --start-session
    for j in range(args.start_session, args.num_sessions+1):

            print("******************* SESSION %s ******************" % j)

            if 'session_%s' % j not in agent_a or args.overwrite_session:

                # 連続時間セッション: セッション1を基準日とし、以降は1日ずつ進める簡易モデル
                if j>1:
                    prev_date = datetimeStr2Obj(agent_a['session_%s_date_time' % (j-1)])
                    prev_date_time_string = agent_a['session_%s_date_time' % (j-1)]
                    curr_date = (prev_date + timedelta(days=1)).date()
                else:
                    curr_date = get_random_date()
                    prev_date_time_string = None
                curr_date_time_string = datetimeObj2Str(curr_date)
                agent_a['session_%s_date_time' % j] = curr_date_time_string
                agent_b['session_%s_date_time' % j] = curr_date_time_string
                save_agents([agent_a, agent_b], args)
                
                session = get_session(agent_a, agent_b, args,
                                      prev_date_time_string=prev_date_time_string, curr_date_time_string=curr_date_time_string,
                                      curr_sess_id=j, captioner=captioner, img_processor=img_processor, reflection=args.reflection)
                
                agent_a['session_%s' % j] = session
                agent_b['session_%s' % j] = session

                save_agents([agent_a, agent_b], args)

            if args.facts and (('session_%s_facts' % j not in agent_a) or args.overwrite_session):

                facts = get_session_facts(args, agent_a, agent_b, j)

                agent_a['session_%s_facts' % j] = facts
                agent_b['session_%s_facts' % j] = facts

                logging.info("--------- Session %s Facts (extracted) ---------" % (j))
                logging.info(json.dumps(facts, ensure_ascii=False))

                save_agents([agent_a, agent_b], args)
                # --- store facts into memory stream ---
                if args.memory_stream:
                    try:
                        ms_a = MemoryStore(agent_a, lang=args.lang)
                        ms_b = MemoryStore(agent_b, lang=args.lang)
                        sess_date = agent_a.get('session_%s_date_time' % j)
                        ms_a.add_from_facts(agent_a['name'], facts, session_date=sess_date, about='self')
                        ms_a.add_from_facts(agent_b['name'], facts, session_date=sess_date, about='other')
                        ms_b.add_from_facts(agent_b['name'], facts, session_date=sess_date, about='self')
                        ms_b.add_from_facts(agent_a['name'], facts, session_date=sess_date, about='other')
                        ms_a.save(); ms_b.save()
                        save_agents([agent_a, agent_b], args)
                    except Exception as e:
                        logging.warning(f"Failed to add facts to memory stream: {e}")

            if args.reflection and (('session_%s_reflection' % j not in agent_a) or args.overwrite_session):
                # 早期反省(early reflection)が既に生成済みならスキップ（上書き要求がある場合を除く）
                if ('session_%s_reflection' % j in agent_a) and not args.overwrite_session:
                    pass
                else:
                    reflections = get_session_reflection(args, agent_a, agent_b, j)
                    agent_a['session_%s_reflection' % j] = reflections['a']
                    agent_b['session_%s_reflection' % j] = reflections['b']
                    print(" --------- Session %s Reflection for Agent A---------" % (j))
                    print(reflections)
                    # 最終内省もトレース
                    end_entry = {'turn': None, 'a': reflections['a'], 'b': reflections['b'], 'type': 'end', 'time': datetime.utcnow().isoformat()}
                    turns_key = f'session_{j}_reflection_turns'
                    agent_a.setdefault(turns_key, []).append(end_entry)
                    agent_b.setdefault(turns_key, []).append(end_entry)
                    try:
                        trace_path = os.path.join(args.out_dir, f'reflection_trace_session_{j}.jsonl')
                        with open(trace_path, 'a', encoding='utf-8') as tf:
                            tf.write(json.dumps(end_entry, ensure_ascii=False) + "\n")
                    except Exception as _e:
                        logging.debug(f"Failed to append reflection trace (end): {_e}")
                    save_agents([agent_a, agent_b], args)
                # --- store reflections into memory stream ---
                if args.memory_stream:
                    try:
                        ms_a = MemoryStore(agent_a, lang=args.lang)
                        ms_b = MemoryStore(agent_b, lang=args.lang)
                        sess_date = agent_a.get('session_%s_date_time' % j)
                        # A's self and A's other
                        ms_a.add_from_reflections(reflections['a'].get('self', []), session_date=sess_date, about='self')
                        ms_a.add_from_reflections(reflections['a'].get('other', []), session_date=sess_date, about='other')
                        # B's self and B's other
                        ms_b.add_from_reflections(reflections['b'].get('self', []), session_date=sess_date, about='self')
                        ms_b.add_from_reflections(reflections['b'].get('other', []), session_date=sess_date, about='other')
                        ms_a.save(); ms_b.save()
                        save_agents([agent_a, agent_b], args)
                    except Exception as e:
                        logging.warning(f"Failed to add reflections to memory stream: {e}")

            if args.summary and ('session_%s_summary' % j not in agent_a or args.overwrite_session):

                summary = get_session_summary(agent_a['session_%s' % j], agent_a, agent_b, agent_a['session_%s_date_time' % j], 
                                              previous_summary=None if j==1 else agent_a['session_%s_summary' % (j-1)])

                agent_a['session_%s_summary' % j] = summary
                agent_b['session_%s_summary' % j] = summary
                logging.info(f"Generated session {j} summary (len={len(summary)})")

                save_agents([agent_a, agent_b], args)

            # Relationship scores per session (optional)
            if args.relationships and ('session_%s_relationships' % j not in agent_a or args.overwrite_session):
                rels = get_session_relationships(args, agent_a, agent_b, j)
                agent_a['session_%s_relationships' % j] = rels
                agent_b['session_%s_relationships' % j] = rels
                logging.info(f"Session {j} relationships: {rels}")
                save_agents([agent_a, agent_b], args)

    agent_a, agent_b = load_agents(args)
    convert_to_chat_html(agent_a, agent_b, outfile=os.path.join(args.out_dir, 'sessions.html'), use_events=False, img_dir=args.out_dir)

    # 生成された全セッションを JSON として out-dir に書き出し
    export_path = os.path.join(args.out_dir, 'all_sessions_export.json')
    export_payload = {
        'agent_a': agent_a['name'],
        'agent_b': agent_b['name'],
        'language': args.lang,
        'sessions': []
    }
    for k, v in agent_a.items():
        if k.startswith('session_') and k.split('_')[-1].isdigit():
            sid = k.split('_')[-1]
            if isinstance(v, list):  # 会話本体
                # 元のフル出力（dialog に v 全体を入れる）はユーザー要望によりコメントアウト
                # export_payload['sessions'].append({
                #     'session_id': int(sid),
                #     'date_time': agent_a.get(f'session_{sid}_date_time'),
                #     'dialog': v,
                #     'facts': agent_a.get(f'session_{sid}_facts'),
                #     'reflection': agent_a.get(f'session_{sid}_reflection'),
                #     'summary': agent_a.get(f'session_{sid}_summary'),
                #     'relationships': agent_a.get(f'session_{sid}_relationships')
                # })
                # 簡略形式: 各発話の speaker, clean_txt, retrieved のみを出力
                simplified_dialog = []
                for d in v:
                    if not isinstance(d, dict):
                        continue
                    simplified_dialog.append({
                        'speaker': d.get('speaker'),
                        'clean_txt': d.get('clean_text', ''),
                        'retrieved': d.get('retrieved', [])
                    })
                export_payload['sessions'].append({
                    'session_id': int(sid),
                    'date_time': agent_a.get(f'session_{sid}_date_time'),
                    'dialog': simplified_dialog,
                    'facts': agent_a.get(f'session_{sid}_facts'),
                    'reflection': agent_a.get(f'session_{sid}_reflection'),
                    'summary': agent_a.get(f'session_{sid}_summary'),
                    'relationships': agent_a.get(f'session_{sid}_relationships')
                })
    export_payload['sessions'] = sorted(export_payload['sessions'], key=lambda x: x['session_id'])
    with open(export_path, 'w') as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)
    logging.info(f"Exported all sessions JSON to {export_path}")

    # --- メモリストリームが空なら facts/reflection から自動バックフィル ---
    if args.memory_stream:
        try:
            def _has_mem(a):
                ms = a.get('memory_stream')
                return isinstance(ms, list) and len(ms) > 0
            if not _has_mem(agent_a) or not _has_mem(agent_b):
                logging.info("Memory stream is empty; backfilling from existing facts/reflections...")
                ms_a = MemoryStore(agent_a, lang=args.lang)
                ms_b = MemoryStore(agent_b, lang=args.lang)
                # すべてのセッションを走査
                sess_ids = [s['session_id'] for s in export_payload['sessions']]
                for sid in sess_ids:
                    sess_date = agent_a.get(f'session_{sid}_date_time')
                    facts = agent_a.get(f'session_{sid}_facts')
                    if isinstance(facts, dict):
                        ms_a.add_from_facts(agent_a['name'], facts, session_date=sess_date, about='self')
                        ms_a.add_from_facts(agent_b['name'], facts, session_date=sess_date, about='other')
                        ms_b.add_from_facts(agent_b['name'], facts, session_date=sess_date, about='self')
                        ms_b.add_from_facts(agent_a['name'], facts, session_date=sess_date, about='other')
                    refl_a = agent_a.get(f'session_{sid}_reflection') or {}
                    if isinstance(refl_a, dict):
                        ms_a.add_from_reflections((refl_a.get('self') or []), session_date=sess_date, about='self')
                        ms_a.add_from_reflections((refl_a.get('other') or []), session_date=sess_date, about='other')
                    refl_b = agent_b.get(f'session_{sid}_reflection') or {}
                    if isinstance(refl_b, dict):
                        ms_b.add_from_reflections((refl_b.get('self') or []), session_date=sess_date, about='self')
                        ms_b.add_from_reflections((refl_b.get('other') or []), session_date=sess_date, about='other')
                ms_a.save(); ms_b.save()
                save_agents([agent_a, agent_b], args)
                logging.info("Backfilled memory stream from facts/reflections.")
        except Exception as e:
            # 例外の詳細なトレースを残して次のデバッグに備える
            logging.exception(f"Failed to backfill memory stream: {e}")

    # メモリストリームを別ファイルにエクスポート（embedding は除外した軽量版）
    def _light_entries(entries):
        light = []
        for e in entries or []:
            if not isinstance(e, dict):
                continue
            light.append({
                'id': e.get('id'),
                'text': e.get('text'),
                'created_at': e.get('created_at'),
                'last_accessed_at': e.get('last_accessed_at'),
                'source_type': e.get('source_type'),
                'importance': e.get('importance'),
                'recency_weight': e.get('recency_weight'),
                'retrieval_score': e.get('retrieval_score'),
                'related_memory_ids': e.get('related_memory_ids'),
            })
        return light

    mem_export = {
        'agent_a': agent_a['name'],
        'agent_b': agent_b['name'],
        'language': args.lang,
        'memory_stream': {
            'agent_a': _light_entries(agent_a.get('memory_stream') or []),
            'agent_b': _light_entries(agent_b.get('memory_stream') or []),
        },
        'stats': {
            'agent_a': {'count': len(agent_a.get('memory_stream') or [])},
            'agent_b': {'count': len(agent_b.get('memory_stream') or [])},
        }
    }
    mem_export_path = os.path.join(args.out_dir, 'memory_stream_export.json')
    with open(mem_export_path, 'w') as f:
        json.dump(mem_export, f, ensure_ascii=False, indent=2)
    logging.info(f"Exported memory stream JSON to {mem_export_path}")

    # 追加: エージェントごとの個別ファイルにも書き出し
    a_mem = _light_entries(agent_a.get('memory_stream') or [])
    b_mem = _light_entries(agent_b.get('memory_stream') or [])

    mem_export_a = {
        'agent': agent_a['name'],
        'language': args.lang,
        'memory_stream': a_mem,
        'stats': {'count': len(a_mem)}
    }
    mem_export_b = {
        'agent': agent_b['name'],
        'language': args.lang,
        'memory_stream': b_mem,
        'stats': {'count': len(b_mem)}
    }

    mem_export_a_path = os.path.join(args.out_dir, 'memory_stream_agent_a.json')
    mem_export_b_path = os.path.join(args.out_dir, 'memory_stream_agent_b.json')
    with open(mem_export_a_path, 'w') as f:
        json.dump(mem_export_a, f, ensure_ascii=False, indent=2)
    with open(mem_export_b_path, 'w') as f:
        json.dump(mem_export_b, f, ensure_ascii=False, indent=2)
    logging.info(f"Exported per-agent memory streams to {mem_export_a_path} and {mem_export_b_path}")


if __name__ == "__main__":
    main()
