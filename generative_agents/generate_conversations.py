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
    parser.add_argument('--persona-dir', type=str, default=None, help="Directory containing persona files (agent_a.json / agent_b.json etc.). Overrides LOCOMO_PERSONA_DIR if set.")  # ペルソナファイル配置ディレクトリ（指定時は環境変数より優先）
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
    # Relationship reflection (-3..+3) options (independent of normal reflection/relationships)
    parser.add_argument('--relationship-reflection', action='store_true', help='Enable 7-point (-3..+3) relationship_reflection (Intimacy/Power) using HLQ+evidence')
    parser.add_argument('--relationship-reflection-every-turn', action='store_true', help='If set with --relationship-reflection, compute a relationship_reflection snapshot each turn for the upcoming speaker')
    parser.add_argument('--relationship-reflection-no-inject', action='store_true', help='If set with --relationship-reflection, compute relationship values but do NOT inject them into generation prompts (for ablation study)')
    # Memory stream options
    parser.add_argument('--memory-stream', action='store_true', help='Enable memory stream: store facts/reflections and retrieve top-K memories each turn')  # 記憶を保存し各ターンで上位K件をリトリーブ
    parser.add_argument('--memory-topk', type=int, default=5, help='Top-K memories to retrieve per turn when memory stream is enabled')  # リトリーブ件数K
    parser.add_argument('--memory-retrieve-inject', action='store_true', help='If set with --memory-stream, inject retrieved memories into generation prompts. If not set, memory is stored but not injected.')  # retrieve結果をプロンプトに注入するか
    parser.add_argument('--reflection-evidence-topk', type=int, default=10, help='Number of recent memory entries to use as evidence candidates for relationship reflection')  # 関係性内省のエビデンス候補数
    parser.add_argument('--min-turns-before-stop', type=int, default=6, help='Do not include stop instruction until at least this turn index (0-based)')  # 終了指示[END]を入れ始める最小ターン
    parser.add_argument('--exact-turns-per-session', type=int, default=None, help='この値が指定された場合、セッションはちょうどNターン生成される。N-1ターンまでは[END]を出力させない（ストップ指示無効化）')  # ちょうどNターン生成
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda','mps'], help='Device for BLIP model (auto: prefer cuda > mps > cpu)')  # 無効: 画像系モデルのデバイス
    parser.add_argument('--image-search', action='store_true', help='(Disabled) image retrieval removed')  # 無効: 旧画像検索
    parser.add_argument('--lang', type=str, default='en', choices=['en','ja'], help='Conversation language (default en)')  # 会話の出力言語
    parser.add_argument('--session-topic', type=str, default=None, help='Optional conversation theme/topic to guide the dialogue (applied across turns)')  # 会話テーマ
    parser.add_argument('--session-topics-file', type=str, default=None, help='Path to JSON mapping of session id to topic. Accepts {"1":"...","2":"..."} or ["...","..."] (1-based).')  # セッション別テーマ
    # シナリオファイル: 設定を一括で指定可能。連続会話モードもサポート
    parser.add_argument('--scenario-file', type=str, default=None, help='Path to scenario JSON file. Consolidates settings, events, and supports continuous conversation mode.')

    args = parser.parse_args()
    return args


def load_scenario_file(args):
    """シナリオファイルを読み込み、args と events/settings を返す。
    
    シナリオファイルのフォーマット:
    {
      "mode": "continuous" または "session" (省略時は "session"),
      "settings": {
        "lang": "ja",
        "num_sessions": 1,           // continuous モードでは 1 を推奨
        "total_turns": 60,           // continuous モードでの総ターン数
        "event_interval": 20,        // イベント切り替え間隔（ターン数）
        "relationship_reflection": true,
        "relationship_reflection_every_turn": true,
        "memory_stream": true,
        "memory_topk": 3,
        "reflection": true
      },
      "events": [
        {
          "turn": 1,
          "description": "最初の状況説明..."
        },
        {
          "turn": 20,
          "description": "次のイベント..."
        }
      ],
      // 旧形式との互換: "topics" も受け付ける (セッション区切りモード用)
      "topics": {"1": "...", "2": "..."}
    }
    """
    if not args.scenario_file or not os.path.exists(args.scenario_file):
        return args, None, {}
    
    try:
        with open(args.scenario_file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load scenario file: {e}")
        return args, None, {}
    
    # settings を args に反映（コマンドライン引数が未指定の場合のみ上書き）
    settings = scenario.get('settings', {})
    
    # 文字列 -> bool 変換用ヘルパー
    def _to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        return bool(val)
    
    # 設定マッピング（scenario キー -> args 属性名 -> デフォルト値/型変換）
    setting_mappings = {
        'lang': ('lang', str),
        'num_sessions': ('num_sessions', int),
        'total_turns': ('exact_turns_per_session', int),  # continuous モードでは exact_turns として扱う
        'max_turns_per_session': ('max_turns_per_session', int),
        'event_interval': ('event_interval', int),
        'relationship_reflection': ('relationship_reflection', _to_bool),
        'relationship_reflection_every_turn': ('relationship_reflection_every_turn', _to_bool),
        'relationship_reflection_no_inject': ('relationship_reflection_no_inject', _to_bool),
        'memory_stream': ('memory_stream', _to_bool),
        'memory_topk': ('memory_topk', int),
        'memory_retrieve_inject': ('memory_retrieve_inject', _to_bool),
        'reflection_evidence_topk': ('reflection_evidence_topk', int),
        'reflection': ('reflection', _to_bool),
        'reflection_every_turn': ('reflection_every_turn', _to_bool),
        'facts': ('facts', _to_bool),
        'summary': ('summary', _to_bool),
    }
    
    for scenario_key, (arg_attr, converter) in setting_mappings.items():
        if scenario_key in settings:
            try:
                setattr(args, arg_attr, converter(settings[scenario_key]))
            except Exception:
                pass
    
    # event_interval はシナリオファイル固有のため新規属性として追加
    if not hasattr(args, 'event_interval'):
        args.event_interval = settings.get('event_interval', 20)
    
    # モード判定
    mode = scenario.get('mode', 'session')
    args.continuous_mode = (mode == 'continuous')
    
    # events リストの取得
    events = scenario.get('events', [])
    
    # 旧形式 topics との互換: events が空で topics がある場合はセッションモードとして扱う
    session_topics = {}
    if not events and 'topics' in scenario:
        topics_data = scenario['topics']
        if isinstance(topics_data, dict):
            for k, v in topics_data.items():
                try:
                    session_topics[int(k)] = v
                except Exception:
                    continue
        elif isinstance(topics_data, list):
            for idx, v in enumerate(topics_data, start=1):
                session_topics[idx] = v
    
    # initial_relationship フィールドがあれば args に保存
    initial_relationship = scenario.get('initial_relationship')
    if initial_relationship:
        args.initial_relationship = initial_relationship
        logging.info(f"Initial relationship loaded: {initial_relationship}")
    
    logging.info(f"Loaded scenario: mode={mode}, events={len(events)}, settings={list(settings.keys())}")
    return args, events, session_topics


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
    # 優先順位: --persona-dir > LOCOMO_PERSONA_DIR 環境変数 > ハードコード既定
    default_dir = args.persona_dir or os.environ.get('LOCOMO_PERSONA_DIR', '/Users/tomokai/locomo/locomo/personas')
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
    output = run_chatgpt(query, 1, 1000, 'chatgpt')
    output = output.strip()
    return output


def get_image_queries(events):

    images = [e["image"] for e in events]
    input_query = "\nInput: ".join(images)

    output = run_chatgpt(EVENT2QUERY_PROMPT % input_query, 1, 1000, 'chatgpt')
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
                    use_events=False, instruct_stop=False, dialog_id=0, last_dialog='', embeddings=None, reflection=False, language='en', relationships=None, memory_snippet=None, topic=None, relationship_reflection=None):

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
    # Optional: conversation topic guidance
    if topic:
        if language == 'ja':
            query += f"\n\n会話のテーマ: {topic}\n"
        else:
            query += f"\n\nConversation theme: {topic}\nStick to this theme naturally. Avoid repetition and reply with a single, natural utterance that responds to your partner."
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
    # 関係値（7点尺度 -3..+3）を提供する場合
    if relationship_reflection:
        try:
            rr = relationship_reflection
            # only pass self (speaker_1) -> other (speaker_2)
            v12 = None
            if isinstance(rr, dict) and 'by_speaker' in rr and speaker_1['name'] in rr['by_speaker']:
                v12 = rr['by_speaker'][speaker_1['name']].get('vector')
            def _fmt2(v):
                if not isinstance(v, dict):
                    return 'Intimacy=?, Power=?, TaskOriented=?'
                # backward compatibility: allow old keys
                intimacy = v.get('Intimacy', v.get('Politeness', v.get('attentiveness','?')))
                power = v.get('Power', v.get('Self-Disclosure', v.get('positivity','?')))
                task_oriented = v.get('TaskOriented', v.get('GoalOrientation', '?'))
                return 'Intimacy=%s, Power=%s, TaskOriented=%s' % (intimacy, power, task_oriented)
            rr_snip = '\n\nRelationship reflection (7-point -3..+3):\n'
            rr_snip += f"{speaker_1['name']} -> {speaker_2['name']}: " + _fmt2(v12) + '\n'
            if language == 'ja':
                rr_snip += '数値は -3(低)〜+3(高) で、Intimacy（親密度）：相手に対して感じる近しさ・好意の度合い。高いほど親しい ・Power（力関係）：対話主体が相手に対して主観的に認知している社会的・個人的な力関係を指す ・TaskOriented: タスク指向対話（やり取りがどれだけタスク指向か。高いほどタスク志向、低いほど雑談・社交的）であることを表しています。この値を参考に、発話スタイルを調整してください。\n'
            else:
                rr_snip += 'Use this to adjust intimacy, power, and task orientation (-3 low .. +3 high).\n'
            query += rr_snip
        except Exception:
            pass
    
    # 直前の発話をプロンプトに含める
    if last_dialog:
        if language == 'ja':
            query += f"\n\n相手の直前の発話:\n{last_dialog}\n\nこれに対して自然に返答してください。"
        else:
            query += f"\n\nPartner's last utterance:\n{last_dialog}\n\nRespond naturally to this."
    
    return query


def get_session(agent_a, agent_b, args, prev_date_time_string='', curr_date_time_string='', curr_sess_id=0, captioner=None, img_processor=None, reflection=False, session_topic=None, events_list=None):
    """会話セッションを生成する。
    
    Args:
        events_list: 連続会話モードで使用するイベントリスト。
                     [{"turn": N, "description": "..."}, ...] 形式。
                     指定された場合、turn に応じてトピックを動的に切り替える。
    """
    
    # load embeddings for retrieving relevant observations from previous conversations
    # if the embeddings file does not exist (e.g. --facts not used), proceed without fine-grained retrieval
    if curr_sess_id == 1:
        embeddings = None
    else:
        try:
            if os.path.exists(args.emb_file):
                embeddings = pkl.load(open(args.emb_file, 'rb'))
            else:
                logging.info(f"Embeddings file not found at {args.emb_file}; proceeding without embeddings-based retrieval for session {curr_sess_id}.")
                embeddings = None
        except Exception as e:
            logging.warning(f"Failed to load embeddings file {args.emb_file}: {e}. Proceeding without embeddings.")
            embeddings = None

    # --- 連続会話モード: イベント管理の初期化 ---
    current_event_idx = 0
    current_event = None
    event_history = []  # イベント切り替えの履歴を記録
    if events_list and len(events_list) > 0:
        # events_list を turn でソート
        events_list = sorted(events_list, key=lambda x: x.get('turn', 0))
        current_event = events_list[0]
        # 最初のイベントを session_topic として設定（events_list が優先）
        session_topic = current_event.get('description', session_topic)
        # 初期イベントを履歴に記録
        event_history.append({
            'turn': 0,
            'event_idx': 0,
            'type': current_event.get('type', 'initial'),
            'description': current_event.get('description', '')
        })
        logging.info(f"========== [EVENT INJECTED] Turn 0 ==========")
        logging.info(f"  Type: {current_event.get('type', 'initial')}")
        logging.info(f"  Description: {current_event.get('description', '(none)')}")
        logging.info(f"==============================================")

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
    # current in-session relationship_reflection snapshot (-3..+3 pair), default zeros
    def _zero_rr():
        return {
            'a_to_b': {'Intimacy': 0, 'Power': 0, 'TaskOriented': 0},
            'b_to_a': {'Intimacy': 0, 'Power': 0, 'TaskOriented': 0},
            'by_speaker': {
                agent_a['name']: {'toward': agent_b['name'], 'vector': {'Intimacy': 0, 'Power': 0, 'TaskOriented': 0}},
                agent_b['name']: {'toward': agent_a['name'], 'vector': {'Intimacy': 0, 'Power': 0, 'TaskOriented': 0}}
            }
        }
    # Carry-over: if previous session has relationship_reflection, start from it
    prev_rr = None
    try:
        if curr_sess_id > 1:
            prev_rr = agent_a.get(f'session_{curr_sess_id-1}_relationship_reflection')
    except Exception:
        prev_rr = None
    
    # initial_relationship がシナリオファイルで指定されている場合は初期値として使用
    initial_rel = getattr(args, 'initial_relationship', None)
    if prev_rr and isinstance(prev_rr, dict):
        current_relationship_reflection = prev_rr
    elif initial_rel and isinstance(initial_rel, dict):
        # シナリオファイルからの初期値を適用
        a_to_b = initial_rel.get('agent_a_to_b', {'Intimacy': 0, 'Power': 0, 'TaskOriented': 0})
        b_to_a = initial_rel.get('agent_b_to_a', {'Intimacy': 0, 'Power': 0, 'TaskOriented': 0})
        current_relationship_reflection = {
            'a_to_b': a_to_b,
            'b_to_a': b_to_a,
            'by_speaker': {
                agent_a['name']: {'toward': agent_b['name'], 'vector': a_to_b},
                agent_b['name']: {'toward': agent_a['name'], 'vector': b_to_a}
            }
        }
        logging.info(f"[relationship] Initial relationship values applied: a_to_b={a_to_b}, b_to_a={b_to_a}")
    else:
        current_relationship_reflection = _zero_rr()
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

        # (変更) relationship_reflection は相手の発話を受けた後に算出し、次の発話に渡す。初期値は 0。

        # 各ターンでのリトリーブ結果（テキストのみ）を収集するための一時領域
        retrieved_texts_for_turn = []

        # --- 連続会話モード: イベント切り替えチェック ---
        if events_list and len(events_list) > 0:
            # 次のイベントがあるかチェック
            if current_event_idx < len(events_list) - 1:
                next_event = events_list[current_event_idx + 1]
                next_turn = next_event.get('turn', float('inf'))
                if i >= next_turn:
                    current_event_idx += 1
                    current_event = next_event
                    new_topic = current_event.get('description', '')
                    event_type = current_event.get('type', 'event')
                    
                    # イベント切り替え履歴を記録
                    event_history.append({
                        'turn': i,
                        'event_idx': current_event_idx,
                        'type': event_type,
                        'description': new_topic
                    })
                    
                    # トピックを更新
                    session_topic = new_topic
                    
                    # 詳細なイベント注入ログ
                    logging.info(f"========== [EVENT INJECTED] Turn {i} ==========")
                    logging.info(f"  Event Index: {current_event_idx}")
                    logging.info(f"  Type: {event_type}")
                    logging.info(f"  Description: {new_topic}")
                    logging.info(f"==============================================")
                    
                    # イベント切り替え時に関係値を評価・記録（relationship_reflection が有効な場合）
                    if args.relationship_reflection:
                        try:
                            rr_at_event = get_relationship_reflection(args, agent_a, agent_b, curr_sess_id, target='both', session_dialog=session)
                            event_history[-1]['relationship_reflection'] = rr_at_event
                            logging.info(f"[continuous] Event change relationship snapshot recorded at turn {i}")
                        except Exception as e:
                            logging.debug(f"Failed to capture relationship at event change: {e}")

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

        # （変更）ターン開始時の内省は行わない

        if curr_speaker == 0:
            # Memory retrieval for Agent A's turn
            mem_snip = None
            if args.memory_stream and mem_a is not None:
                #直前の相手の発話をクエリとする
                qtext = '' if i == 0 else (session[-1]['speaker'] + ' says, ' + session[-1]['clean_text'])
                #直前の発話に対してretrieve（上位k件抽出）を行なっている
                top = mem_a.retrieve(qtext or agent_a.get('persona_summary',''), topk=args.memory_topk, now_date=curr_date_time_string)
                _log_retrieval(agent_a.get('name','A'), i, top)
                # memory_retrieve_inject が有効な場合のみプロンプトに注入
                if getattr(args, 'memory_retrieve_inject', False):
                    mem_snip = MemoryStore.format_snippet(top, lang=args.lang)
                # エクスポート用にテキストのみ保持
                try:
                    retrieved_texts_for_turn = [e.get('text') for e in (top or []) if isinstance(e, dict) and e.get('text')]
                except Exception:
                    retrieved_texts_for_turn = []
            # relationship_reflection を注入するかどうか: --relationship-reflection-no-inject が指定されていれば None を渡す
            rr_for_prompt = None if getattr(args, 'relationship_reflection_no_inject', False) else current_relationship_reflection
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
                relationship_reflection=rr_for_prompt,
                memory_snippet=mem_snip,
                topic=session_topic
            )
            # （削除）詳細トレースは出力しない
        else:
            mem_snip = None
            if args.memory_stream and mem_b is not None:
                qtext = '' if i == 0 else (session[-1]['speaker'] + ' says, ' + session[-1]['clean_text'])
                top = mem_b.retrieve(qtext or agent_b.get('persona_summary',''), topk=args.memory_topk, now_date=curr_date_time_string)
                _log_retrieval(agent_b.get('name','B'), i, top)
                # memory_retrieve_inject が有効な場合のみプロンプトに注入
                if getattr(args, 'memory_retrieve_inject', False):
                    mem_snip = MemoryStore.format_snippet(top, lang=args.lang)
                # エクスポート用にテキストのみ保持
                try:
                    retrieved_texts_for_turn = [e.get('text') for e in (top or []) if isinstance(e, dict) and e.get('text')]
                except Exception:
                    retrieved_texts_for_turn = []
            # relationship_reflection を注入するかどうか: --relationship-reflection-no-inject が指定されていれば None を渡す
            rr_for_prompt = None if getattr(args, 'relationship_reflection_no_inject', False) else current_relationship_reflection
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
                relationship_reflection=rr_for_prompt,
                memory_snippet=mem_snip,
                topic=session_topic
            )
            # （削除）詳細トレースは出力しない

        # 画像関連機能無効化 (placeholder)
        # if args.image_search ...

        # 発話生成用の完全なプロンプトを構築 
        full_prompt = agent_query + conv_so_far
        
        # トークン上限を拡大（Gemini 2.5 thinking対応で1000トークンに増加）
        raw = run_chatgpt(full_prompt, 1, 1000, 'chatgpt', temperature=1.2)
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
                    output["clean_text"] = run_chatgpt(CASUAL_DIALOG_PROMPT % text_replaced_caption, 1, 1000, 'chatgpt').strip()
        else:
            output["clean_text"] = ""

        output["dia_id"] = f'D{curr_sess_id}:{i+1}'
        # 各ターンのリトリーブ結果（テキスト配列）を保存（エクスポート用）
        output["retrieved"] = retrieved_texts_for_turn or []
        # 発話生成に使用したプロンプトを記録（agent_queryのみ、会話履歴全体は含めない）
        # agent_query には last_dialog として直前の発話が含まれている
        output["generation_prompt"] = agent_query
        # 現在のイベント情報を記録
        if current_event:
            output["current_event"] = {
                'event_idx': current_event_idx,
                'type': current_event.get('type', ''),
                'description': current_event.get('description', '')
            }
        # 関係値情報を記録（注入/非注入を区別）
        if args.relationship_reflection and current_relationship_reflection:
            is_injected = not getattr(args, 'relationship_reflection_no_inject', False)
            speaker_name = output.get('speaker')
            if speaker_name and 'by_speaker' in current_relationship_reflection:
                speaker_rr = current_relationship_reflection['by_speaker'].get(speaker_name, {})
                output["relationship_reflection_used"] = {
                    'speaker': speaker_name,
                    'toward': speaker_rr.get('toward', ''),
                    'vector': speaker_rr.get('vector', {}),
                    'injected': is_injected
                }
        session.append(output)
        # --- export minimal per-turn trace (utterance + retrieved + relationship_reflection used-direction) ---
        try:
            trace_path = os.path.join(args.out_dir, f'prompt_trace_session_{curr_sess_id}.jsonl')
            # derive the vector passed to this speaker (self->other) for logging
            def _vec_for(name_self, rr):
                if isinstance(rr, dict) and 'by_speaker' in rr and name_self in rr['by_speaker']:
                    return rr['by_speaker'][name_self].get('vector')
                return None
            used_vec = _vec_for(output.get('speaker'), current_relationship_reflection)
            minimal = {
                'session_id': curr_sess_id,
                'turn': i,
                'speaker': output.get('speaker'),
                'utterance': output.get('clean_text', ''),
                'retrieved_texts': output.get('retrieved', []),
                'relationship_reflection': used_vec
            }
            with open(trace_path, 'a', encoding='utf-8') as tf:
                tf.write(json.dumps(minimal, ensure_ascii=False) + "\n")
        except Exception as _e:
            logging.debug(f"Failed to append minimal prompt trace: {_e}")
        # concise console log for relationship_reflection (self->other) used this turn
        try:
            if args.relationship_reflection and current_relationship_reflection:
                def _fmt(v):
                    if not isinstance(v, dict):
                        return 'Power=?, Intimacy=?, TaskOriented=?'
                    return (
                        f"Power={v.get('Power', v.get('Self-Disclosure', v.get('positivity','?')))}, "
                        f"Intimacy={v.get('Intimacy', v.get('Politeness', v.get('attentiveness','?')))}, "
                        f"TaskOriented={v.get('TaskOriented', v.get('GoalOrientation', '?'))}"
                    )
                used = used_vec if 'used_vec' in locals() else None
                logging.info(f"[rr] turn={i} used self→other: {output.get('speaker')}: {_fmt(used)}")
        except Exception as _e:
            logging.debug(f"Failed to log relationship_reflection snapshot: {_e}")

        # もし、メモリストリームが有効なら、発話を両エージェントのメモリにミラー保存し、importantce 統計は話した側のみ更新
        if args.memory_stream:
            try:
                # 話者名を明示したテキストで保存して、後段の検索時に誰の発話か判別しやすくする
                entry_text = f"{output['speaker']}: {output['clean_text']}".strip()
                # 両エージェントへミラー保存
                m_a = mem_a.add_memory(text=entry_text, created_at=curr_date_time_string, source_type='conversation') if mem_a is not None else None
                m_b = mem_b.add_memory(text=entry_text, created_at=curr_date_time_string, source_type='conversation') if mem_b is not None else None

                # 重要度統計（early reflection トリガ）は従来通り「発話した側」のみ更新
                if curr_speaker == 0 and m_a is not None:
                    imp = int(m_a.get('importance', 0))
                    stats_a['cumulative_total'] += imp
                    rw = stats_a['recent_window']
                    rw.append(imp)
                    if len(rw) > RECENT_WINDOW_SIZE:
                        rw.pop(0)
                elif curr_speaker == 1 and m_b is not None:
                    imp = int(m_b.get('importance', 0))
                    stats_b['cumulative_total'] += imp
                    rw = stats_b['recent_window']
                    rw.append(imp)
                    if len(rw) > RECENT_WINDOW_SIZE:
                        rw.pop(0)

                # エージェント辞書へ反映（ファイル保存は既存の save_agents タイミングに委ねる）
                if mem_a is not None:
                    mem_a.save()
                if mem_b is not None:
                    mem_b.save()
            except Exception as e:
                logging.warning(f"Failed to add dialog memory for dynamic reflection trigger: {e}")

    # Per-turn reflection（変更後）: ターン終了後に「次に話す側（リスナー）」が内省する
        if args.reflection and args.reflection_every_turn:
            try:
                # セッション進行状況を保存
                agent_a[f'session_{curr_sess_id}'] = session
                agent_b[f'session_{curr_sess_id}'] = session
                # 次の発話者（リスナー）を判定: 直後に curr_speaker がトグルされるため、
                # ここでは現時点の curr_speaker が「今話した側」。次に話すのは not curr_speaker。
                next_is_a = (curr_speaker == 1)  # 今話したのがBなら次はA、今話したのがAなら次はB
                # 片側のみ生成: 次に話す側のみ
                refl_all = get_session_reflection(args, agent_a, agent_b, curr_sess_id, target='a' if next_is_a else 'b')
                if next_is_a:
                    agent_a[f'session_{curr_sess_id}_reflection'] = refl_all['a']
                    if args.memory_stream and mem_a is not None:
                        sess_date = agent_a.get('session_%s_date_time' % curr_sess_id)
                        mem_a.add_from_reflections(refl_all['a'].get('self', []) or [], session_date=sess_date, about='self')
                        mem_a.add_from_reflections(refl_all['a'].get('other', []) or [], session_date=sess_date, about='other')
                        mem_a.save(); save_agents([agent_a, agent_b], args)
                    a_self = (refl_all['a'].get('self') or [])
                    a_other = (refl_all['a'].get('other') or [])
                    logging.info(f"（内省者：次に話す側）{agent_a['name']} reflects -> self:{len(a_self)}, other:{len(a_other)}")
                else:
                    agent_b[f'session_{curr_sess_id}_reflection'] = refl_all['b']
                    if args.memory_stream and mem_b is not None:
                        sess_date = agent_b.get('session_%s_date_time' % curr_sess_id)
                        mem_b.add_from_reflections(refl_all['b'].get('self', []) or [], session_date=sess_date, about='self')
                        mem_b.add_from_reflections(refl_all['b'].get('other', []) or [], session_date=sess_date, about='other')
                        mem_b.save(); save_agents([agent_a, agent_b], args)
                    b_self = (refl_all['b'].get('self') or [])
                    b_other = (refl_all['b'].get('other') or [])
                    logging.info(f"（内省者：次に話す側）{agent_b['name']} reflects -> self:{len(b_self)}, other:{len(b_other)}")
            except Exception as e:
                logging.warning(f"Post-turn reflection failed at turn {i}: {e}")
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
                        # 片側のみ生成: トリガになった側だけ
                        refl = get_session_reflection(args, agent_a, agent_b, curr_sess_id, target=('a' if trigger_agent == 'A' else 'b'))
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

        # --- After the utterance, compute relationship_reflection for the next speaker ---
        # NOTE: This must run BEFORE the break to ensure relationship values are updated each turn
        if args.relationship_reflection and args.relationship_reflection_every_turn:
            try:
                # Save the session so far (including this utterance)
                agent_a[f'session_{curr_sess_id}'] = session
                agent_b[f'session_{curr_sess_id}'] = session
                # Next speaker: curr_speaker has been toggled already above
                next_is_a = (curr_speaker == 0)
                target = 'a' if next_is_a else 'b'
                rr_next = get_relationship_reflection(args, agent_a, agent_b, curr_sess_id, target=target, session_dialog=session)
                # Merge into current snapshot, preserving the opposite direction
                if next_is_a:
                    current_relationship_reflection['a_to_b'] = rr_next.get('a_to_b', current_relationship_reflection['a_to_b'])
                    current_relationship_reflection['by_speaker'][agent_a['name']] = rr_next.get('by_speaker', {}).get(agent_a['name'], current_relationship_reflection['by_speaker'][agent_a['name']])
                else:
                    current_relationship_reflection['b_to_a'] = rr_next.get('b_to_a', current_relationship_reflection['b_to_a'])
                    current_relationship_reflection['by_speaker'][agent_b['name']] = rr_next.get('by_speaker', {}).get(agent_b['name'], current_relationship_reflection['by_speaker'][agent_b['name']])
                turns_key = f'session_{curr_sess_id}_relationship_reflection_turns'
                injected = not getattr(args, 'relationship_reflection_no_inject', False)
                entry_post = {'turn': i, 'rr': rr_next, 'phase': 'after_generation', 'next_speaker': agent_a['name'] if next_is_a else agent_b['name'], 'time': datetime.utcnow().isoformat(), 'injected': injected}
                agent_a.setdefault(turns_key, []).append(entry_post)
                agent_b.setdefault(turns_key, []).append(entry_post)
                save_agents([agent_a, agent_b], args)
                # backward compatibility for logging values
                target_vec = rr_next.get('a_to_b' if next_is_a else 'b_to_a', {})
                intimacy = target_vec.get('Intimacy', target_vec.get('Politeness', target_vec.get('attentiveness','?')))
                power = target_vec.get('Power', target_vec.get('Self-Disclosure', target_vec.get('positivity','?')))
                task_oriented = target_vec.get('TaskOriented', target_vec.get('GoalOrientation', '?'))
                inject_label = '' if injected else ' [NOT INJECTED]'
                logging.info(
                    f"[rr/post] turn={i} updated for next speaker {entry_post['next_speaker']}: Power={power}, Intimacy={intimacy}, TaskOriented={task_oriented}{inject_label}"
                )
            except Exception as _e:
                logging.warning(f"relationship_reflection post-turn failed at turn {i}: {_e}")

        # 強制的に exact-turns で終了 (関係値内省の後に配置)
        if args.exact_turns_per_session is not None and (i+1) >= args.exact_turns_per_session:
            break

    # 連続会話モードの場合、イベント履歴を返す
    if events_list and len(event_history) > 0:
        return session, event_history
    return session


def main():

    # get arguments
    args = parse_args()

    set_openai_key()
    # lang を環境変数経由で下位関数へ (簡易共有)
    os.environ['LOCOMO_LANG'] = args.lang

    args.emb_file = os.path.join(args.out_dir, args.emb_file)

    # シナリオファイルの読み込み（--scenario-file が指定されている場合）
    events_list = None
    scenario_session_topics = {}
    if args.scenario_file:
        args, events_list, scenario_session_topics = load_scenario_file(args)
        logging.info(f"Scenario loaded: continuous_mode={getattr(args, 'continuous_mode', False)}, events={len(events_list) if events_list else 0}")

    # セッション別テーマの読み込み（任意）- シナリオファイルの topics または --session-topics-file から
    session_topics = scenario_session_topics.copy() if scenario_session_topics else {}
    if args.session_topics_file and os.path.exists(args.session_topics_file):
        try:
            with open(args.session_topics_file, 'r', encoding='utf-8') as tf:
                data = json.load(tf)
            if isinstance(data, dict):
                # keys may be strings; normalize to int
                for k, v in data.items():
                    try:
                        session_topics[int(k)] = v
                    except Exception:
                        continue
            elif isinstance(data, list):
                # 1-based indexing: index 0 -> session 1
                for idx, v in enumerate(data, start=1):
                    session_topics[idx] = v
            else:
                logging.warning(f"Unsupported session-topics-file format: {type(data)}")
        except Exception as e:
            logging.warning(f"Failed to read session-topics-file: {e}")

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

        # このセッションに適用するテーマ（単一指定 > ファイル指定の順で優先）
        # 連続会話モードでは events_list を使用するため session_topic は初期値のみ
        session_topic = args.session_topic if args.session_topic else session_topics.get(j)

        print("******************* SESSION %s ******************" % j)

        if 'session_%s' % j not in agent_a or args.overwrite_session:
            # 連続時間セッション: セッション1を基準日とし、以降は1日ずつ進める簡易モデル
            if j > 1:
                prev_date = datetimeStr2Obj(agent_a['session_%s_date_time' % (j-1)])
                prev_date_time_string = agent_a['session_%s_date_time' % (j-1)]
                curr_date = (prev_date + timedelta(days=1)).date()
            else:
                curr_date = get_random_date()
                prev_date_time_string = None
            curr_date_time_string = datetimeObj2Str(curr_date)
            agent_a['session_%s_date_time' % j] = curr_date_time_string
            agent_b['session_%s_date_time' % j] = curr_date_time_string
            # テーマがあれば保存（表示とエクスポート用）
            if session_topic:
                agent_a['session_%s_topic' % j] = session_topic
                agent_b['session_%s_topic' % j] = session_topic
            save_agents([agent_a, agent_b], args)

            # 連続会話モードかどうかで呼び出し方を分岐
            result = get_session(
                agent_a, agent_b, args,
                prev_date_time_string=prev_date_time_string, curr_date_time_string=curr_date_time_string,
                curr_sess_id=j, captioner=captioner, img_processor=img_processor, reflection=args.reflection,
                session_topic=session_topic,
                events_list=events_list if getattr(args, 'continuous_mode', False) else None
            )
            
            # 連続会話モードの場合は (session, event_history) のタプルが返る
            if isinstance(result, tuple):
                session, event_history = result
                # イベント履歴をエージェントに保存
                agent_a['session_%s_event_history' % j] = event_history
                agent_b['session_%s_event_history' % j] = event_history
                logging.info(f"[continuous] Session {j} completed with {len(event_history)} event transitions")
            else:
                session = result

            agent_a['session_%s' % j] = session
            agent_b['session_%s' % j] = session

            save_agents([agent_a, agent_b], args)
        else:
            # 既存セッションがあり再生成しない場合でも、テーマ指定があれば保存して表示/エクスポートに反映
            if session_topic:
                try:
                    agent_a['session_%s_topic' % j] = session_topic
                    agent_b['session_%s_topic' % j] = session_topic
                    save_agents([agent_a, agent_b], args)
                except Exception:
                    pass

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

        # セッション終了時の内省は、毎ターン内省フラグ（--reflection-every-turn）が有効な場合のみ実行
        if (args.reflection and args.reflection_every_turn) and (('session_%s_reflection' % j not in agent_a) or args.overwrite_session):
                # 早期反省(early reflection)が既に生成済みならスキップ（上書き要求がある場合を除く）
                if ('session_%s_reflection' % j in agent_a) and not args.overwrite_session:
                    pass
                else:
                    # セッション終了時の最終内省は両側生成（従来互換）
                    reflections = get_session_reflection(args, agent_a, agent_b, j, target='both')
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
        else:
            # ログ: 毎ターン内省が無効なため、セッション終了時の内省もスキップ
            if args.reflection and not args.reflection_every_turn:
                logging.info(f"[reflection] --reflection-every-turn が未指定のため、Session {j} の終了時内省は実行されません")

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
        # Final relationship_reflection snapshot per session (independent)
        if args.relationship_reflection and (('session_%s_relationship_reflection' % j not in agent_a) or args.overwrite_session):
                try:
                    rr_final = get_relationship_reflection(args, agent_a, agent_b, j, target='both')
                    agent_a['session_%s_relationship_reflection' % j] = rr_final
                    agent_b['session_%s_relationship_reflection' % j] = rr_final
                    save_agents([agent_a, agent_b], args)
                    logging.info(f"Session {j} relationship_reflection: {rr_final}")
                except Exception as e:
                    logging.debug(f"Failed to compute final relationship_reflection for session {j}: {e}")
        else:
            # 既存セッションがあり再生成しない場合でも、テーマ指定があれば保存して表示/エクスポートに反映
            if session_topic:
                try:
                    agent_a['session_%s_topic' % j] = session_topic
                    agent_b['session_%s_topic' % j] = session_topic
                    save_agents([agent_a, agent_b], args)
                except Exception:
                    pass

    agent_a, agent_b = load_agents(args)
    convert_to_chat_html(agent_a, agent_b, outfile=os.path.join(args.out_dir, 'sessions.html'), use_events=False, img_dir=args.out_dir)

    # 生成された全セッションを JSON として out-dir に書き出し
    export_path = os.path.join(args.out_dir, 'all_sessions_export.json')
    
    # コマンド引数を記録
    args_dict = vars(args).copy()
    # 保存不可能なオブジェクトを除外
    for key in list(args_dict.keys()):
        try:
            json.dumps(args_dict[key])
        except (TypeError, ValueError):
            args_dict[key] = str(args_dict[key])
    
    # 使用したGeminiモデル名を追加
    gemini_model_used = os.environ.get("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
    # global_methods から実際に使用されたモデルを取得（可能なら）
    try:
        from global_methods import _LAST_GOOD_GEMINI_MODEL
        if _LAST_GOOD_GEMINI_MODEL:
            gemini_model_used = _LAST_GOOD_GEMINI_MODEL
    except ImportError:
        pass
    args_dict['gemini_model'] = gemini_model_used
    
    export_payload = {
        'command_args': args_dict,
        'agent_a': agent_a['name'],
        'agent_b': agent_b['name'],
        'language': args.lang,
        'sessions': []
    }
    for k, v in agent_a.items():
        if k.startswith('session_') and k.split('_')[-1].isdigit():
            sid = k.split('_')[-1]
            if isinstance(v, list):  # 会話本体
                # 拡張形式: 発話、関係値、想起、イベントを含む
                enriched_dialog = []
                last_event_turn = -1
                for d in v:
                    if not isinstance(d, dict):
                        continue
                    turn_idx = len(enriched_dialog)
                    
                    # イベント切り替えがあれば、会話の前に挿入
                    current_ev = d.get('current_event')
                    if current_ev:
                        ev_turn = current_ev.get('event_idx', 0)
                        # 新しいイベントの場合のみ挿入
                        if ev_turn > 0 and enriched_dialog and 'event_change' not in enriched_dialog[-1]:
                            # イベント変更マーカーを直前に確認
                            pass
                    
                    entry = {
                        'turn': turn_idx,
                        'speaker': d.get('speaker'),
                        'utterance': d.get('clean_text', '')
                    }
                    
                    # 関係値情報（注入/非注入を区別）
                    rr_used = d.get('relationship_reflection_used')
                    if rr_used:
                        entry['relationship_reflection'] = {
                            'from': rr_used.get('speaker'),
                            'toward': rr_used.get('toward'),
                            'vector': rr_used.get('vector'),
                            'was_injected': rr_used.get('injected', False)
                        }
                    
                    # 想起情報
                    retrieved = d.get('retrieved', [])
                    if retrieved:
                        entry['retrieved_memories'] = retrieved
                    
                    # 現在のイベント情報
                    if current_ev:
                        entry['active_event'] = {
                            'event_idx': current_ev.get('event_idx'),
                            'type': current_ev.get('type'),
                            'description': current_ev.get('description')
                        }
                    
                    enriched_dialog.append(entry)
                
                # イベント履歴を取得（relationship_reflection_turns から event_history を抽出）
                event_hist = agent_a.get(f'session_{sid}_event_history', [])
                
                export_payload['sessions'].append({
                    'session_id': int(sid),
                    'date_time': agent_a.get(f'session_{sid}_date_time'),
                    'topic': agent_a.get(f'session_{sid}_topic'),
                    'event_history': event_hist,
                    'dialog': enriched_dialog,
                    'final_relationship_reflection': agent_a.get(f'session_{sid}_relationship_reflection'),
                    'facts': agent_a.get(f'session_{sid}_facts'),
                    'reflection': agent_a.get(f'session_{sid}_reflection'),
                    'summary': agent_a.get(f'session_{sid}_summary'),
                    'relationships': agent_a.get(f'session_{sid}_relationships')
                })
    export_payload['sessions'] = sorted(export_payload['sessions'], key=lambda x: x['session_id'])
    with open(export_path, 'w') as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)
    logging.info(f"Exported all sessions JSON to {export_path}")

    # --- プロンプトと発話のペアをファイルに出力 ---
    prompt_export_path = os.path.join(args.out_dir, 'prompt_utterance_pairs.jsonl')
    try:
        with open(prompt_export_path, 'w', encoding='utf-8') as pf:
            for k, v in agent_a.items():
                if k.startswith('session_') and k.split('_')[-1].isdigit():
                    sid = k.split('_')[-1]
                    if isinstance(v, list):
                        for turn_idx, d in enumerate(v):
                            if not isinstance(d, dict):
                                continue
                            entry = {
                                'session_id': int(sid),
                                'turn': turn_idx,
                                'speaker': d.get('speaker'),
                                'utterance': d.get('clean_text', ''),
                                'prompt': d.get('generation_prompt', '')
                            }
                            pf.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logging.info(f"Exported prompt-utterance pairs to {prompt_export_path}")
    except Exception as e:
        logging.warning(f"Failed to export prompt-utterance pairs: {e}")

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

    # 自動グラフ生成（関係値推移）
    try:
        import sys
        import importlib.util
        # scripts ディレクトリのパスを構築
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
        plot_module_path = os.path.join(scripts_dir, 'plot_single_experiment.py')
        
        if os.path.exists(plot_module_path):
            # 動的にモジュールをロード
            spec = importlib.util.spec_from_file_location("plot_single_experiment", plot_module_path)
            plot_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plot_module)
            
            logging.info(f"Generating relationship graphs in {args.out_dir}/graphs/...")
            generated = plot_module.generate_all_graphs(args.out_dir, show=False)
            if generated:
                logging.info(f"Generated {len(generated)} graph files:")
                for gpath in generated:
                    logging.info(f"  - {gpath}")
        else:
            logging.warning(f"Plot script not found: {plot_module_path}")
    except ImportError as e:
        logging.warning(f"Could not import plot_single_experiment: {e}")
        logging.warning("Skipping automatic graph generation. Run manually with: python scripts/plot_single_experiment.py --dir <out_dir>")
    except Exception as e:
        logging.warning(f"Failed to generate graphs: {e}")


if __name__ == "__main__":
    main()
