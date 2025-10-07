## LLM プロンプト集（会話生成）

この文書は、locomo リポジトリ内で会話生成・要約・ファクト抽出・自己反省などに用いられている主要なプロンプト（指示文）を日本語で分かりやすくまとめたものです。ファイルの所在や使用上の注意も記載しています。

### 概要

- 主なプロンプトは `generative_agents/conversation_utils.py`、`generative_agents/memory_utils.py`、`generative_agents/event_utils.py`、`global_methods.py` に定義されています。
- 実際の in-context examples（入出力例）は `prompt_examples/` ディレクトリの JSON ファイル群を参照しており、生成時はこれらの例を組み合わせて与えています。

---

## 重要なプロンプトと用途（要約）

### PERSONA_FROM_MSC_PROMPT

- 用途: MSC（属性データ）から「名前」と「persona（本文）」を含む JSON を生成する指示。出力は厳密に JSON（キーは `persona` と `name`）で返すことを要求します。
- 備考: 日本語出力が必要な場合は呼び出し側で日本語化の追記を追加します（例: args.lang=='ja' のとき）。生成後、JSON 抽出・パースを堅牢化するために複数回のリトライや簡易修復を行います。

### AGENT*CONV_PROMPT 系（AGENT_CONV_PROMPT_SESS_1, AGENT_CONV_PROMPT, *\_W*EVENTS, *\_V2 等）

- 用途: 各エージェントの「次に言う一言」を生成するメインの会話プロンプト群。
- 共通ルール（多くのテンプレで指定）:
  - 返答は短い一文（概ね 20 語相当以内）にする。
  - 個人的な内容（家族、好み、将来の希望など）に触れる。
  - これまで共有した情報は繰り返さない。
  - 時間参照（'last Friday'、'next month' 等）や具体的な場所名を含めることを推奨。
  - 会話終了時は 'Bye!' を出力することが指定されている場合がある。
- バリエーション:
  - `SESS_1`: 初対面用の文言。初回会話向けの導入が入る。
  - `_W_EVENTS`, `_V2`: エージェントの人生で起きた出来事（EVENTS）を与え、それを会話に活かすように指示。`RELEVANT_CONTEXT` や `SUMMARY` を追加して文脈を拡張するバージョンもあります。
- プレースホルダ例: PERSONALITY, speaker names, date/time, SUMMARY, EVENTS, RELEVANT_CONTEXT, stop_instruction

### SESSION_SUMMARY_PROMPT / SESSION_SUMMARY_INIT_PROMPT

- 用途: これまでの会話や直近の会話を要約して、次のセッションの文脈として利用する。出力長や含むべき要素（両者のキーファクト、時間参照など）を指定します。

### CONVERSATION2FACTS_PROMPT_EN / CONVERSATION2FACTS_PROMPT_JA

- 用途: 会話から「観察できる客観的事実（facts/observations）」を漏れなく抽出し、JSON オブジェクトとして返すように指示する。
- 出力形式: {"<SpeakerName>": [["observation", "D1:1"], ...], ...} のような JSON。JA 版は観察文を必ず日本語で返すことを明記しています。

### REFLECTION*PROMPT 系（REFLECTION_INIT_PROMPT / REFLECTION_CONTINUE_PROMPT / SELF_REFLECTION*\*）

- 用途: 会話をもとに「その人物が持つ最も重要な洞察（insights）」を 3 つ出すように求める。自己反省（self）と相手についての反省（other）の両方を生成します。
- 出力: JSON 配列（短文のリスト）を期待するケースが多い。日本語モードでは追加ヒントを足して日本語の短文配列を返すよう要求します。

### EVENT2QUERY_PROMPT / DIALOG2IMAGE_QUERY_PROMPT

- 用途: イベント記述や共有された画像説明から、画像検索用の短いクエリを生成する。このクエリは人物名や年など不要な情報を含めないよう指示します。

### CASUAL_DIALOG_PROMPT

- 用途: 与えられた文をよりカジュアルで短い表現に変換するための簡易テンプレ。

### ALIGNMENT_PROMPT

- 用途: ダイアログと画像キャプションが関連するかを 0/1 で判定するための指示（画像の関連度判定）。

### VISUAL_QUESTION_PROMPT

- 用途: 画像やキャプションを踏まえた自然な質問やコメントを生成するためのテンプレート。

---

## 実装時の注記・運用ルール

- 言語切替: スクリプト（例: `generate_conversations.py`, `memory_utils.py`）は `args.lang` を見て日本語用の追加ヒントを各プロンプト末尾に付与します。これにより同じテンプレを英語/日本語で使い分けています。
- トークン量: 要所で `num_tokens_request` を指定しており、JSON が途中で切れないよう余裕を持った上限を与えています（例: 300 ～ 900 トークンなど）。
- 出力形式の堅牢化: JSON を期待するプロンプト（persona 生成や facts/reflection 出力など）では、出力の前処理（最初の '{' から対応する '}' を抜き出すなど）とリトライ/簡易修復を行っています。
- 参照例: 実際の in-context examples は `prompt_examples/*.json` にまとまっているため、テンプレ内で `%s` 等に埋め込むことで直接例を与えています。編集時は該当 JSON を更新してください。

---

## 参照ファイル一覧（主要な配置）

- `generative_agents/conversation_utils.py` — 会話生成テンプレ（AGENT_CONV_PROMPT 系、PERSONA_FROM_MSC_PROMPT など）
- `generative_agents/memory_utils.py` — facts/reflection 生成テンプレ（CONVERSATION2FACTS*PROMPT*\_, REFLECTION\_\_ など）
- `generative_agents/event_utils.py` — イベント生成関連テンプレ
- `generative_agents/generate_conversations.py` — オーケストレーション（prompt の組み立て、引数処理）
- `prompt_examples/` — in-context examples（JSON）

---

もし特定のプロンプト全文（テンプレートそのもの）を LLM.MD に全文収録したい場合は優先度と目的（ドキュメント用か試験用か）を教えてください。全文を載せると長くなりますが、別ファイルとしてプロンプト辞書を出力することも可能です。

作成日: 2025-09-18

## テンプレへの埋め込み順序（generate_conversations.py に基づく詳細）

以下は `generative_agents/generate_conversations.py` の `get_agent_query` / `get_session` の実装に基づき、最終的に LLM に渡されるプロンプトがどの順序でどの要素を含むかを正確に示したものです。プロンプトは内部で組み立てられた `agent_query`（テンプレ本体）と、現在のセッション内での会話履歴 `conv_so_far` を連結した文字列として LLM に渡されます。

ステップ（上から順にテンプレへ差し込まれる順序）:

1. 初期処理

   - `stop_instruction`（会話終了を示す特殊トークン指示）が設定される（必要に応じて末尾で挿入）。
   - フラグ確認: `use_events`（イベントを使うか）、`reflection`（リフレクションを使うか）、`language`（ja/en）など。

2. Persona（常に最初に挿入される）

   - `speaker_1['persona_summary']`（エージェント自身のペルソナ説明）がテンプレートの先頭に入ります。

3. 話者名・相手名（テンプレに早めに出現）

   - `speaker_1['name']` と `speaker_2['name']` がテンプレの所定箇所にセットされます（呼びかけや文脈表記のため）。

4. 日付情報

   - `prev_sess_date_time`（存在する場合） と `curr_sess_date_time`（現在セッションの日付）が差し込まれます。

5. 直前の要約 / 全セッション要約

   - `curr_sess_id == 1`（初回会話）の場合は直前要約は無い。
   - `use_events == False` の一般経路では、`get_all_session_summary(speaker_1, curr_sess_id)` による『これまでの全セッション要約』が SUMMARY として埋め込まれる。
   - `use_events == True` かつ新セッション開始 (`dialog_id == 0`) では、直前セッションの要約（`speaker_1['session_{n-1}_summary']`）が挿入される（v2_init の流れ）。

6. EVENTS（有効な場合）

   - `use_events == True` の場合、`events`（`speaker_1['events_session_%s']` を整形した文字列）が `EVENTS` プレースホルダへ挿入されます。

7. 過去の関連コンテキスト（ongoing session の場合）

   - 既にセッションが進行中で `dialog_id != 0` の場合、`get_relevant_context(...)` により埋め込み検索（embedding retrieval）が行われ、その戻り値 `past_context` がテンプレに挿入されます（v2 の ongoing セッション用プレースホルダ）。

8. stop 指示（必要なら）

   - `instruct_stop` が True（内部でランダムに決定される）なら、`stop_instruction`（例: 'To end the conversation, write [END] at the end of the dialog.'）がテンプレ指定箇所へ挿入されます。

9. 言語ヒント（最後に付与）

   - `language == 'ja'` の場合、テンプレ末尾に日本語で出力するような注意書き（例: 日本語でカジュアルに 1 発話のみ）を付け加えます。

10. セッション内会話履歴の連結（最終段階）

    - `agent_query`（上記を差し込んだテンプレ本体）の末尾に `conv_so_far` を連結して `run_chatgpt(agent_query + conv_so_far, ...)` を実行します。
    - `conv_so_far` は現在のセッションで既に生成された発話を "SpeakerName: utterance" の形で連結したものです。次に生成すべき話者名（末尾の 'OtherAgent: ' 等）が付与されているため、LLM は続きの一文を出力するよう促されます。

11. 生成後の後処理
    - LLM の生の出力は `clean_dialog` 等で正規化され、必要に応じて `CASUAL_DIALOG_PROMPT` に投げ直して短くカジュアルに整形（英語モードのみ）します。
    - 生成セッションはファイルへ保存され、続いて `get_session_facts`（CONVERSATION2FACTS*PROMPT を使用）で facts を抽出、`get_session_reflection`（REFLECTION*_）でリフレクションを作る（`--reflection` フラグが有効な場合）、最後に `get_session_summary`（SESSION*SUMMARY*_）で要約を作成する順序で処理されます。

テンプレの実際の差し込み例（概念的、簡略化）:

- agent_query = AGENT_CONV_PROMPT % (persona_summary,
  speaker_name,
  other_name,
  prev_sess_date_time,
  SUMMARY (all previous sessions),
  curr_sess_date_time,
  speaker_name,
  other_name,
  speaker_name)
- final_prompt = agent_query + conv_so_far

- past_context = get_relevant_context(...)
- agent_query = AGENT_CONV_PROMPT_W_EVENTS_V2 % (persona_summary,
  speaker_name,
  other_name,
  prev_sess_date_time,
  curr_sess_date_time,
  speaker_name,
  last_session_summary,
  events,
  past_context,
  stop_instruction_if_any,
  other_name)
- final_prompt = agent_query + conv_so_far

注意: 実際のテンプレ文字列内での `%s` の数・順序はテンプレごとに異なります（`conversation_utils.py` 内の定義を参照）。上の箇条は論理的な挿入順を示したもので、テンプレ側のフォーマット順と完全一致するようにコードでフォーマットされています。

## 関係値 JSON の読み方（a_to_b / b_to_a / by_speaker）

例:

"relationships": {
"a_to_b": {"intimacy": 7, "power": 2, "social_distance": 4, "trust": 8},
"b_to_a": {"intimacy": 7, "power": 6, "social_distance": 4, "trust": 8},
"by_speaker": {
"セイバー": {"toward": "衛宮士郎", "scores": {"intimacy":7, "power":2, "social_distance":4, "trust":8}},
"衛宮士郎": {"toward": "セイバー", "scores": {"intimacy":7, "power":6, "social_distance":4, "trust":8}}
}
}

説明（わかりやすく）:

- `a_to_b`: エージェント A がエージェント B に対して感じている関係値（A→B）。上の例では「A が B に対して親密度 7、力関係 2、社会的距離 4、信頼 8」と読めます。
- `b_to_a`: エージェント B がエージェント A に対して感じている関係値（B→A）。上の例では「B が A に対して親密度 7、力関係 6、社会的距離 4、信頼 8」です。
- `by_speaker`: 話者名をキーにして同様の情報を持つ拡張フォーマット。`by_speaker` の各エントリは `{ "toward": "相手名", "scores": {...} }` の形で、誰が誰に向けた値かが名前ベースで明示されます。

読み方のポイント:

- JSON の `a` / `b` は内部での A/B の順序に依存するため、`by_speaker` があると読み取りミスが減ります（`by_speaker` は名前で対応先を明示しているため最も信頼できます）。
- 値は 1〜10 の整数で、スコアは次の意味です: `intimacy`（親密度）、`power`（力関係：高いほどその話者が優位と感じる）、`social_distance`（社会的距離：高いほど距離がある）、`trust`（信頼）。
- 用例: UI 表示やトランスクリプトの注釈では `by_speaker` を優先的に参照し、無い場合は `a_to_b`/`b_to_a` を用いてください。

注意:

- `a_to_b`/`b_to_a` と `by_speaker` の値が不整合になっているケースは稀ですが得られた場合は `by_speaker` の方を正とし、ログや検査で差分を確認してください。

## セッション内暫定関係評価（intra-relationships）

目的:

- セッション終了時の最終評価だけでなく、セッション途中の会話進行に合わせて関係値を暫定的に更新し、以後の発話生成に反映させるための低コストな仕組みです。

主要挙動（簡潔）:

- `--intra-relationships` を有効にすると、セッション内で N ターンごとに `get_session_relationships` を呼び出して暫定の関係値を算出・保存します。
- デフォルトの間隔は `--intra-frequency 5`（5 ターンごと）。頻度を上げるほど LLM 呼び出し回数が増えます。
- 算出された暫定関係値は `agent_a` / `agent_b` の `session_{n}_relationships` に保存され、以後の `get_agent_query` 呼び出しに短い "Relationship snapshot" テキストとして渡されます。

CLI フラグ:

- `--intra-relationships` : セッション内での中間評価を有効にする。
- `--intra-frequency N` : N ターンごとに評価を行う（デフォルト 5）。

振る舞いの詳細:

- 実装は最小変更方針に従い、セッションループ（`get_session` 内）で i (現在ターン) が `i % intra_frequency == 0` のときに mid-session 評価を実行します。
- mid-session 評価は会話のこれまで（`session_dialog` 引数）を渡して行われるため、最終評価に近い暫定スナップショットが得られます。
- 評価は A→B と B→A の双方を LLM に問い、返ってきた JSON をそのまま `session_{n}_relationships` として保存します。失敗した場合はログに警告を出し、生成は継続されます。

プロンプトへの反映方法:

- `get_agent_query(..., relationships=...)` という引数が渡されると、プロンプト末尾に短い "Relationship snapshot" テキストが付与されます。例:

  Relationship snapshot:
  Alice -> Bob: intimacy=6, power=2, social_distance=4, trust=7
  Bob -> Alice: intimacy=5, power=4, social_distance=5, trust=6

- LLM には「トーンや礼儀（高い親密度 → よりカジュアル、高い力関係 → より命令的等）を反映するように」と指示的に追記します。必要に応じてこの文面は微調整してください。

コストと注意点:

- mid-session 評価は追加の LLM 呼び出し（A→B と B→A の双方）を行うため、API コストが増加します。`--intra-frequency` を大きくして頻度を下げることでコストを制御できます。
- 一貫性: 暫定スコアは最終的なセッション評価と差が出ることがあります。UI やログでは暫定か最終かを明示してください。

ファイル保存先:

- 暫定（および最終）関係値は各エージェント JSON の `session_{n}_relationships` に保存されます。
- エクスポート（`all_sessions_export.json`）にも `relationships` フィールドとして含まれます。

利用例（短いテストラン）:

zsh 環境での簡易テスト（1 セッション、最大 6 ターン、2 ターンごとに評価）:

```bash
python locomo/generative_agents/generate_conversations.py \
  --out-dir ./out_test --prompt-dir ./prompt_examples \
  --session --num-sessions 1 --max-turns-per-session 6 \
  --intra-relationships --intra-frequency 2 --lang en
```

運用のヒント:

- 頻度を低く（大きな N）にしてコストを抑え、必要に応じて小さな N に下げて観察してください。
- より滑らかな変化（スムージング）が欲しい場合は、取得した暫定スコアに対して EMA（指数移動平均）やローカル補正を追加することを推奨します（未実装）。
