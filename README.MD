# LoCoMo: LLM Agent 長期会話メモリ評価フレームワーク

**ACL 2024 論文 "Evaluating Very Long-Term Conversational Memory of LLM Agents" のデータとコード**

**原著者**: [Adyasha Maharana](https://adymaharana.github.io/), [Dong-Ho Lee](https://www.danny-lee.info/), [Sergey Tulyakov](https://stulyakov.com/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/), [Francesco Barbieri](https://fvancesco.github.io/), [Yuwei Fang](https://yuwfan.github.io/)

**論文**: [pdf](https://github.com/snap-research/locomo/tree/main/static/paper/locomo.pdf)

---

## 拡張機能

このフォークでは、以下の機能を追加しています：

### 1. 関係値追跡システム (Relationship Reflection)

- **3 次元の関係性評価**: Power（力関係）、Intimacy（親密度）、TaskOriented（タスク指向）
- **7 点スケール**: -3（低い）〜 +3（高い）
- **毎ターン評価**: 各発話後に話者が相手に対する関係値を評価
- **証拠ベース評価**: メモリストリームから最新の発話を根拠として使用

### 2. イベント注入システム

- **シナリオファイル**: JSON で事前定義されたイベントを特定のターンで注入
- **継続モード**: セッション間でメモリと関係値を維持
- **イベントタイプ**: conflict, awkward_silence, reconciliation など

### 3. 日本語対応

- `--lang ja` オプションで日本語会話を生成
- プロンプトとシステムメッセージの日本語翻訳

---

## データ

LoCoMo データセットは `./data/locomo10.json` に含まれています。各サンプルは会話とそのアノテーションを含みます：

- `sample_id`: サンプル識別子
- `conversation`: セッションリストとタイムスタンプ
- `observation`: 各セッションの観察（生成）
- `session_summary`: セッションレベル要約（生成）
- `event_summary`: 各話者の重要イベント（アノテーション）
- `qa`: 質問応答アノテーション

---

## 環境設定

```bash
# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt

# 環境変数の設定
cp scripts/env.sh.example scripts/env.sh
# scripts/env.sh を編集してAPIキーを設定
```

---

## 会話生成

### 基本的な使用方法

```bash
bash scripts/generate_conversations.sh \
  --out-dir ./out/my_conversation \
  --scenario-file ./topics/doraemon/intimacy.json \
  --persona-dir personas/doraemon
```

### シナリオファイル形式

シナリオファイル（JSON）で会話の設定とイベントを定義します：

```json
{
  "mode": "continuous",
  "settings": {
    "lang": "ja",
    "num_sessions": 1,
    "total_turns": 40,
    "event_interval": 20,
    "memory_stream": true,
    "relationship_reflection": true,
    "relationship_reflection_every_turn": true,
    "reflection_evidence_topk": 3
  },
  "events": [
    {
      "turn": 0,
      "type": "conflict",
      "description": "最初のイベント説明..."
    },
    {
      "turn": 20,
      "type": "reconciliation",
      "description": "中間イベント説明..."
    }
  ]
}
```

#### 設定パラメータ

| パラメータ                           | 説明                           | デフォルト |
| ------------------------------------ | ------------------------------ | ---------- |
| `lang`                               | 言語 (`en` / `ja`)             | `en`       |
| `num_sessions`                       | セッション数                   | 1          |
| `total_turns`                        | 総ターン数                     | 40         |
| `event_interval`                     | イベント切り替え間隔           | 20         |
| `memory_stream`                      | メモリストリーム有効化         | false      |
| `relationship_reflection`            | 関係値追跡有効化               | false      |
| `relationship_reflection_every_turn` | 毎ターン関係値評価             | false      |
| `relationship_reflection_no_inject`  | 関係値をプロンプトに注入しない | false      |
| `reflection_evidence_topk`           | 証拠として使用する発話数       | 3          |
| `memory_topk`                        | 取得するメモリ数               | 0          |

### ペルソナファイル形式

各エージェントのペルソナを `agent_a.json` / `agent_b.json` で定義：

```json
{
  "name": "ドラえもん",
  "persona_summary": "22世紀の未来から来たネコ型ロボット。のび太の世話係として送り込まれた。",
  "age": 12,
  "gender": "male",
  "occupation": "子守り用ロボット",
  "speaking_style": "親しみやすく、保護者的。「〜だよ」「〜だね」という語尾が多い。",
  "interests": ["どら焼き", "のび太の世話"],
  "notes": "一人称は「ぼく」。二人称は「きみ」または「のび太くん」。"
}
```

### 主要オプション

```bash
# 基本オプション
--out-dir PATH          # 出力ディレクトリ
--scenario-file PATH    # シナリオファイル
--persona-dir PATH      # ペルソナディレクトリ
--lang {en,ja}          # 言語

# 会話制御
--exact-turns-per-session N   # 正確なターン数を指定
--min-turns-before-stop N     # 最小ターン数

# 関係値追跡
--relationship-reflection            # 関係値追跡を有効化
--relationship-reflection-every-turn # 毎ターン評価
--relationship-reflection-no-inject  # プロンプト注入なし（アブレーション用）
--reflection-evidence-topk N         # 証拠として使用する発話数

# メモリ
--memory-stream          # メモリストリーム有効化
--memory-topk N          # 取得するメモリ数
--memory-retrieve-inject # 取得したメモリをプロンプトに注入
```

---

## 出力ファイル

生成された会話は以下のファイルに保存されます：

| ファイル                                | 説明                                   |
| --------------------------------------- | -------------------------------------- |
| `all_sessions_export.json`              | 全セッションの会話データ（関係値含む） |
| `sessions.html`                         | 会話の HTML 表示                       |
| `relationship_reflection_details.jsonl` | 関係値評価の詳細ログ                   |
| `rr_prompt_trace_session_N.jsonl`       | 関係値評価のプロンプトトレース         |
| `memory_stream_agent_a.json`            | Agent A のメモリストリーム             |
| `memory_stream_agent_b.json`            | Agent B のメモリストリーム             |
| `prompt_utterance_pairs.jsonl`          | プロンプトと発話のペア                 |

---

## 関係値推移グラフ

生成された会話の関係値推移をグラフ化するスクリプトを用意しています：

```bash
# 基本的な使用方法
python scripts/plot_relationship_from_jsonl.py --out-dir ./out/親密度・関係値あり

# オプション
python scripts/plot_relationship_from_jsonl.py \
  -o ./out/my_conversation \
  --session 1 \
  --output graph.png \
  --no-show
```

生成されるグラフ：

- **3 つのサブプロット**: Power, Intimacy, TaskOriented
- **2 方向の線**: A→B（青）と B→A（赤）
- **イベント注入マーカー**: 垂直線でイベントタイミングを表示

---

## 評価

### LoCoMo QA タスク

```bash
# OpenAI モデル
bash scripts/evaluate_gpts.sh

# Anthropic モデル
bash scripts/evaluate_claude.sh

# Gemini モデル
bash scripts/evaluate_gemini.sh

# Huggingface モデル
bash scripts/evaluate_hf_llm.sh
```

### RAG 評価

```bash
# 観察の生成
bash scripts/generate_observations.sh

# セッション要約の生成
bash scripts/generate_session_summaries.sh

# RAG評価
bash scripts/evaluate_rag_gpts.sh
```

---

## サンプルペルソナ

以下のサンプルペルソナが含まれています：

| ディレクトリ              | 説明                |
| ------------------------- | ------------------- |
| `personas/doraemon/`      | ドラえもん & のび太 |
| `personas/fate/`          | 衛宮士郎 & セイバー |
| `personas/senpai_kouhai/` | 先輩 & 後輩         |

---

## 参照

このプロジェクトを使用する場合は、以下の論文を引用してください：

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```
