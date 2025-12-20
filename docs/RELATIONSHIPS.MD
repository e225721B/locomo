# 関係値システム（Relationship Reflection）

会話セッションにおける話者間の関係性を内省するシステムです。HLQ + 根拠パイプラインによる 3 指標での関係内省を提供します。

---

## Relationship Reflection（7 点スケール: -3〜+3）

メモリストリームから取得した根拠に基づき、3 指標で関係を内省します。

| 指標           | 説明                                             |
| -------------- | ------------------------------------------------ |
| `Power`        | 力関係（+3 で優位、-3 で劣位）                   |
| `Intimacy`     | 親密度（+3 で親密、-3 で疎遠）                   |
| `TaskOriented` | タスク指向（+3 でタスク集中、-3 で雑談・社交的） |

### 実装

- **プロンプト**: `RELN_REFLECT_PROMPT_EN` / `RELN_REFLECT_PROMPT_JA`（`memory_utils.py`）
- **関数**: `get_relationship_reflection(args, agent_a, agent_b, session_idx, target='both')`
- **パイプライン**:
  1. メモリストリームから最新 10 件を収集
  2. High-Level Questions（HLQ）を生成
  3. 類似度・重要度・再近性でスコアリングし Top-K を根拠として選択
  4. HLQ と根拠からプロンプト生成 → LLM で 7 点スケールの値を取得
- **戻り値スキーマ**:
  ```json
  {
    "a_to_b": { "Power": 1, "Intimacy": 2, "TaskOriented": -1 },
    "b_to_a": { "Power": -1, "Intimacy": 2, "TaskOriented": 0 },
    "by_speaker": {
      "<AgentAName>": {
        "toward": "<AgentBName>",
        "vector": { "Power": 1, "Intimacy": 2, "TaskOriented": -1 }
      },
      "<AgentBName>": {
        "toward": "<AgentAName>",
        "vector": { "Power": -1, "Intimacy": 2, "TaskOriented": 0 }
      }
    }
  }
  ```
- 数値は整数に丸め、[-3, +3] にクリップ

### CLI フラグ

| フラグ                                 | 説明                                                         |
| -------------------------------------- | ------------------------------------------------------------ |
| `--relationship-reflection`            | 7 点スケールの関係内省を有効化                               |
| `--relationship-reflection-every-turn` | 毎ターン、発話者視点で内省を計算                             |
| `--relationship-reflection-no-inject`  | 内省は計算するがプロンプトには注入しない（アブレーション用） |

---

## パイプライン統合

`generative_agents/generate_conversations.py` での統合:

- **Assessment**: `--relationships` 有効時、各セッション終了後に `get_session_relationships()` を呼び出し
- **Reflection**: `--relationship-reflection` 有効時、HLQ + 根拠パイプラインで内省値を計算
- 結果は `session_{n}_relationships` / `session_{n}_relationship_reflection` として `agent_a.json` / `agent_b.json` に保存
- `--relationship-reflection-no-inject` がない場合、内省値は次ターンのプロンプトに注入される

### プロンプトへの注入

`get_agent_query()` 関数で以下を追加:

```
Relationship reflection (7-point -3..+3):
Alice -> Bob: Power=1, Intimacy=2, TaskOriented=0
この値を参考に、力関係・親密度・タスク指向度を調整してください。
```

---

## トランスクリプト表示

`scripts/export_transcript.py` で各セッションの末尾に関係値を表示:

```markdown
**Relationship scores:**

- Alice → Bob: intimacy=7, power=4, social_distance=3, trust=6
- Bob → Alice: intimacy=6, power=5, social_distance=4, trust=7
```

---

## 使い方

### 基本実行

```bash
# generate_conversations.sh はデフォルトで relationship-reflection を有効化
bash scripts/generate_conversations.sh --persona-dir personas/senpai_kouhai
```

### 現在のデフォルト設定（`generate_conversations.sh`）

```bash
python3 generative_agents/generate_conversations.py \
    --session --reflection --memory-topk 3 \
    --num-sessions 3 --exact-turns-per-session 10 \
    --relationship-reflection --relationship-reflection-every-turn \
    --relationship-reflection-no-inject \
    --session-topics-file ./topics/relationship_senpai_tekitai.json \
    --lang ja
```

### 出力の確認場所

| ファイル                                      | 内容                                                               |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `out/<run>/agent_a.json`                      | `session_{n}_relationships`, `session_{n}_relationship_reflection` |
| `out/<run>/rr_prompt_trace_session_{n}.jsonl` | 関係内省のプロンプトトレース（デバッグ用）                         |
| `out/<run>/agent_a_transcript.md`             | 人間可読なトランスクリプト                                         |

### 結果の可視化

```bash
python scripts/plot_relationship_reflection.py \
    out/with_inject/agent_a.json \
    out/without_inject/agent_a.json \
    --labels "注入あり" "注入なし" \
    -o comparison.png
```

---

## 関連ドキュメント

- [RELATIONSHIP_REFLECTION_FLOW.md](./RELATIONSHIP_REFLECTION_FLOW.md): 関係内省の詳細フロー
- [GENERATION_METHOD.md](./GENERATION_METHOD.md): 会話生成パイプラインの全体像
