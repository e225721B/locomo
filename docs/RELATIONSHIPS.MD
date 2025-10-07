# 関係値（Relationship Scoring）

会話セッションごとに、話者 A→B／B→A の関係性を以下の 4 指標（1〜10）で評価し、保存・出力します。

- intimacy（親密度）
- power（力関係; 値が高いほど自分が優位と感じる）
- social_distance（社会的距離; 高いほど距離がある）
- trust（信頼）

## 実装概要

- 定義とロジック: `generative_agents/memory_utils.py`

  - プロンプト: `RELATIONSHIP_ASSESS_PROMPT_EN` / `RELATIONSHIP_ASSESS_PROMPT_JA`
  - 実行関数: `get_session_relationships(args, agent_a, agent_b, session_idx)`
  - 戻り値スキーマ:
    ```json
    {
      "a_to_b": { "intimacy": 7, "power": 4, "social_distance": 3, "trust": 6 },
      "b_to_a": { "intimacy": 6, "power": 5, "social_distance": 4, "trust": 7 },
      "by_speaker": {
        "<AgentAName>": {
          "toward": "<AgentBName>",
          "scores": {
            "intimacy": 7,
            "power": 4,
            "social_distance": 3,
            "trust": 6
          }
        },
        "<AgentBName>": {
          "toward": "<AgentAName>",
          "scores": {
            "intimacy": 6,
            "power": 5,
            "social_distance": 4,
            "trust": 7
          }
        }
      }
    }
    ```
  - 数値は厳密に整数へ丸め、[1,10] にクリップ。想定外の出力はデフォルト 5 にフォールバック。

- パイプライン統合: `generative_agents/generate_conversations.py`

  - CLI フラグ `--relationships` を追加。
  - 各セッションの後処理で `get_session_relationships` を呼び、結果を `session_{n}_relationships` として `agent_a.json` / `agent_b.json` に保存。
  - `all_sessions_export.json` の各セッションにも `relationships` を含めて出力。

- トランスクリプト表示: `scripts/export_transcript.py`
  - 各セッションの末尾に関係値を表示（`by_speaker` があれば実名で、無ければ A→B/B→A）。

## 使い方

- ランナー: `scripts/generate_conversations.sh` は `--relationships` を有効化済みです。他のパラメータと組み合わせて実行してください。
- 出力の確認場所:
  - `out/<run>/agent_a.json` / `agent_b.json` 内の `session_{n}_relationships`
  - `out/<run>/all_sessions_export.json`（各 `sessions[*].relationships`）
  - `scripts/export_transcript.py` 実行で生成される Markdown にも関係値が表示されます。

## 今後の拡張（任意）

- 継続的関係値: 前セッションまでの移動平均や指数移動平均による“長期的関係ベクトル”を保持し、次セッションのプロンプトへコンテキストとして与える。
- 根拠の保存: どの発話に基づきそのスコアになったのか（ダイアログ ID と理由文）を `rationale` として出力し、トランスクリプトにも併記。
- プロンプトへの直接活用: `--relationship-context` のようなフラグを追加し、直前セッションの関係値を会話生成プロンプトへ注入して応答のトーンへ反映（デフォルトは現状維持で安全に）。
