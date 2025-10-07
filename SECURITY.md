# セキュリティと秘密情報の取り扱い

このリポジトリでは API キー等の秘密情報をコミットしない運用を徹底します。誤って履歴に残った場合の除去（履歴書き換え）とキーのローテーション手順もここにまとめます。

## 方針（要点）
- 秘密情報（API キー/トークン/証明書など）は Git にコミットしない
- すべての秘密は `scripts/.env`（未追跡）や OS の環境変数で管理する
- 事前検知として Git Hooks（pre-commit / pre-push）でシークレットスキャンを行う
- 万一コミットや履歴に混入した場合は、速やかに「キーのローテーション」と「履歴からの除去（rewrite）」を実施する

## セットアップ
1. Git hooks を有効化
   - `bash scripts/setup-git-hooks.sh`
   - 以降、commit/push 時に秘密情報のパターンを自動検出してブロックします
2. 環境変数の配置
   - `scripts/.env.example` をコピーして `scripts/.env` を作成し、各自のローカル秘密を設定（Git にはコミットしない）

## 秘密情報の格納場所
- ローカル: `scripts/.env`（未追跡）
- CI/CD: 各プラットフォームのシークレットストア（GitHub Actions: Encrypted Secrets など）

## よくある環境変数
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`
- `HF_TOKEN`

`scripts/env.sh` がこれらを読み込み、必要に応じてエクスポートします。`scripts/env.sh` 自体にキーを直書きしないでください。

## 履歴に混入した秘密の除去

1. 該当キーの「ローテーション / 無効化」
   - まず各プロバイダのコンソールで該当キーを失効し、新しいキーを発行
   - ローカルの `scripts/.env` を更新（新キー）、旧キーは完全に破棄

2. Git 履歴からの除去（書き換え）
   - macOS: `brew install git-filter-repo` を事前に実行
   - 本リポジトリには安全な履歴パージ補助スクリプト `scripts/purge-secret-history.sh` を同梱しています

   例A: 特定の正規表現パターン（例: Google API Key）を全履歴からマスク
   - `bash scripts/purge-secret-history.sh --pattern 'AIza[0-9A-Za-z_\-]{35}'`

   例B: 誤って追跡してしまったファイルを履歴から完全削除
   - `bash scripts/purge-secret-history.sh --remove-file 'scripts/env.sh'`

   注意: 履歴書き換えは破壊的操作です。ローカルブランチのバックアップを推奨します。

3. 強制プッシュと周知
   - 履歴書き換え後、リモートに force-push（例: `git push --force-with-lease origin <branch>`）
   - 共同開発者に履歴が書き換わった旨を周知し、各自 `git fetch --all` と安全なリベース/クローンし直しを依頼

4. 再スキャン
   - 変更後に再度スキャン（hooks の警告、もしくは任意ツール）を実行し、混入がないことを確認

## 任意の追加対策
- CI でのシークレットスキャン（例: gitleaks）
- PR テンプレートに「秘密の直書き禁止」のチェック項目を追加
- `.gitignore` に成果物・ログ・.env を確実に含める

---
不明点があれば、メンテナに連絡してください。
