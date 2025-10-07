#!/usr/bin/env bash
set -euo pipefail

source scripts/.env.sh

python3 generative_agents/generate_conversations.py \
    --out-dir ./out/test \
    --prompt-dir ./prompt_examples \
    --session --summary --reflection --memory-stream --memory-topk 3 --num-sessions 5 \
    --persona \
    --lang ja \
    --max-turns-per-session 20 \
    "$@"

#ディレクトリを指定して実行する場合 （./out/test_runに吐き出す場合）
#bash scripts/generate_conversations.sh --out-dir ./out/test_run

#わかりやすくまとめて出力
#python3 scripts/export_transcript.py out/test_relationships2/agent_a.json out/test_relationships2/agent_a_transcript.
