#!/usr/bin/env python3
"""
Intimacy（親密度）の関係値遷移を比較するグラフを出力するスクリプト
二つのディレクトリから prompt_trace_session_1.jsonl を読み込み、
a_to_b と b_to_a の Intimacy 値を同一グラフ上で比較表示します。
イベント切り替えポイントも縦線で表示します。

各ペルソナは60ターン中30回発話するため、各方向の関係値は30データポイントとなります。
X軸は実際のターン番号（0-59）を使用します。
"""

import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 日本語フォント設定（macOS）
plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_relationship_from_prompt_trace(out_dir: str) -> dict:
    """prompt_trace_session_1.jsonl から全ターンの関係値を読み込む
    
    Args:
        out_dir: 出力ディレクトリのパス
    
    Returns:
        {'a_to_b': [(turn, intimacy), ...], 'b_to_a': [(turn, intimacy), ...]}
        A（偶数ターン発話者）とB（奇数ターン発話者）を判定
    """
    data = {'a_to_b': [], 'b_to_a': []}
    prompt_trace_path = Path(out_dir) / 'prompt_trace_session_1.jsonl'
    
    if not prompt_trace_path.exists():
        print(f"警告: {prompt_trace_path} が見つかりません")
        return data
    
    with open(prompt_trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            turn = entry.get('turn', -1)
            rr = entry.get('relationship_reflection', {})
            intimacy = rr.get('Intimacy', 0)
            
            # 偶数ターン（0,2,4...）はAの発話 → A→Bの関係値
            # 奇数ターン（1,3,5...）はBの発話 → B→Aの関係値
            if turn % 2 == 0:
                data['a_to_b'].append((turn, intimacy))
            else:
                data['b_to_a'].append((turn, intimacy))
    
    return data


def plot_intimacy_comparison(dir1: str, dir2: str, label1: str, label2: str, 
                              event_turns: list = None, event_labels: dict = None,
                              output_path: str = None):
    """2つのディレクトリの Intimacy 値を比較するグラフを作成
    
    Args:
        dir1: 比較対象ディレクトリ1（実線で表示）
        dir2: 比較対象ディレクトリ2（点線で表示）
        label1: dir1のラベル
        label2: dir2のラベル
        event_turns: イベント切り替えターンのリスト（縦線を表示）
        event_labels: イベントのラベル辞書 {turn: label}
        output_path: 出力ファイルパス
    """
    
    # データ読み込み（prompt_trace から実際のターン番号と値を取得）
    data1 = load_relationship_from_prompt_trace(dir1)
    data2 = load_relationship_from_prompt_trace(dir2)
    
    # デフォルトのイベントラベル
    if event_labels is None:
        event_labels = {
            0: "衝突",
            20: "停電・閉じ込め",
            40: "脱出・帰り道"
        }
    
    # データからターン番号と値を分離
    turns1_ab = [t for t, v in data1['a_to_b']]
    vals1_ab = [v for t, v in data1['a_to_b']]
    turns2_ab = [t for t, v in data2['a_to_b']]
    vals2_ab = [v for t, v in data2['a_to_b']]
    turns1_ba = [t for t, v in data1['b_to_a']]
    vals1_ba = [v for t, v in data1['b_to_a']]
    turns2_ba = [t for t, v in data2['b_to_a']]
    vals2_ba = [v for t, v in data2['b_to_a']]
    
    # グラフ作成（2行1列のサブプロット: A→B と B→A）
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Intimacy（親密度）の遷移比較\n60ターン（各ペルソナ30発話）', fontsize=16, fontweight='bold')
    
    # 上: A → B（偶数ターン：0,2,4,...,58）
    ax1 = axes[0]
    
    # dir1 は実線、dir2 は点線
    ax1.plot(turns1_ab, vals1_ab, 'b-o', markersize=3, linewidth=1.5, 
             label=f'{label1}', alpha=0.8)
    ax1.plot(turns2_ab, vals2_ab, 'r--s', markersize=3, linewidth=1.5, 
             label=f'{label2}', alpha=0.8)
    
    ax1.set_title('A → B（アスカ → シンジ への親密度）', fontsize=12)
    ax1.set_xlabel('ターン')
    ax1.set_ylabel('Intimacy (-3 〜 +3)')
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_xlim(0, 60)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # イベント切り替えの縦線を追加
    if event_turns:
        for turn in event_turns:
            ax1.axvline(x=turn, color='green', linestyle=':', alpha=0.7, linewidth=2)
            if turn in event_labels:
                ax1.text(turn + 0.5, 3.2, event_labels[turn], fontsize=9, 
                        color='green', rotation=0, ha='left')
    
    # 下: B → A（奇数ターン：1,3,5,...,59）
    ax2 = axes[1]
    
    # dir1 は実線、dir2 は点線
    ax2.plot(turns1_ba, vals1_ba, 'b-o', markersize=3, linewidth=1.5, 
             label=f'{label1}', alpha=0.8)
    ax2.plot(turns2_ba, vals2_ba, 'r--s', markersize=3, linewidth=1.5, 
             label=f'{label2}', alpha=0.8)
    
    ax2.set_title('B → A（シンジ → アスカ への親密度）', fontsize=12)
    ax2.set_xlabel('ターン')
    ax2.set_ylabel('Intimacy (-3 〜 +3)')
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_xlim(0, 60)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # イベント切り替えの縦線を追加
    if event_turns:
        for turn in event_turns:
            ax2.axvline(x=turn, color='green', linestyle=':', alpha=0.7, linewidth=2)
            if turn in event_labels:
                ax2.text(turn + 0.5, 3.2, event_labels[turn], fontsize=9, 
                        color='green', rotation=0, ha='left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"グラフを保存しました: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Intimacy（親密度）の関係値遷移を比較するグラフを出力')
    parser.add_argument('--dir1', type=str, default='./out/eva_intimacy',
                        help='比較対象ディレクトリ1（実線で表示）')
    parser.add_argument('--dir2', type=str, default='./out/eva_intimacy_inject',
                        help='比較対象ディレクトリ2（点線で表示）')
    parser.add_argument('--label1', type=str, default='関係値注入なし',
                        help='ディレクトリ1のラベル')
    parser.add_argument('--label2', type=str, default='関係値注入あり',
                        help='ディレクトリ2のラベル')
    parser.add_argument('--events', type=str, default='0,20,40',
                        help='イベント切り替えターン（カンマ区切り）')
    parser.add_argument('--output', type=str, default=None,
                        help='出力ファイルパス（PNG）')
    
    args = parser.parse_args()
    
    # イベントターンをパース
    event_turns = [int(t.strip()) for t in args.events.split(',') if t.strip()]
    
    plot_intimacy_comparison(args.dir1, args.dir2, args.label1, args.label2, 
                              event_turns, None, args.output)


if __name__ == '__main__':
    main()


"""
python relationships_plot/plot_intimacy.py \
  --dir1 ./out/eva_intimacy \
  --dir2 ./out/eva_intimacy_inject \
  --label1 "注入なし" \
  --label2 "注入あり" \
  --output ./graph/intimacy_comparison.png
"""
