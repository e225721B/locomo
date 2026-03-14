#!/usr/bin/env python3
"""
Power（力関係）の関係値遷移を比較するグラフを出力するスクリプト
二つのディレクトリから relationship_reflection_details.jsonl を読み込み、
a_to_b と b_to_a の Power 値を折れ線グラフで表示します。
"""

import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 日本語フォント設定（macOS）
plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_relationship_data(jsonl_path: str) -> dict:
    """relationship_reflection_details.jsonl を読み込み、ターンごとの関係値を抽出"""
    data = {'a_to_b': [], 'b_to_a': []}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            direction = entry.get('direction')
            values = entry.get('generated_values', {})
            power = values.get('Power', 0)
            
            if direction == 'a_to_b':
                data['a_to_b'].append(power)
            elif direction == 'b_to_a':
                data['b_to_a'].append(power)
    
    return data


def plot_power_comparison(dir1: str, dir2: str, label1: str, label2: str, output_path: str = None):
    """2つのディレクトリの Power 値を比較するグラフを作成"""
    
    # データ読み込み
    path1 = Path(dir1) / 'relationship_reflection_details.jsonl'
    path2 = Path(dir2) / 'relationship_reflection_details.jsonl'
    
    data1 = load_relationship_data(str(path1))
    data2 = load_relationship_data(str(path2))
    
    # グラフ作成（2x2のサブプロット）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Power（力関係）の遷移比較', fontsize=16, fontweight='bold')
    
    # 左上: dir1 の a_to_b
    ax1 = axes[0, 0]
    turns1_ab = list(range(len(data1['a_to_b'])))
    ax1.plot(turns1_ab, data1['a_to_b'], 'b-o', markersize=4, linewidth=1.5)
    ax1.set_title(f'{label1}\nA → B', fontsize=12)
    ax1.set_xlabel('ターン')
    ax1.set_ylabel('Power (-3 〜 +3)')
    ax1.set_ylim(-3.5, 3.5)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 右上: dir2 の a_to_b
    ax2 = axes[0, 1]
    turns2_ab = list(range(len(data2['a_to_b'])))
    ax2.plot(turns2_ab, data2['a_to_b'], 'r-o', markersize=4, linewidth=1.5)
    ax2.set_title(f'{label2}\nA → B', fontsize=12)
    ax2.set_xlabel('ターン')
    ax2.set_ylabel('Power (-3 〜 +3)')
    ax2.set_ylim(-3.5, 3.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 左下: dir1 の b_to_a
    ax3 = axes[1, 0]
    turns1_ba = list(range(len(data1['b_to_a'])))
    ax3.plot(turns1_ba, data1['b_to_a'], 'b-s', markersize=4, linewidth=1.5)
    ax3.set_title(f'{label1}\nB → A', fontsize=12)
    ax3.set_xlabel('ターン')
    ax3.set_ylabel('Power (-3 〜 +3)')
    ax3.set_ylim(-3.5, 3.5)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 右下: dir2 の b_to_a
    ax4 = axes[1, 1]
    turns2_ba = list(range(len(data2['b_to_a'])))
    ax4.plot(turns2_ba, data2['b_to_a'], 'r-s', markersize=4, linewidth=1.5)
    ax4.set_title(f'{label2}\nB → A', fontsize=12)
    ax4.set_xlabel('ターン')
    ax4.set_ylabel('Power (-3 〜 +3)')
    ax4.set_ylim(-3.5, 3.5)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"グラフを保存しました: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Power（力関係）の関係値遷移を比較するグラフを出力')
    parser.add_argument('--dir1', type=str, default='./out/eva_intimacy',
                        help='比較対象ディレクトリ1')
    parser.add_argument('--dir2', type=str, default='./out/eva_intimacy_inject',
                        help='比較対象ディレクトリ2')
    parser.add_argument('--label1', type=str, default='関係値注入なし',
                        help='ディレクトリ1のラベル')
    parser.add_argument('--label2', type=str, default='関係値注入あり',
                        help='ディレクトリ2のラベル')
    parser.add_argument('--output', type=str, default=None,
                        help='出力ファイルパス（PNG）')
    
    args = parser.parse_args()
    
    plot_power_comparison(args.dir1, args.dir2, args.label1, args.label2, args.output)


if __name__ == '__main__':
    main()
