#!/usr/bin/env python3
"""
関係値内省 (Relationship Reflection) の推移を2つのプロジェクト間で比較するスクリプト

使用方法:
    python scripts/compare_relationship_reflection.py \
        --inject-dir ./out/tekitai_test_投げ込みプロンプト確認_2 \
        --no-inject-dir ./out/tekitai_test_投げ込みプロンプト確認_no_inject_2 \
        --output ./out/comparison_graph.png
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib
    # 日本語フォント設定 (macOS)
    import platform
    if platform.system() == 'Darwin':
        available_fonts = ['Hiragino Maru Gothic Pro', 'AppleGothic', 'Arial Unicode MS']
        for font in available_fonts:
            try:
                matplotlib.rcParams['font.family'] = font
                break
            except:
                continue
    import warnings
    warnings.filterwarnings('ignore', message='.*font.*')
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def load_agent_data(dirpath: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """出力ディレクトリからエージェントデータを読み込む"""
    agent_a_path = os.path.join(dirpath, 'agent_a.json')
    agent_b_path = os.path.join(dirpath, 'agent_b.json')
    
    with open(agent_a_path, 'r', encoding='utf-8') as f:
        agent_a = json.load(f)
    with open(agent_b_path, 'r', encoding='utf-8') as f:
        agent_b = json.load(f)
    
    return agent_a, agent_b


def extract_rr_timeline(agent_a: Dict, agent_b: Dict) -> Dict[str, List[Dict]]:
    """
    エージェントデータから関係値内省のタイムラインを抽出する
    
    Returns:
        {
            'a_to_b': [{'turn': 0, 'Intimacy': 1, 'Power': 2, 'TaskOriented': 0}, ...],
            'b_to_a': [{'turn': 0, 'Intimacy': 0, 'Power': -1, 'TaskOriented': 1}, ...]
        }
    """
    timelines = {'a_to_b': [], 'b_to_a': []}
    
    # agent_a から relationship_reflection_turns を探す
    for key in agent_a.keys():
        if '_relationship_reflection_turns' in key:
            turns_data = agent_a[key]
            if not isinstance(turns_data, list):
                continue
            
            for entry in turns_data:
                if not isinstance(entry, dict):
                    continue
                
                turn = entry.get('turn', 0)
                rr = entry.get('rr', {})
                
                # a_to_b
                a_to_b = rr.get('a_to_b', {})
                if a_to_b:
                    timelines['a_to_b'].append({
                        'turn': turn,
                        'Intimacy': a_to_b.get('Intimacy', 0),
                        'Power': a_to_b.get('Power', 0),
                        'TaskOriented': a_to_b.get('TaskOriented', 0)
                    })
                
                # b_to_a
                b_to_a = rr.get('b_to_a', {})
                if b_to_a:
                    timelines['b_to_a'].append({
                        'turn': turn,
                        'Intimacy': b_to_a.get('Intimacy', 0),
                        'Power': b_to_a.get('Power', 0),
                        'TaskOriented': b_to_a.get('TaskOriented', 0)
                    })
    
    # ターン順にソート
    for key in timelines:
        timelines[key] = sorted(timelines[key], key=lambda x: x['turn'])
    
    return timelines


def plot_comparison(inject_timelines: Dict, no_inject_timelines: Dict, 
                   agent_a_name: str, agent_b_name: str,
                   output_path: str, show: bool = True):
    """2つの条件の関係値推移を比較プロット"""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'関係値推移の比較: 注入あり vs 注入なし\n({agent_a_name} ↔ {agent_b_name})', 
                 fontsize=14, fontweight='bold')
    
    dimensions = ['Intimacy', 'Power', 'TaskOriented']
    dim_labels = {'Intimacy': '親密度 (Intimacy)', 'Power': '力関係 (Power)', 'TaskOriented': '目的指向性 (TaskOriented)'}
    colors = {'inject': '#2ecc71', 'no_inject': '#e74c3c'}
    
    for row, dim in enumerate(dimensions):
        # 左列: A → B
        ax1 = axes[row, 0]
        
        # 注入あり
        turns_inj = [d['turn'] for d in inject_timelines['a_to_b']]
        vals_inj = [d[dim] for d in inject_timelines['a_to_b']]
        ax1.plot(turns_inj, vals_inj, 'o-', color=colors['inject'], 
                label='注入あり', linewidth=2, markersize=4)
        
        # 注入なし
        turns_no = [d['turn'] for d in no_inject_timelines['a_to_b']]
        vals_no = [d[dim] for d in no_inject_timelines['a_to_b']]
        ax1.plot(turns_no, vals_no, 's--', color=colors['no_inject'], 
                label='注入なし', linewidth=2, markersize=4)
        
        ax1.set_ylabel(dim_labels[dim], fontsize=10)
        ax1.set_ylim(-3.5, 3.5)
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        if row == 0:
            ax1.set_title(f'{agent_a_name} → {agent_b_name}', fontsize=12)
            ax1.legend(loc='upper right')
        if row == 2:
            ax1.set_xlabel('ターン', fontsize=10)
        
        # 右列: B → A
        ax2 = axes[row, 1]
        
        # 注入あり
        turns_inj = [d['turn'] for d in inject_timelines['b_to_a']]
        vals_inj = [d[dim] for d in inject_timelines['b_to_a']]
        ax2.plot(turns_inj, vals_inj, 'o-', color=colors['inject'], 
                label='注入あり', linewidth=2, markersize=4)
        
        # 注入なし
        turns_no = [d['turn'] for d in no_inject_timelines['b_to_a']]
        vals_no = [d[dim] for d in no_inject_timelines['b_to_a']]
        ax2.plot(turns_no, vals_no, 's--', color=colors['no_inject'], 
                label='注入なし', linewidth=2, markersize=4)
        
        ax2.set_ylim(-3.5, 3.5)
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        if row == 0:
            ax2.set_title(f'{agent_b_name} → {agent_a_name}', fontsize=12)
            ax2.legend(loc='upper right')
        if row == 2:
            ax2.set_xlabel('ターン', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"グラフを保存しました: {output_path}")
    
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='関係値注入あり/なしの推移を比較')
    parser.add_argument('--inject-dir', required=True, 
                       help='関係値を注入した出力ディレクトリ')
    parser.add_argument('--no-inject-dir', required=True, 
                       help='関係値を注入しなかった出力ディレクトリ')
    parser.add_argument('--output', '-o', default='comparison_graph.png',
                       help='出力グラフのパス (default: comparison_graph.png)')
    parser.add_argument('--no-show', action='store_true',
                       help='グラフを表示せずファイル保存のみ')
    
    args = parser.parse_args()
    
    # データ読み込み
    print(f"注入ありデータ読み込み: {args.inject_dir}")
    inject_a, inject_b = load_agent_data(args.inject_dir)
    
    print(f"注入なしデータ読み込み: {args.no_inject_dir}")
    no_inject_a, no_inject_b = load_agent_data(args.no_inject_dir)
    
    # タイムライン抽出
    inject_timelines = extract_rr_timeline(inject_a, inject_b)
    no_inject_timelines = extract_rr_timeline(no_inject_a, no_inject_b)
    
    print(f"注入あり: a_to_b {len(inject_timelines['a_to_b'])}件, b_to_a {len(inject_timelines['b_to_a'])}件")
    print(f"注入なし: a_to_b {len(no_inject_timelines['a_to_b'])}件, b_to_a {len(no_inject_timelines['b_to_a'])}件")
    
    # エージェント名取得
    agent_a_name = inject_a.get('name', 'Agent A')
    agent_b_name = inject_b.get('name', 'Agent B')
    
    # グラフ生成
    plot_comparison(inject_timelines, no_inject_timelines, 
                   agent_a_name, agent_b_name,
                   args.output, show=not args.no_show)


if __name__ == '__main__':
    main()
