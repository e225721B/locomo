#!/usr/bin/env python3
"""
関係値内省 (Relationship Reflection) の推移を可視化するスクリプト

使用方法:
    python scripts/plot_relationship_reflection.py out/senpai_kouhai/agent_a.json

    # 複数の出力ディレクトリを比較（注入あり vs 注入なし）
    python scripts/plot_relationship_reflection.py \
        out/with_inject/agent_a.json \
        out/without_inject/agent_a.json \
        --labels "Injected" "Not Injected"

    # セッションを指定
    python scripts/plot_relationship_reflection.py out/test/agent_a.json --sessions 1 2 3
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    # 日本語フォント設定 (macOS) - より汎用的な設定
    import platform
    if platform.system() == 'Darwin':  # macOS
        # 利用可能なフォントを試す
        available_fonts = ['Hiragino Maru Gothic Pro', 'AppleGothic', 'Arial Unicode MS']
        for font in available_fonts:
            try:
                matplotlib.rcParams['font.family'] = font
                break
            except:
                continue
    # 警告を抑制
    import warnings
    warnings.filterwarnings('ignore', message='.*font.*')
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def load_agent_data(filepath: str) -> Dict[str, Any]:
    """エージェントのJSONファイルを読み込む"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_rr_timeline(agent: Dict[str, Any], session_ids: Optional[List[int]] = None) -> Dict[str, List[Dict]]:
    """
    エージェントデータから関係値内省のタイムラインを抽出する
    
    Returns:
        {
            'speaker_name': [
                {'session': 1, 'turn': 0, 'Intimacy': 1, 'Power': 2, 'injected': True},
                ...
            ]
        }
    """
    timelines = {}
    
    # 全てのセッションの relationship_reflection_turns を探す
    for key in agent.keys():
        if '_relationship_reflection_turns' in key:
            # session_X_relationship_reflection_turns からセッション番号を抽出
            try:
                session_id = int(key.split('_')[1])
            except (ValueError, IndexError):
                continue
            
            # セッションフィルタ
            if session_ids and session_id not in session_ids:
                continue
            
            turns_data = agent[key]
            if not isinstance(turns_data, list):
                continue
            
            for entry in turns_data:
                if not isinstance(entry, dict):
                    continue
                
                speaker = entry.get('next_speaker', 'unknown')
                turn = entry.get('turn', 0)
                injected = entry.get('injected', True)  # デフォルトは注入あり（後方互換）
                rr = entry.get('rr', {})
                
                # a_to_b と b_to_a を取得
                for direction in ['a_to_b', 'b_to_a']:
                    vec = rr.get(direction, {})
                    if not vec:
                        continue
                    
                    # by_speaker から話者名を特定
                    by_speaker = rr.get('by_speaker', {})
                    for spk_name, spk_data in by_speaker.items():
                        spk_vec = spk_data.get('vector')
                        if not spk_vec:  # None または空の場合はスキップ
                            continue
                        # 3次元対応: Power, Intimacy, TaskOriented（後方互換も維持）
                        power = spk_vec.get('Power', spk_vec.get('Self-Disclosure', spk_vec.get('positivity', None)))
                        intimacy = spk_vec.get('Intimacy', spk_vec.get('Politeness', spk_vec.get('attentiveness', None)))
                        task_oriented = spk_vec.get('TaskOriented', spk_vec.get('GoalOrientation', None))
                        
                        if power is None and intimacy is None and task_oriented is None:
                            continue
                        
                        if spk_name not in timelines:
                            timelines[spk_name] = []
                        
                        # 重複を避ける（同じセッション・ターンのエントリ）
                        existing = [e for e in timelines[spk_name] if e['session'] == session_id and e['turn'] == turn]
                        if not existing:
                            timelines[spk_name].append({
                                'session': session_id,
                                'turn': turn,
                                'Power': power,
                                'Intimacy': intimacy,
                                'TaskOriented': task_oriented,
                                'injected': injected,
                                'toward': spk_data.get('toward', 'unknown')
                            })
    
    # ソート
    for spk in timelines:
        timelines[spk].sort(key=lambda x: (x['session'], x['turn']))
    
    return timelines


def plot_single_file(filepath: str, session_ids: Optional[List[int]] = None, output_path: Optional[str] = None):
    """単一ファイルの関係値推移をプロット"""
    agent = load_agent_data(filepath)
    timelines = extract_rr_timeline(agent, session_ids)
    
    if not timelines:
        print(f"No relationship reflection data found in {filepath}")
        return
    
    # 3次元対応: TaskOrientedがあれば3段、なければ2段
    has_task = any(e.get('TaskOriented') is not None for data in timelines.values() for e in data)
    num_plots = 3 if has_task else 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
    
    colors = plt.cm.tab10.colors
    
    for idx, (speaker, data) in enumerate(timelines.items()):
        color = colors[idx % len(colors)]
        
        # X軸: セッション内ターンを通し番号に変換
        x_vals = []
        power_vals = []
        intimacy_vals = []
        task_vals = []
        x_labels = []
        
        for i, entry in enumerate(data):
            x_vals.append(i)
            power_vals.append(entry.get('Power'))
            intimacy_vals.append(entry.get('Intimacy'))
            task_vals.append(entry.get('TaskOriented'))
            x_labels.append(f"S{entry['session']}:T{entry['turn']}")
        
        marker = 'o' if data[0].get('injected', True) else 'x'
        direction_label = f"{speaker} → {data[0].get('toward', '?')}"
        
        # Power プロット
        axes[0].plot(x_vals, power_vals, marker=marker, color=color, label=direction_label, linewidth=2, markersize=6)
        
        # Intimacy プロット
        axes[1].plot(x_vals, intimacy_vals, marker=marker, color=color, label=direction_label, linewidth=2, markersize=6)
        
        # TaskOriented プロット（あれば）
        if has_task:
            axes[2].plot(x_vals, task_vals, marker=marker, color=color, label=direction_label, linewidth=2, markersize=6)
    
    # グラフ設定
    axes[0].set_ylabel('Power (力関係)', fontsize=12)
    axes[0].set_ylim(-3.5, 3.5)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Intimacy (親密度)', fontsize=12)
    axes[1].set_ylim(-3.5, 3.5)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    if has_task:
        axes[2].set_ylabel('TaskOriented (タスク指向対話)', fontsize=12)
        axes[2].set_ylim(-3.5, 3.5)
        axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
    
    # X軸ラベル
    last_ax = axes[-1]
    if x_labels:
        last_ax.set_xticks(x_vals)
        last_ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    last_ax.set_xlabel('Session:Turn', fontsize=12)
    
    # タイトル
    basename = os.path.basename(os.path.dirname(filepath))
    injected_status = "Injected" if data[0].get('injected', True) else "NOT Injected"
    fig.suptitle(f'Relationship Reflection 推移\n({basename}, {injected_status})', fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_comparison(filepaths: List[str], labels: List[str], session_ids: Optional[List[int]] = None, output_path: Optional[str] = None):
    """複数ファイルの関係値推移を比較プロット"""
    
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10.colors
    
    # まず全ファイルを読んで TaskOriented があるか確認し、方向ごとの色マッピングを作成
    all_timelines = []
    has_task = False
    direction_set = set()  # 全ての方向 (speaker→toward) を収集
    
    for filepath in filepaths:
        agent = load_agent_data(filepath)
        timelines = extract_rr_timeline(agent, session_ids)
        all_timelines.append(timelines)
        if timelines:
            for speaker, data in timelines.items():
                if data:
                    toward = data[0].get('toward', '?')
                    direction_set.add(f"{speaker}→{toward}")
                if any(e.get('TaskOriented') is not None for e in data):
                    has_task = True
    
    # 方向ごとに固定の色を割り当て
    direction_list = sorted(direction_set)
    direction_colors = {d: colors[i % len(colors)] for i, d in enumerate(direction_list)}
    
    num_plots = 3 if has_task else 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
    
    all_x_labels = []
    max_x = 0
    has_data = False
    
    for file_idx, (filepath, label, timelines) in enumerate(zip(filepaths, labels, all_timelines)):
        if not timelines:
            print(f"Warning: No relationship reflection data found in {filepath}")
            print(f"  Hint: Make sure --relationship-reflection --relationship-reflection-every-turn were used during generation.")
            continue
        
        has_data = True
        linestyle = linestyles[file_idx % len(linestyles)]
        
        for spk_idx, (speaker, data) in enumerate(timelines.items()):
            toward = data[0].get('toward', '?') if data else '?'
            direction_key = f"{speaker}→{toward}"
            color = direction_colors.get(direction_key, colors[spk_idx % len(colors)])
            
            x_vals = list(range(len(data)))
            power_vals = [e.get('Power') for e in data]
            intimacy_vals = [e.get('Intimacy') for e in data]
            task_vals = [e.get('TaskOriented') for e in data]
            x_labels = [f"S{e['session']}:T{e['turn']}" for e in data]
            
            if len(x_vals) > max_x:
                max_x = len(x_vals)
                all_x_labels = x_labels
            
            marker = 'o' if data[0].get('injected', True) else 'x'
            
            # ラベルに「誰→誰」を含める
            direction_label = f"{speaker}→{toward}"
            
            # Power
            axes[0].plot(x_vals, power_vals, marker=marker, color=color, linestyle=linestyle,
                        label=f"{label}: {direction_label}", linewidth=2, markersize=5, alpha=0.8)
            
            # Intimacy
            axes[1].plot(x_vals, intimacy_vals, marker=marker, color=color, linestyle=linestyle,
                        label=f"{label}: {direction_label}", linewidth=2, markersize=5, alpha=0.8)
            
            # TaskOriented（あれば）
            if has_task:
                axes[2].plot(x_vals, task_vals, marker=marker, color=color, linestyle=linestyle,
                            label=f"{label}: {direction_label}", linewidth=2, markersize=5, alpha=0.8)
    
    # グラフ設定
    if not has_data:
        print("Error: No relationship reflection data found in any of the input files.")
        print("Make sure the conversation was generated with --relationship-reflection --relationship-reflection-every-turn")
        plt.close(fig)
        return
    
    axes[0].set_ylabel('Power (力関係)', fontsize=12)
    axes[0].set_ylim(-3.5, 3.5)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Intimacy (親密度)', fontsize=12)
    axes[1].set_ylim(-3.5, 3.5)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    if has_task:
        axes[2].set_ylabel('TaskOriented (タスク指向対話)', fontsize=12)
        axes[2].set_ylim(-3.5, 3.5)
        axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[2].legend(loc='upper right', fontsize=9)
        axes[2].grid(True, alpha=0.3)
    
    last_ax = axes[-1]
    if all_x_labels:
        last_ax.set_xticks(range(len(all_x_labels)))
        last_ax.set_xticklabels(all_x_labels, rotation=45, ha='right', fontsize=8)
    last_ax.set_xlabel('Session:Turn', fontsize=12)
    
    fig.suptitle('Relationship Reflection 比較\n(o: 注入あり, x: 注入なし)', fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    else:
        plt.show()


def export_csv(filepath: str, session_ids: Optional[List[int]] = None, output_path: Optional[str] = None):
    """関係値データをCSVにエクスポート"""
    agent = load_agent_data(filepath)
    timelines = extract_rr_timeline(agent, session_ids)
    
    if not timelines:
        print(f"No relationship reflection data found in {filepath}")
        return
    
    if output_path is None:
        output_path = filepath.replace('.json', '_rr_timeline.csv')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('speaker,toward,session,turn,power,intimacy,task_oriented,injected\n')
        for speaker, data in timelines.items():
            for entry in data:
                power = entry.get('Power', '')
                intimacy = entry.get('Intimacy', '')
                task = entry.get('TaskOriented', '')
                f.write(f"{speaker},{entry.get('toward','?')},{entry['session']},{entry['turn']},{power},{intimacy},{task},{entry.get('injected', True)}\n")
    
    print(f"Exported CSV to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='関係値内省の推移を可視化')
    parser.add_argument('files', nargs='+', help='エージェントJSONファイルのパス')
    parser.add_argument('--labels', nargs='*', help='比較時の各ファイルのラベル')
    parser.add_argument('--sessions', nargs='*', type=int, help='表示するセッションID（指定しなければ全て）')
    parser.add_argument('--output', '-o', type=str, help='出力画像ファイルパス (PNG/PDF)')
    parser.add_argument('--csv', action='store_true', help='CSVファイルとしてエクスポート')
    
    args = parser.parse_args()
    
    if args.csv:
        for filepath in args.files:
            export_csv(filepath, args.sessions)
        return
    
    if len(args.files) == 1:
        plot_single_file(args.files[0], args.sessions, args.output)
    else:
        labels = args.labels if args.labels else [os.path.basename(os.path.dirname(f)) for f in args.files]
        if len(labels) < len(args.files):
            labels.extend([f"File{i}" for i in range(len(labels), len(args.files))])
        plot_comparison(args.files, labels, args.sessions, args.output)


if __name__ == '__main__':
    main()
