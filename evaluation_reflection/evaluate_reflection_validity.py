#!/usr/bin/env python3
"""
内省による関係値推定の妥当性評価スクリプト（LLM as a Judge）

【概要】
システムが内省によって推定した関係値が、会話内容とイベントを踏まえて
妥当かどうかをLLM（Judge）に判定させます。

【評価の仕組み】
1. 会話履歴（根拠として使用された発話）を取得
2. 前回の関係値と今回の内省結果を取得
3. 別のLLM（Judge）に「この変化は会話内容を踏まえて妥当か」を判定させる
4. 妥当性スコアと理由を記録

【利点】
- イベント設計の質に依存しない（イベントを考慮して判定するため）
- シナリオ意図ではなく「実際の会話内容」との整合性を評価

使用方法:
    source scripts/env.sh
    python3 evaluation_reflection/evaluate_reflection_validity.py --data-dir out/eva_intimacy_increase
    python3 evaluation_reflection/evaluate_reflection_validity.py --data-dir out --all-scenarios --sample-size 10

評価指標:
    - 妥当性スコア（1-5）の平均
    - 次元別の妥当性
    - 変化方向の適切性
"""

import argparse
import json
import os
import sys
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from global_methods import run_gemini, set_gemini_key
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Gemini モデルの初期化
_gemini_model = None

def get_gemini_model():
    """Geminiモデルを取得（シングルトン）"""
    global _gemini_model
    if _gemini_model is None:
        set_gemini_key()
        model_name = os.environ.get('GEMINI_MODEL_NAME', 'gemini-2.0-flash')
        _gemini_model = genai.GenerativeModel(model_name)
    return _gemini_model


# 妥当性判定プロンプト
VALIDITY_PROMPT = """あなたは会話分析の専門家です。以下の情報を基に、キャラクターの関係値内省が妥当かを評価してください。

【評価対象】
話者: {src}
相手: {dst}

【前回の関係値】
Power: {prev_power}, Intimacy: {prev_intimacy}, TaskOriented: {prev_task}

【発生したイベント】
{event_description}

【会話履歴（内省の根拠）】
{evidence}

【今回の内省結果】
Power: {new_power} (変化: {delta_power:+d})
Intimacy: {new_intimacy} (変化: {delta_intimacy:+d})
TaskOriented: {new_task} (変化: {delta_task:+d})

【関係値の意味】
- Power（力関係）: -3=相手に従う ～ 0=対等 ～ +3=主導権を持つ
- Intimacy（親密度）: -3=冷淡 ～ 0=普通 ～ +3=親密
- TaskOriented（タスク志向）: -3=雑談中心 ～ 0=バランス ～ +3=目的達成中心

【評価基準】
各次元について、以下を評価してください：
1. イベント内容から予想される変化と、実際の変化の整合性
2. 会話履歴から読み取れる態度変化との一致
3. 変化の大きさは妥当か（急激すぎないか）
4. キャラクターの性格や関係性を考慮した解釈として適切か

【出力形式】
以下のJSON形式で出力してください：
{{
    "overall_validity": 1-5の整数（1=不適切、3=妥当、5=非常に適切）,
    "power_validity": 1-5の整数,
    "intimacy_validity": 1-5の整数,
    "task_validity": 1-5の整数,
    "expected_delta_power": イベントから予測される変化（-3〜+3の整数）,
    "expected_delta_intimacy": イベントから予測される変化（-3〜+3の整数）,
    "expected_delta_task": イベントから予測される変化（-3〜+3の整数）,
    "reasoning": "100文字以内の日本語での理由"
}}

JSONのみを出力してください。
"""


def load_reflection_data(data_dir: str) -> List[Dict]:
    """relationship_reflection_details.jsonlを読み込む"""
    filepath = os.path.join(data_dir, 'relationship_reflection_details.jsonl')
    if not os.path.exists(filepath):
        return []
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                entry = json.loads(line)
                entry['_line_index'] = idx
                data.append(entry)
    return data


def load_event_history(data_dir: str) -> Dict[Tuple[int, int], Dict]:
    """all_sessions_export.jsonからイベント履歴を読み込む
    
    Returns:
        {(session_idx, turn): event_info, ...}
    """
    filepath = os.path.join(data_dir, 'all_sessions_export.json')
    if not os.path.exists(filepath):
        return {}
    
    event_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for session in data.get('sessions', []):
        session_idx = session.get('session_id', 0)
        for event in session.get('event_history', []):
            turn = event.get('turn', 0)
            event_map[(session_idx, turn)] = {
                'type': event.get('type', ''),
                'description': event.get('description', ''),
                'relationship_reflection': event.get('relationship_reflection', {})
            }
    
    return event_map


def find_event_for_turn(event_map: Dict, session_idx: int, turn: int) -> Optional[Dict]:
    """指定ターンに最も近い（直前の）イベントを探す"""
    # 直前のイベントを探す
    best_event = None
    best_turn = -1
    
    for (s_idx, e_turn), event in event_map.items():
        if s_idx == session_idx and e_turn <= turn and e_turn > best_turn:
            best_event = event
            best_turn = e_turn
    
    return best_event


def extract_previous_values(prompt: str) -> Optional[Dict[str, int]]:
    """プロンプトから前回の関係値を抽出"""
    # パターン: 【前回の関係値】\nPower: X, Intimacy: Y, TaskOriented: Z
    pattern = r'【前回の関係値】\s*\n\s*Power:\s*(-?\d+),\s*Intimacy:\s*(-?\d+),\s*TaskOriented:\s*(-?\d+)'
    match = re.search(pattern, prompt)
    if match:
        return {
            'Power': int(match.group(1)),
            'Intimacy': int(match.group(2)),
            'TaskOriented': int(match.group(3))
        }
    return None


def extract_evidence(prompt: str) -> str:
    """プロンプトから根拠（会話履歴）を抽出"""
    # 根拠: で始まる部分から番号付きリストを抽出
    evidence_lines = []
    
    # 各行を処理
    lines = prompt.split('\n')
    in_evidence = False
    
    for line in lines:
        # 根拠: で始まる行を検出
        if '根拠:' in line:
            in_evidence = True
            continue
        
        # 根拠セクション内の番号付き行を収集
        if in_evidence:
            # 番号付きリスト（1. [名前] 内容）をマッチ
            if re.match(r'^\d+\.\s*\[.*?\]', line.strip()):
                evidence_lines.append(line.strip())
            # 次のセクション（【】で始まる or 空行が続く）に到達したら終了
            elif line.strip().startswith('【') or line.strip().startswith('前回'):
                in_evidence = False
    
    # ユニークな行のみを返す
    seen = set()
    unique_lines = []
    for line in evidence_lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    
    if unique_lines:
        return '\n'.join(unique_lines)
    return "（会話履歴なし）"


def judge_validity(entry: Dict, event_info: Optional[Dict] = None) -> Optional[Dict]:
    """LLMを使って内省の妥当性を判定
    
    Args:
        entry: relationship_reflection_details.jsonlの1エントリ
        event_info: イベント情報（type, description）
    """
    if not HAS_GEMINI:
        return None
    
    try:
        model = get_gemini_model()
        
        # 情報を抽出
        src = entry.get('src', '話者A')
        dst = entry.get('dst', '話者B')
        prompt_text = entry.get('prompt', '')
        generated = entry.get('generated_values', {})
        
        # 前回の関係値を抽出
        prev_values = extract_previous_values(prompt_text)
        if not prev_values:
            prev_values = {'Power': 0, 'Intimacy': 0, 'TaskOriented': 0}
        
        # 根拠を抽出
        evidence = extract_evidence(prompt_text)
        
        # イベント情報を構築
        if event_info and event_info.get('description'):
            event_description = f"タイプ: {event_info.get('type', '不明')}\n内容: {event_info.get('description', '')}"
        else:
            event_description = "（イベント情報なし）"
        
        # 差分を計算
        delta_power = generated.get('Power', 0) - prev_values.get('Power', 0)
        delta_intimacy = generated.get('Intimacy', 0) - prev_values.get('Intimacy', 0)
        delta_task = generated.get('TaskOriented', 0) - prev_values.get('TaskOriented', 0)
        
        # プロンプトを構築
        judge_prompt = VALIDITY_PROMPT.format(
            src=src,
            dst=dst,
            prev_power=prev_values.get('Power', 0),
            prev_intimacy=prev_values.get('Intimacy', 0),
            prev_task=prev_values.get('TaskOriented', 0),
            event_description=event_description,
            evidence=evidence,
            new_power=generated.get('Power', 0),
            new_intimacy=generated.get('Intimacy', 0),
            new_task=generated.get('TaskOriented', 0),
            delta_power=delta_power,
            delta_intimacy=delta_intimacy,
            delta_task=delta_task
        )
        
        response = run_gemini(model, judge_prompt, max_tokens=400, temperature=0.0)
        
        # JSON抽出
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # スコアを1-5の範囲にクリップ
            for key in ['overall_validity', 'power_validity', 'intimacy_validity', 'task_validity']:
                if key in result:
                    result[key] = max(1, min(5, int(result[key])))
            
            # 予測された変化量を保存（-3〜+3にクリップ）
            for key in ['expected_delta_power', 'expected_delta_intimacy', 'expected_delta_task']:
                if key in result:
                    result[key] = max(-3, min(3, int(result[key])))
            
            # 実際の変化量も保存
            result['actual_delta_power'] = delta_power
            result['actual_delta_intimacy'] = delta_intimacy
            result['actual_delta_task'] = delta_task
            
            return result
            
    except Exception as e:
        print(f"  判定エラー: {e}")
    
    return None


def evaluate_scenario(data_dir: str, sample_size: int = 20, verbose: bool = False) -> Dict:
    """1つのシナリオを評価"""
    
    # データ読み込み
    data = load_reflection_data(data_dir)
    
    if not data:
        print(f"  データなし: {data_dir}")
        return {}
    
    # イベント履歴を読み込み
    event_map = load_event_history(data_dir)
    if verbose and event_map:
        print(f"  イベント数: {len(event_map)}件")
    
    # サンプリング（均等に選択）
    if len(data) > sample_size:
        step = len(data) // sample_size
        data = data[::step][:sample_size]
    
    results = {
        'scenario': os.path.basename(data_dir),
        'total_samples': len(data),
        'evaluated': 0,
        'scores': {
            'overall': [],
            'power': [],
            'intimacy': [],
            'task': []
        },
        'prediction_accuracy': {
            'power': {'matches': 0, 'within_one': 0, 'total': 0},
            'intimacy': {'matches': 0, 'within_one': 0, 'total': 0},
            'task': {'matches': 0, 'within_one': 0, 'total': 0}
        },
        'reasonings': []
    }
    
    for i, entry in enumerate(data):
        if verbose:
            print(f"  [{i+1}/{len(data)}] 判定中...")
        
        # イベント情報を取得
        session_idx = entry.get('session_idx', 0)
        turn = entry.get('turn', 0)
        event_info = find_event_for_turn(event_map, session_idx, turn)
        
        judgment = judge_validity(entry, event_info)
        if not judgment:
            continue
        
        results['evaluated'] += 1
        results['scores']['overall'].append(judgment.get('overall_validity', 3))
        results['scores']['power'].append(judgment.get('power_validity', 3))
        results['scores']['intimacy'].append(judgment.get('intimacy_validity', 3))
        results['scores']['task'].append(judgment.get('task_validity', 3))
        
        # 予測と実際の変化を比較
        for dim, key_prefix in [('power', 'Power'), ('intimacy', 'Intimacy'), ('task', 'TaskOriented')]:
            expected = judgment.get(f'expected_delta_{dim}')
            actual = judgment.get(f'actual_delta_{dim}')
            if expected is not None and actual is not None:
                results['prediction_accuracy'][dim]['total'] += 1
                if expected == actual:
                    results['prediction_accuracy'][dim]['matches'] += 1
                if abs(expected - actual) <= 1:
                    results['prediction_accuracy'][dim]['within_one'] += 1
        
        if 'reasoning' in judgment:
            results['reasonings'].append(judgment['reasoning'])
        
        if verbose:
            print(f"    妥当性: {judgment.get('overall_validity', '?')}/5")
            # 予測 vs 実際を表示
            for dim in ['power', 'intimacy', 'task']:
                exp = judgment.get(f'expected_delta_{dim}', '?')
                act = judgment.get(f'actual_delta_{dim}', '?')
                print(f"    {dim}: 予測={exp:+d}, 実際={act:+d}" if isinstance(exp, int) and isinstance(act, int) else f"    {dim}: 予測={exp}, 実際={act}")
            if 'reasoning' in judgment:
                print(f"    理由: {judgment['reasoning'][:50]}...")
    
    # 統計を計算
    if results['scores']['overall']:
        results['statistics'] = {
            'overall_mean': statistics.mean(results['scores']['overall']),
            'power_mean': statistics.mean(results['scores']['power']),
            'intimacy_mean': statistics.mean(results['scores']['intimacy']),
            'task_mean': statistics.mean(results['scores']['task']),
        }
        if len(results['scores']['overall']) > 1:
            results['statistics']['overall_std'] = statistics.stdev(results['scores']['overall'])
        
        # 予測精度を計算
        for dim in ['power', 'intimacy', 'task']:
            total = results['prediction_accuracy'][dim]['total']
            if total > 0:
                results['prediction_accuracy'][dim]['exact_rate'] = results['prediction_accuracy'][dim]['matches'] / total
                results['prediction_accuracy'][dim]['within_one_rate'] = results['prediction_accuracy'][dim]['within_one'] / total
    
    return results


def find_all_scenarios(base_dir: str) -> List[str]:
    """すべてのシナリオディレクトリを探す"""
    scenarios = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, 'relationship_reflection_details.jsonl')):
                scenarios.append(path)
    return sorted(scenarios)


def print_report(results: List[Dict]):
    """評価レポートを出力"""
    print("\n" + "=" * 70)
    print("内省妥当性評価レポート（LLM as a Judge）")
    print("=" * 70)
    
    # 全体集計
    all_overall = []
    all_power = []
    all_intimacy = []
    all_task = []
    
    # 予測精度の集計
    pred_accuracy = {
        'power': {'exact': [], 'within_one': []},
        'intimacy': {'exact': [], 'within_one': []},
        'task': {'exact': [], 'within_one': []}
    }
    
    for res in results:
        if 'statistics' in res:
            all_overall.append(res['statistics']['overall_mean'])
            all_power.append(res['statistics']['power_mean'])
            all_intimacy.append(res['statistics']['intimacy_mean'])
            all_task.append(res['statistics']['task_mean'])
        
        # 予測精度
        if 'prediction_accuracy' in res:
            for dim in ['power', 'intimacy', 'task']:
                if res['prediction_accuracy'][dim].get('exact_rate') is not None:
                    pred_accuracy[dim]['exact'].append(res['prediction_accuracy'][dim]['exact_rate'])
                    pred_accuracy[dim]['within_one'].append(res['prediction_accuracy'][dim]['within_one_rate'])
    
    if all_overall:
        print("\n【全体サマリー】")
        print(f"  評価シナリオ数: {len(results)}")
        print(f"  平均妥当性スコア: {statistics.mean(all_overall):.2f}/5")
        print(f"    - Power: {statistics.mean(all_power):.2f}/5")
        print(f"    - Intimacy: {statistics.mean(all_intimacy):.2f}/5")
        print(f"    - TaskOriented: {statistics.mean(all_task):.2f}/5")
    
    # イベント予測 vs 内省結果の精度
    print("\n【イベント予測 vs 実際の内省結果】")
    print("  (LLMがイベントから予測した変化 vs システムが実際に内省した変化)")
    has_pred = False
    for dim in ['power', 'intimacy', 'task']:
        if pred_accuracy[dim]['exact']:
            has_pred = True
            exact_mean = statistics.mean(pred_accuracy[dim]['exact']) * 100
            within_one_mean = statistics.mean(pred_accuracy[dim]['within_one']) * 100
            print(f"    {dim}: 完全一致 {exact_mean:.1f}%, ±1以内 {within_one_mean:.1f}%")
    
    if not has_pred:
        print("    (予測データなし)")
    
    # シナリオ別
    print("\n" + "-" * 70)
    print("【シナリオ別詳細】")
    print("-" * 70)
    
    for res in results:
        scenario = res.get('scenario', 'Unknown')
        print(f"\n■ {scenario}")
        print(f"  評価数: {res.get('evaluated', 0)}/{res.get('total_samples', 0)}")
        
        if 'statistics' in res:
            stats = res['statistics']
            print(f"  妥当性スコア: {stats['overall_mean']:.2f}/5", end="")
            if 'overall_std' in stats:
                print(f" (σ={stats['overall_std']:.2f})")
            else:
                print()
            print(f"    Power: {stats['power_mean']:.2f}, Intimacy: {stats['intimacy_mean']:.2f}, Task: {stats['task_mean']:.2f}")
        
        # 予測精度を表示
        if 'prediction_accuracy' in res:
            pred = res['prediction_accuracy']
            pred_strs = []
            for dim in ['power', 'intimacy', 'task']:
                if pred[dim].get('exact_rate') is not None:
                    pred_strs.append(f"{dim}={pred[dim]['exact_rate']*100:.0f}%")
            if pred_strs:
                print(f"  予測一致率: {', '.join(pred_strs)}")
    
    # LaTeXテーブル
    print("\n" + "-" * 70)
    print("【LaTeX テーブル】")
    print("-" * 70)
    print("""
\\begin{table}[h]
\\centering
\\caption{Reflection Validity Scores (LLM as a Judge)}
\\begin{tabular}{lcccc}
\\hline
Scenario & Overall & Power & Intimacy & TaskOriented \\\\
\\hline""")
    
    for res in results:
        if 'statistics' in res:
            s = res['statistics']
            print(f"{res['scenario']} & {s['overall_mean']:.2f} & {s['power_mean']:.2f} & {s['intimacy_mean']:.2f} & {s['task_mean']:.2f} \\\\")
    
    if all_overall:
        print("\\hline")
        print(f"Average & {statistics.mean(all_overall):.2f} & {statistics.mean(all_power):.2f} & {statistics.mean(all_intimacy):.2f} & {statistics.mean(all_task):.2f} \\\\")
    
    print("""\\hline
\\end{tabular}
\\end{table}""")


def compare_injection(base_dir: str, sample_size: int, verbose: bool):
    """通常版と_reject版の妥当性を比較"""
    
    # ペアを探す
    pairs = []
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    normal_dirs = [d for d in dirs if not d.endswith('_reject')]
    
    for normal in normal_dirs:
        reject = normal + '_reject'
        if reject in dirs:
            normal_path = os.path.join(base_dir, normal)
            reject_path = os.path.join(base_dir, reject)
            if (os.path.exists(os.path.join(normal_path, 'relationship_reflection_details.jsonl')) and
                os.path.exists(os.path.join(reject_path, 'relationship_reflection_details.jsonl'))):
                pairs.append((normal, normal_path, reject_path))
    
    if not pairs:
        print("比較可能なシナリオペアが見つかりません")
        return
    
    print(f"\n発見したシナリオペア: {len(pairs)}組")
    
    normal_results = []
    reject_results = []
    
    for name, normal_path, reject_path in pairs:
        print(f"\n{'='*60}")
        print(f"比較評価: {name}")
        print(f"{'='*60}")
        
        print("\n【通常版（関係値注入あり）】")
        normal_res = evaluate_scenario(normal_path, sample_size, verbose)
        if normal_res and 'statistics' in normal_res:
            normal_results.append(normal_res)
            print(f"  妥当性: {normal_res['statistics']['overall_mean']:.2f}/5")
        
        print("\n【_reject版（関係値注入なし）】")
        reject_res = evaluate_scenario(reject_path, sample_size, verbose)
        if reject_res and 'statistics' in reject_res:
            reject_results.append(reject_res)
            print(f"  妥当性: {reject_res['statistics']['overall_mean']:.2f}/5")
    
    # サマリー
    print("\n" + "=" * 70)
    print("比較サマリー")
    print("=" * 70)
    
    if normal_results and reject_results:
        normal_avg = statistics.mean([r['statistics']['overall_mean'] for r in normal_results])
        reject_avg = statistics.mean([r['statistics']['overall_mean'] for r in reject_results])
        
        print(f"\n通常版（注入あり）: {normal_avg:.2f}/5")
        print(f"_reject版（注入なし）: {reject_avg:.2f}/5")
        print(f"差分: {normal_avg - reject_avg:+.2f}")
        
        # 予測精度の比較
        def calc_pred_avg(results_list):
            """予測精度の平均を計算"""
            pred = {'power': [], 'intimacy': [], 'task': []}
            for r in results_list:
                if 'prediction_accuracy' in r:
                    for dim in ['power', 'intimacy', 'task']:
                        if r['prediction_accuracy'][dim].get('exact_rate') is not None:
                            pred[dim].append(r['prediction_accuracy'][dim]['exact_rate'])
            return {dim: statistics.mean(vals) if vals else None for dim, vals in pred.items()}
        
        normal_pred = calc_pred_avg(normal_results)
        reject_pred = calc_pred_avg(reject_results)
        
        # 予測精度がある場合は表示
        has_pred = any(v is not None for v in normal_pred.values())
        if has_pred:
            print("\n【イベント予測との一致度】")
            print("  LLMがイベント内容から予測した変化 vs 実際の内省結果")
            for dim in ['power', 'intimacy', 'task']:
                if normal_pred[dim] is not None and reject_pred[dim] is not None:
                    diff = normal_pred[dim] - reject_pred[dim]
                    print(f"  {dim}: 注入あり {normal_pred[dim]*100:.1f}% vs 注入なし {reject_pred[dim]*100:.1f}% (差: {diff*100:+.1f}%)")
        
        if normal_avg > reject_avg:
            print("\n→ 関係値注入により、内省の妥当性が向上")
        else:
            print("\n→ 関係値注入なしでも、内省の妥当性は同等以上")


def main():
    parser = argparse.ArgumentParser(description='内省妥当性評価（LLM as a Judge）')
    parser.add_argument('--data-dir', type=str, default='out',
                       help='データディレクトリ')
    parser.add_argument('--all-scenarios', action='store_true',
                       help='すべてのシナリオを評価')
    parser.add_argument('--compare-injection', action='store_true',
                       help='通常版と_reject版を比較')
    parser.add_argument('--sample-size', type=int, default=10,
                       help='シナリオあたりのサンプル数')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='詳細出力')
    parser.add_argument('--output', type=str,
                       help='結果をJSONで保存')
    
    args = parser.parse_args()
    
    # API設定
    if HAS_GEMINI:
        set_gemini_key()
    else:
        print("Warning: Gemini APIが利用できません")
        return
    
    # 比較モード
    if args.compare_injection:
        compare_injection(args.data_dir, args.sample_size, args.verbose)
        return
    
    # シナリオ探索
    if args.all_scenarios:
        scenarios = find_all_scenarios(args.data_dir)
        print(f"発見したシナリオ: {len(scenarios)}件")
    else:
        if os.path.exists(os.path.join(args.data_dir, 'relationship_reflection_details.jsonl')):
            scenarios = [args.data_dir]
        else:
            scenarios = find_all_scenarios(args.data_dir)
    
    if not scenarios:
        print("評価対象のシナリオが見つかりません")
        return
    
    # 評価実行
    results = []
    for scenario_dir in scenarios:
        print(f"\n評価中: {os.path.basename(scenario_dir)}")
        result = evaluate_scenario(scenario_dir, args.sample_size, args.verbose)
        if result:
            results.append(result)
    
    # レポート出力
    if results:
        print_report(results)
        
        # JSON保存
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n結果を保存: {args.output}")


if __name__ == '__main__':
    main()
