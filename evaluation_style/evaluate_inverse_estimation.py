#!/usr/bin/env python3


"""
関係値逆推定評価スクリプト

# 比較評価（通常版 vs _reject版）
python3 evaluation_style/evaluate_inverse_estimation.py --data-dir out --compare-injection --sample-size 60

# 単一シナリオの評価
python3 evaluation_style/evaluate_inverse_estimation.py --data-dir out/eva_intimacy_increase --use-generated-values


【概要】
提案システムが生成した発話から、第三者（推定器）が「元となった関係値」を
どれだけ正確に逆推定できるかを検証する実験です。
これにより、システム内部のパラメータ（原因）が、外部から観測可能な
言語スタイル（結果）として適切に出力されているかを定量的に評価します。

【評価の仕組み（逆算のプロセス）】
1. 入力（意図）：システムに対し、特定の関係値を指定して発話を生成させる
2. 出力（結果）：システムが発話テキストを出力する
3. 推定（逆算）：生成元とは異なるLLM（評価器）が発話から関係値を推定
4. 判定（答え合わせ）：その発話に指定した値、またはgenerated_valuesと推定値の一致度を計算

使用方法:
    source scripts/env.sh
    python3 evaluation_style/evaluate_inverse_estimation.py --data-dir out/eva_intimacy_increase
    python3 evaluation_style/evaluate_inverse_estimation.py --data-dir out --all-scenarios

評価指標:
    - 完全一致率 (Exact Match Rate)
    - 許容誤差内一致率 (±1以内)
    - 平均絶対誤差 (MAE)
    - ピアソン相関係数
    - 次元別精度
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


# 推定プロンプトテンプレート
ESTIMATION_PROMPT = """以下の発話から、話者の相手に対する関係性を推定してください。

【発話】
「{utterance}」

【評価する3つの次元】（それぞれ -3 から +3 の整数）

1. Intimacy（親密度）:
   - Intimacy（親密度）-3〜+3: -3〜-1は敬語や距離感のある言い方（ただし必要な情報は簡潔に伝える）、0は普通の会話トーン、+1〜+3は砕けた言い方・冗談・感情の共有が増える

2. Power（力関係）:
   - Power（力関係）-3〜+3: -3〜-1は相手を立てる言い方（ただしタスク遂行時は実務的に対応可能）、0は対等な立場での会話、+1〜+3は主導権を握る言い方・指示・提案が増える

3. TaskOriented（タスク志向）:
   - TaskOriented（タスク志向）-3〜+3: -3〜-1は雑談・感情共有・関係構築が中心、0はバランス型、+1〜+3は目的達成優先で感情表現より情報伝達・行動提案を重視

【出力形式】
JSON形式で出力してください:
{{"Intimacy": 数値, "Power": 数値, "TaskOriented": 数値}}

JSONのみを出力し、説明は不要です。
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


def load_prompt_utterance_pairs(data_dir: str) -> List[Dict]:
    """prompt_utterance_pairs.jsonlを読み込む"""
    filepath = os.path.join(data_dir, 'prompt_utterance_pairs.jsonl')
    if not os.path.exists(filepath):
        return []
    
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                entry = json.loads(line)
                entry['_line_index'] = idx
                pairs.append(entry)
    return pairs


def build_utterance_to_generated_values_map(data_dir: str) -> Dict[str, Dict[str, int]]:
    """発話と内省値(generated_values)の対応マップを構築
    
    relationship_reflection_details.jsonlから読み込み、
    発話テキストをキーとしてgenerated_valuesを返すマップを作成
    """
    reflection_data = load_reflection_data(data_dir)
    pairs = load_prompt_utterance_pairs(data_dir)
    
    # 発話→generated_valuesのマップ
    # (session, turn, speaker) をキーにして対応付け
    # reflection_detailsは各ターンの内省結果を持つ
    
    # まず reflection_details から (session, turn) -> generated_values のマップを作成
    reflection_map = {}
    for r in reflection_data:
        session = r.get('session_idx', r.get('session'))
        # direction から話者を判定
        direction = r.get('direction')
        src = r.get('src', '')
        generated = r.get('generated_values', {})
        
        if session and direction and generated:
            key = (session, direction)
            # 同じ(session, direction)で複数ある場合は最新の行番号のものを使用
            if key not in reflection_map or r.get('_line_index', 0) > reflection_map[key].get('_line_index', 0):
                reflection_map[key] = {
                    'generated_values': generated,
                    'src': src,
                    '_line_index': r.get('_line_index', 0)
                }
    
    # pairs から発話→generated_valuesのマップを構築
    utterance_map = {}
    for p in pairs:
        session = p.get('session_id')
        speaker = p.get('speaker', '')
        utterance = p.get('utterance', '')
        
        # speakerからdirectionを推定（Aならa_to_b、Bならb_to_a）
        # ここでは簡易的に、アスカ系の名前ならa_to_b、シンジ系ならb_to_aと判定
        if 'アスカ' in speaker or 'Asuka' in speaker:
            direction = 'a_to_b'
        elif 'シンジ' in speaker or 'Shinji' in speaker:
            direction = 'b_to_a'
        else:
            # 他のケース: src名で判定
            direction = None
            for key, val in reflection_map.items():
                if key[0] == session and val.get('src') == speaker:
                    direction = key[1]
                    break
        
        if session and direction:
            key = (session, direction)
            if key in reflection_map:
                utterance_map[utterance] = reflection_map[key]['generated_values']
    
    return utterance_map


def extract_relationship_from_prompt(prompt: str) -> Optional[Dict[str, int]]:
    """プロンプトから関係値を抽出する"""
    # パターン: Intimacy=-3, Power=0, TaskOriented=0
    pattern = r'Intimacy=(-?\d+),?\s*Power=(-?\d+),?\s*TaskOriented=(-?\d+)'
    match = re.search(pattern, prompt)
    if match:
        return {
            'Intimacy': int(match.group(1)),
            'Power': int(match.group(2)),
            'TaskOriented': int(match.group(3))
        }
    return None


def estimate_relationship(utterance: str) -> Optional[Dict[str, int]]:
    """LLMを使って発話から関係値を推定"""
    if not HAS_GEMINI:
        print("Error: Gemini APIが利用できません")
        return None
    
    try:
        model = get_gemini_model()
        
        prompt = ESTIMATION_PROMPT.format(utterance=utterance)
        
        response = run_gemini(model, prompt, max_tokens=100, temperature=0.0)
        
        # JSON抽出（複数のパターンに対応）
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            json_str = json_match.group()
            # 数値がない場合（例: {"Intimacy": , "Power": 0}）の対処
            json_str = re.sub(r':\s*,', ': 0,', json_str)
            json_str = re.sub(r':\s*\}', ': 0}', json_str)
            
            result = json.loads(json_str)
            # 値を-3〜+3の範囲にクリップ
            for key in ['Intimacy', 'Power', 'TaskOriented']:
                if key in result:
                    try:
                        result[key] = max(-3, min(3, int(result[key])))
                    except (ValueError, TypeError):
                        result[key] = 0
            return result
    except json.JSONDecodeError as e:
        # デバッグ用に応答を表示
        pass
    except Exception as e:
        print(f"  推定エラー: {e}")
    
    return None


def calculate_metrics(true_values: List[int], estimated_values: List[int]) -> Dict:
    """評価指標を計算"""
    if not true_values or not estimated_values or len(true_values) != len(estimated_values):
        return {}
    
    n = len(true_values)
    
    # 完全一致率
    exact_matches = sum(1 for t, e in zip(true_values, estimated_values) if t == e)
    exact_match_rate = exact_matches / n
    
    # ±1以内一致率
    within_one_matches = sum(1 for t, e in zip(true_values, estimated_values) if abs(t - e) <= 1)
    within_one_rate = within_one_matches / n
    
    # 平均絶対誤差
    mae = sum(abs(t - e) for t, e in zip(true_values, estimated_values)) / n
    
    # 相関係数（標準偏差が0でない場合のみ）
    correlation = None
    if len(set(true_values)) > 1 and len(set(estimated_values)) > 1:
        try:
            mean_t = statistics.mean(true_values)
            mean_e = statistics.mean(estimated_values)
            std_t = statistics.stdev(true_values)
            std_e = statistics.stdev(estimated_values)
            if std_t > 0 and std_e > 0:
                covariance = sum((t - mean_t) * (e - mean_e) for t, e in zip(true_values, estimated_values)) / n
                correlation = covariance / (std_t * std_e)
        except:
            pass
    
    return {
        'n': n,
        'exact_matches': exact_matches,
        'exact_match_rate': exact_match_rate,
        'within_one_matches': within_one_matches,
        'within_one_rate': within_one_rate,
        'mae': mae,
        'correlation': correlation
    }


def evaluate_scenario(data_dir: str, sample_size: int = 20,
                      use_generated_values: bool = False, verbose: bool = False) -> Dict:
    """一つのシナリオを評価
    
    Args:
        data_dir: データディレクトリ
        sample_size: サンプル数
        use_generated_values: Trueの場合、プロンプトからではなくgenerated_values（内省値）を正解として使用
        verbose: 詳細出力
    """
    
    # データ読み込み
    pairs = load_prompt_utterance_pairs(data_dir)
    
    if not pairs:
        print(f"  データなし: {data_dir}")
        return {}
    
    # generated_valuesを使用する場合、マップを構築
    generated_values_map = {}
    if use_generated_values:
        generated_values_map = build_utterance_to_generated_values_map(data_dir)
        if verbose:
            print(f"  内省値マップ: {len(generated_values_map)}件")
    
    # サンプリング（均等に選択）
    if len(pairs) > sample_size:
        step = len(pairs) // sample_size
        pairs = pairs[::step][:sample_size]
    
    results = {
        'scenario': os.path.basename(data_dir),
        'total_samples': len(pairs),
        'dimensions': {}
    }
    
    true_by_dim = defaultdict(list)
    estimated_by_dim = defaultdict(list)
    
    for i, pair in enumerate(pairs):
        utterance = pair.get('utterance', '')
        prompt = pair.get('prompt', '')
        speaker = pair.get('speaker', '')
        
        # 正解値の取得
        if use_generated_values:
            # generated_values（内省値）を正解として使用
            true_values = generated_values_map.get(utterance)
            if not true_values:
                continue
        else:
            # プロンプトから抽出した指定値を正解として使用
            true_values = extract_relationship_from_prompt(prompt)
            if not true_values:
                continue
        
        # 推定
        if verbose:
            print(f"  [{i+1}/{len(pairs)}] 推定中...")
        
        estimated = estimate_relationship(utterance)
        if not estimated:
            continue
        
        # 記録
        for dim in ['Intimacy', 'Power', 'TaskOriented']:
            if dim in true_values and dim in estimated:
                true_by_dim[dim].append(true_values[dim])
                estimated_by_dim[dim].append(estimated[dim])
        
        if verbose:
            print(f"    正解（{'内省値' if use_generated_values else '指定値'}）: {true_values}")
            print(f"    推定: {estimated}")
    
    # 次元別評価
    for dim in ['Intimacy', 'Power', 'TaskOriented']:
        if true_by_dim[dim]:
            metrics = calculate_metrics(true_by_dim[dim], estimated_by_dim[dim])
            results['dimensions'][dim] = metrics
    
    # 全体評価
    all_true = []
    all_estimated = []
    for dim in ['Intimacy', 'Power', 'TaskOriented']:
        all_true.extend(true_by_dim[dim])
        all_estimated.extend(estimated_by_dim[dim])
    
    if all_true:
        results['overall'] = calculate_metrics(all_true, all_estimated)
    
    return results


def find_all_scenarios(base_dir: str) -> List[str]:
    """すべてのシナリオディレクトリを探す"""
    scenarios = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, 'prompt_utterance_pairs.jsonl')):
                scenarios.append(path)
    return sorted(scenarios)


def find_injection_pairs(base_dir: str) -> List[Tuple[str, str, str]]:
    """通常版と_reject版のペアを探す
    
    Returns:
        [(scenario_name, normal_dir, reject_dir), ...]
    """
    pairs = []
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # _rejectで終わらないディレクトリを探す
    normal_dirs = [d for d in dirs if not d.endswith('_reject')]
    
    for normal in normal_dirs:
        reject = normal + '_reject'
        if reject in dirs:
            normal_path = os.path.join(base_dir, normal)
            reject_path = os.path.join(base_dir, reject)
            # 両方にデータがあるか確認
            if (os.path.exists(os.path.join(normal_path, 'prompt_utterance_pairs.jsonl')) and
                os.path.exists(os.path.join(reject_path, 'prompt_utterance_pairs.jsonl'))):
                pairs.append((normal, normal_path, reject_path))
    
    return sorted(pairs)


def compare_injection_effect(base_dir: str, sample_size: int, verbose: bool, output_csv: str = None):
    """通常版と_reject版の逆推定精度を比較
    
    両方でgenerated_values（内省値）を正解として使用し、
    「内省した関係値が発話に反映されているか」を評価
    """
    
    pairs = find_injection_pairs(base_dir)
    
    if not pairs:
        print("比較可能なシナリオペアが見つかりません")
        return
    
    print(f"\n発見したシナリオペア: {len(pairs)}組")
    for name, _, _ in pairs:
        print(f"  - {name}")
    
    print("\n【評価方法】")
    print("  正解値: generated_values（内省値）")
    print("  推定値: 発話からLLMで逆推定")
    print("  目的: 内省した関係値が発話に反映されているかを評価")
    
    # 結果格納
    normal_results = []
    reject_results = []
    
    for scenario_name, normal_path, reject_path in pairs:
        print(f"\n{'='*60}")
        print(f"比較評価: {scenario_name}")
        print(f"{'='*60}")
        
        # 通常版の評価（generated_valuesを使用）
        print(f"\n【通常版（関係値注入あり）】")
        normal_res = evaluate_scenario(normal_path, sample_size, 
                                       use_generated_values=True, verbose=verbose)
        if normal_res:
            normal_results.append(normal_res)
            if 'overall' in normal_res:
                o = normal_res['overall']
                print(f"  完全一致率: {o['exact_match_rate']*100:.1f}% ({o['exact_matches']}/{o['n']})")
                print(f"  ±1以内一致率: {o['within_one_rate']*100:.1f}% ({o['within_one_matches']}/{o['n']})")
                print(f"  MAE: {o['mae']:.2f}")
        
        # _reject版の評価（generated_valuesを使用）
        print(f"\n【_reject版（関係値注入なし）】")
        reject_res = evaluate_scenario(reject_path, sample_size, 
                                       use_generated_values=True, verbose=verbose)
        if reject_res:
            reject_results.append(reject_res)
            if 'overall' in reject_res:
                o = reject_res['overall']
                print(f"  完全一致率: {o['exact_match_rate']*100:.1f}% ({o['exact_matches']}/{o['n']})")
                print(f"  ±1以内一致率: {o['within_one_rate']*100:.1f}% ({o['within_one_matches']}/{o['n']})")
                print(f"  MAE: {o['mae']:.2f}")
    
    # 全体比較サマリー
    print("\n" + "=" * 70)
    print("関係値注入効果の比較サマリー")
    print("=" * 70)
    
    def aggregate_metrics(results_list):
        """メトリクスを集約（分子・分母も含む）"""
        totals = {
            'Intimacy': {'exact': 0, 'within_one': 0, 'n': 0, 'mae_sum': 0},
            'Power': {'exact': 0, 'within_one': 0, 'n': 0, 'mae_sum': 0},
            'TaskOriented': {'exact': 0, 'within_one': 0, 'n': 0, 'mae_sum': 0},
            'overall': {'exact': 0, 'within_one': 0, 'n': 0, 'mae_sum': 0}
        }
        for res in results_list:
            # 次元別
            if 'dimensions' in res:
                for dim in ['Intimacy', 'Power', 'TaskOriented']:
                    if dim in res['dimensions']:
                        d = res['dimensions'][dim]
                        totals[dim]['exact'] += d.get('exact_matches', 0)
                        totals[dim]['within_one'] += d.get('within_one_matches', 0)
                        totals[dim]['n'] += d.get('n', 0)
                        totals[dim]['mae_sum'] += d.get('mae', 0) * d.get('n', 0)
            # 全体
            if 'overall' in res:
                o = res['overall']
                totals['overall']['exact'] += o.get('exact_matches', 0)
                totals['overall']['within_one'] += o.get('within_one_matches', 0)
                totals['overall']['n'] += o.get('n', 0)
                totals['overall']['mae_sum'] += o.get('mae', 0) * o.get('n', 0)
        
        # レート計算
        result = {}
        for key in totals:
            if totals[key]['n'] > 0:
                result[key] = {
                    'exact_matches': totals[key]['exact'],
                    'within_one_matches': totals[key]['within_one'],
                    'n': totals[key]['n'],
                    'exact_rate': totals[key]['exact'] / totals[key]['n'],
                    'within_one_rate': totals[key]['within_one'] / totals[key]['n'],
                    'mae': totals[key]['mae_sum'] / totals[key]['n']
                }
        return result
    
    normal_agg = aggregate_metrics(normal_results)
    reject_agg = aggregate_metrics(reject_results)
    
    # 次元別の詳細サマリー
    if normal_agg and reject_agg:
        print(f"\n比較シナリオ数: {len(pairs)}")
        
        # 次元別の表示
        for dim, dim_ja in [('Intimacy', '親密度'), ('Power', '力関係'), ('TaskOriented', '目的指向性')]:
            print(f"\n【{dim_ja} ({dim})】")
            print(f"  {'条件':<20} {'完全一致':<20} {'±1以内':<20} {'MAE':<10}")
            print("  " + "-" * 70)
            
            if dim in normal_agg:
                n = normal_agg[dim]
                print(f"  {'注入あり':<20} {n['exact_rate']*100:>5.1f}% ({n['exact_matches']:>3}/{n['n']:<3})    {n['within_one_rate']*100:>5.1f}% ({n['within_one_matches']:>3}/{n['n']:<3})    {n['mae']:.2f}")
            if dim in reject_agg:
                r = reject_agg[dim]
                print(f"  {'注入なし':<20} {r['exact_rate']*100:>5.1f}% ({r['exact_matches']:>3}/{r['n']:<3})    {r['within_one_rate']*100:>5.1f}% ({r['within_one_matches']:>3}/{r['n']:<3})    {r['mae']:.2f}")
            
            if dim in normal_agg and dim in reject_agg:
                diff = normal_agg[dim]['exact_rate'] - reject_agg[dim]['exact_rate']
                print(f"  {'差分':<20} {diff*100:>+5.1f}%")
        
        # 全体サマリー
        print(f"\n【全体】")
        print(f"  {'条件':<20} {'完全一致':<20} {'±1以内':<20} {'MAE':<10}")
        print("  " + "-" * 70)
        
        if 'overall' in normal_agg:
            n = normal_agg['overall']
            print(f"  {'注入あり':<20} {n['exact_rate']*100:>5.1f}% ({n['exact_matches']:>3}/{n['n']:<3})    {n['within_one_rate']*100:>5.1f}% ({n['within_one_matches']:>3}/{n['n']:<3})    {n['mae']:.2f}")
        if 'overall' in reject_agg:
            r = reject_agg['overall']
            print(f"  {'注入なし':<20} {r['exact_rate']*100:>5.1f}% ({r['exact_matches']:>3}/{r['n']:<3})    {r['within_one_rate']*100:>5.1f}% ({r['within_one_matches']:>3}/{r['n']:<3})    {r['mae']:.2f}")
        
        if 'overall' in normal_agg and 'overall' in reject_agg:
            exact_diff = normal_agg['overall']['exact_rate'] - reject_agg['overall']['exact_rate']
            mae_diff = normal_agg['overall']['mae'] - reject_agg['overall']['mae']
            print(f"  {'差分':<20} {exact_diff*100:>+5.1f}%                                      {mae_diff:>+.2f}")
        
        print("\n【解釈】")
        if 'overall' in normal_agg and 'overall' in reject_agg:
            exact_diff = normal_agg['overall']['exact_rate'] - reject_agg['overall']['exact_rate']
            mae_diff = normal_agg['overall']['mae'] - reject_agg['overall']['mae']
            
            if exact_diff > 0:
                print(f"  ✓ 通常版の完全一致率が {exact_diff*100:.1f}ポイント高い")
                print(f"    → 関係値注入により、発話が指定した関係値をより正確に反映")
            else:
                print(f"  ✗ _reject版の完全一致率が {-exact_diff*100:.1f}ポイント高い")
            
            if mae_diff < 0:
                print(f"  ✓ 通常版のMAEが {-mae_diff:.2f}低い")
                print(f"    → 推定誤差が小さく、関係値が発話に反映されている")
            else:
                print(f"  ✗ _reject版のMAEが {mae_diff:.2f}低い")
    
    # CSV出力
    if output_csv:
        import csv
        csv_rows = []
        
        # ヘッダー
        csv_rows.append([
            'Dimension', 'Condition',
            'Exact_Matches', 'Total', 'Exact_Rate',
            'Within1_Matches', 'Within1_Rate', 'MAE'
        ])
        
        # データ行
        for dim in ['Intimacy', 'Power', 'TaskOriented', 'overall']:
            dim_name = {'Intimacy': '親密度', 'Power': '力関係', 'TaskOriented': '目的指向性', 'overall': '全体'}.get(dim, dim)
            
            if dim in normal_agg:
                n = normal_agg[dim]
                csv_rows.append([
                    dim_name, '注入あり',
                    n['exact_matches'], n['n'], f"{n['exact_rate']*100:.1f}%",
                    n['within_one_matches'], f"{n['within_one_rate']*100:.1f}%", f"{n['mae']:.2f}"
                ])
            if dim in reject_agg:
                r = reject_agg[dim]
                csv_rows.append([
                    dim_name, '注入なし',
                    r['exact_matches'], r['n'], f"{r['exact_rate']*100:.1f}%",
                    r['within_one_matches'], f"{r['within_one_rate']*100:.1f}%", f"{r['mae']:.2f}"
                ])
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        
        print(f"\n📊 CSV出力: {output_csv}")
    
    # LaTeXテーブル（次元別）
    print("\n" + "-" * 70)
    print("【LaTeX テーブル】")
    print("-" * 70)
    print("""
\\begin{table}[h]
\\centering
\\caption{Inverse Estimation Accuracy by Dimension}
\\begin{tabular}{llccc}
\\hline
Dimension & Condition & Exact Match & Within ±1 & MAE \\\\
\\hline""")
    
    for dim in ['Intimacy', 'Power', 'TaskOriented']:
        if dim in normal_agg:
            n = normal_agg[dim]
            print(f"{dim} & With Injection & {n['exact_rate']*100:.1f}\\% ({n['exact_matches']}/{n['n']}) & {n['within_one_rate']*100:.1f}\\% & {n['mae']:.2f} \\\\")
        if dim in reject_agg:
            r = reject_agg[dim]
            print(f" & Without Injection & {r['exact_rate']*100:.1f}\\% ({r['exact_matches']}/{r['n']}) & {r['within_one_rate']*100:.1f}\\% & {r['mae']:.2f} \\\\")
        print("\\hline")
    
    # 全体
    if 'overall' in normal_agg:
        n = normal_agg['overall']
        print(f"Overall & With Injection & {n['exact_rate']*100:.1f}\\% ({n['exact_matches']}/{n['n']}) & {n['within_one_rate']*100:.1f}\\% & {n['mae']:.2f} \\\\")
    if 'overall' in reject_agg:
        r = reject_agg['overall']
        print(f" & Without Injection & {r['exact_rate']*100:.1f}\\% ({r['exact_matches']}/{r['n']}) & {r['within_one_rate']*100:.1f}\\% & {r['mae']:.2f} \\\\")
    
    print("""\\hline
\\end{tabular}
\\end{table}""")


def print_report(results: List[Dict]):
    """評価レポートを出力"""
    print("\n" + "=" * 70)
    print("関係値逆推定評価レポート")
    print("=" * 70)
    
    # 全体集計
    all_exact = []
    all_within_one = []
    all_mae = []
    
    for res in results:
        if 'overall' in res:
            all_exact.append(res['overall']['exact_match_rate'])
            all_within_one.append(res['overall']['within_one_rate'])
            all_mae.append(res['overall']['mae'])
    
    if all_exact:
        print("\n【全体サマリー】")
        print(f"  評価シナリオ数: {len(results)}")
        print(f"  平均完全一致率: {statistics.mean(all_exact)*100:.1f}%")
        print(f"  平均±1以内一致率: {statistics.mean(all_within_one)*100:.1f}%")
        print(f"  平均MAE: {statistics.mean(all_mae):.2f}")
    
    # シナリオ別
    print("\n" + "-" * 70)
    print("【シナリオ別詳細】")
    print("-" * 70)
    
    for res in results:
        scenario = res.get('scenario', 'Unknown')
        print(f"\n■ {scenario}")
        print(f"  サンプル数: {res.get('total_samples', 0)}")
        
        if 'overall' in res:
            overall = res['overall']
            print(f"  全体:")
            print(f"    完全一致率: {overall['exact_match_rate']*100:.1f}%")
            print(f"    ±1以内一致率: {overall['within_one_rate']*100:.1f}%")
            print(f"    MAE: {overall['mae']:.2f}")
            if overall.get('correlation') is not None:
                print(f"    相関係数: {overall['correlation']:.3f}")
        
        if 'dimensions' in res:
            for dim, metrics in res['dimensions'].items():
                print(f"  {dim}:")
                print(f"    n={metrics['n']}, 一致率={metrics['exact_match_rate']*100:.0f}%, MAE={metrics['mae']:.2f}")
    
    # LaTeXテーブル
    print("\n" + "-" * 70)
    print("【LaTeX テーブル】")
    print("-" * 70)
    print("""
\\begin{table}[h]
\\centering
\\caption{Inverse Estimation Accuracy}
\\begin{tabular}{lcccc}
\\hline
Scenario & Exact Match & Within ±1 & MAE & Correlation \\\\
\\hline""")
    
    for res in results:
        if 'overall' in res:
            o = res['overall']
            corr = f"{o['correlation']:.3f}" if o.get('correlation') else "-"
            print(f"{res['scenario']} & {o['exact_match_rate']*100:.1f}\\% & {o['within_one_rate']*100:.1f}\\% & {o['mae']:.2f} & {corr} \\\\")
    
    print("""\\hline
\\end{tabular}
\\end{table}""")


def main():
    parser = argparse.ArgumentParser(description='関係値逆推定評価')
    parser.add_argument('--data-dir', type=str, default='out',
                       help='データディレクトリ')
    parser.add_argument('--all-scenarios', action='store_true',
                       help='すべてのシナリオを評価')
    parser.add_argument('--compare-injection', action='store_true',
                       help='通常版と_reject版を比較（generated_valuesを正解として使用）')
    parser.add_argument('--use-generated-values', action='store_true',
                       help='プロンプトの指定値ではなくgenerated_values（内省値）を正解として使用')
    parser.add_argument('--sample-size', type=int, default=20,
                       help='シナリオあたりのサンプル数')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='詳細出力')
    parser.add_argument('--output', type=str,
                       help='結果をJSONで保存')
    parser.add_argument('--output-csv', type=str,
                       help='結果をCSV（スプレッドシート形式）で保存')
    
    args = parser.parse_args()
    
    # API設定
    if HAS_GEMINI:
        set_gemini_key()
    else:
        print("Warning: Gemini APIが利用できません")
        return
    
    # 比較モード
    if args.compare_injection:
        compare_injection_effect(args.data_dir, args.sample_size, args.verbose, args.output_csv)
        return
    
    # シナリオ探索
    if args.all_scenarios:
        scenarios = find_all_scenarios(args.data_dir)
        print(f"発見したシナリオ: {len(scenarios)}件")
    else:
        if os.path.exists(os.path.join(args.data_dir, 'prompt_utterance_pairs.jsonl')):
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
        result = evaluate_scenario(
            scenario_dir,
            sample_size=args.sample_size,
            use_generated_values=args.use_generated_values,
            verbose=args.verbose
        )
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
