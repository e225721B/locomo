#!/usr/bin/env python3
"""
Few-shot分類評価スクリプト

正解データ（true_data_other_zero）を使用して、1つの次元のfew-shot分類精度を評価します。
データをトレーニング（few-shot例）とテスト（評価対象）に分割し、
LLMによる分類結果と正解値の一致率を計算します。

使用例:
    # 親密度の評価
    python3 evaluation_style/evaluate_fewshot_classification.py \
        --dimension Intimacy \
        --speaker asuka \
        --output-dir evaluation_style/fewshot_results

    # 全次元・全スピーカーの評価
    python3 evaluation_style/evaluate_fewshot_classification.py \
        --all-dimensions \
        --all-speakers \
        --output-dir evaluation_style/fewshot_results

    # テスト比率の変更
    python3 evaluation_style/evaluate_fewshot_classification.py \
        --dimension Power \
        --speaker shinji \
        --test-ratio 0.3
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics

# matplotlibのインポート
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # GUIなしで画像保存
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from global_methods import run_gemini, set_gemini_key
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Gemini モデル
_gemini_model = None

def get_gemini_model():
    """Geminiモデルを取得（シングルトン）"""
    global _gemini_model
    if _gemini_model is None:
        set_gemini_key()
        model_name = os.environ.get('GEMINI_MODEL_NAME', 'gemini-2.0-flash')
        _gemini_model = genai.GenerativeModel(model_name)
    return _gemini_model


def load_fewshot_data(data_dir: str, speaker: str, dimension: str) -> List[Tuple[int, str]]:
    """正解データCSVを読み込む
    
    Args:
        data_dir: true_data_other_zeroディレクトリ
        speaker: スピーカー名 (asuka, shinji)
        dimension: 次元名 (Intimacy, Power, TaskOriented)
    
    Returns:
        [(value, utterance), ...]
    """
    dim_lower = dimension.lower()
    csv_path = os.path.join(data_dir, speaker, f"{speaker}_{dim_lower}_fewshot.csv")
    
    if not os.path.exists(csv_path):
        print(f"ファイルが見つかりません: {csv_path}")
        return []
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                value = int(row[dimension])
                utterance = row['utterance']
                data.append((value, utterance))
            except (KeyError, ValueError) as e:
                continue
    
    return data


def split_train_test(data: List[Tuple[int, str]], test_ratio: float = 0.3, 
                     seed: int = 42) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """データをトレーニングとテストに分割
    
    各値（-3〜+3）ごとに均等に分割する
    """
    random.seed(seed)
    
    # 値ごとにグループ化
    by_value = defaultdict(list)
    for value, utterance in data:
        by_value[value].append((value, utterance))
    
    train_data = []
    test_data = []
    
    for value in sorted(by_value.keys()):
        items = by_value[value]
        random.shuffle(items)
        
        n_test = max(1, int(len(items) * test_ratio))
        test_data.extend(items[:n_test])
        train_data.extend(items[n_test:])
    
    return train_data, test_data


def build_fewshot_prompt(dimension: str, train_data: List[Tuple[int, str]], 
                         test_utterance: str) -> str:
    """Few-shot分類用のプロンプトを構築"""
    
    dim_descriptions = {
        'Intimacy': '親密度（話者が相手に対して感じている親しさの度合い）',
        'Power': '力関係（話者が相手に対して持つ主導権の度合い）',
        'TaskOriented': 'タスク志向度（会話の目的達成志向の度合い）'
    }
    
    dim_desc = dim_descriptions.get(dimension, dimension)
    
    # Few-shot例を構築（値ごとに1つずつ選択）
    examples_by_value = defaultdict(list)
    for value, utterance in train_data:
        examples_by_value[value].append(utterance)
    
    examples_text = ""
    for value in sorted(examples_by_value.keys()):
        utterances = examples_by_value[value]
        # 各値から1つ選択
        example_utterance = random.choice(utterances)
        examples_text += f"\n値={value:+d}: 「{example_utterance}」"
    
    prompt = f"""以下の発話から、話者の「{dim_desc}」を -3 から +3 の整数で分類してください。

【{dimension}の尺度】
-3: 非常に低い
-2: 低い
-1: やや低い
 0: 普通・中立
+1: やや高い
+2: 高い
+3: 非常に高い

【分類例】{examples_text}

【分類対象の発話】
「{test_utterance}」

【出力形式】
数値のみを出力してください（-3 から +3 の整数）。
"""
    
    return prompt


def classify_utterance(dimension: str, train_data: List[Tuple[int, str]], 
                       utterance: str, verbose: bool = False) -> Optional[int]:
    """発話を分類"""
    if not HAS_GEMINI:
        return None
    
    try:
        model = get_gemini_model()
        prompt = build_fewshot_prompt(dimension, train_data, utterance)
        
        response = run_gemini(model, prompt, max_tokens=10, temperature=0.0)
        
        # 数値を抽出
        match = re.search(r'(-?\d+)', response.strip())
        if match:
            value = int(match.group(1))
            # -3〜+3にクリップ
            return max(-3, min(3, value))
        
    except Exception as e:
        if verbose:
            print(f"  分類エラー: {e}")
    
    return None


def evaluate_fewshot(data_dir: str, speaker: str, dimension: str, 
                     test_ratio: float = 0.3, seed: int = 42,
                     verbose: bool = False) -> Dict:
    """Few-shot分類を評価
    
    Returns:
        評価結果の辞書
    """
    # データ読み込み
    data = load_fewshot_data(data_dir, speaker, dimension)
    if not data:
        return {}
    
    # 分割
    train_data, test_data = split_train_test(data, test_ratio, seed)
    
    print(f"\n{'='*60}")
    print(f"評価: {speaker} - {dimension}")
    print(f"{'='*60}")
    print(f"全データ数: {len(data)}")
    print(f"トレーニング: {len(train_data)}件（few-shot例として使用）")
    print(f"テスト: {len(test_data)}件（評価対象）")
    
    # 分類実行
    results = []
    for i, (true_value, utterance) in enumerate(test_data):
        if verbose:
            print(f"\n[{i+1}/{len(test_data)}] 分類中...")
            print(f"  発話: 「{utterance[:40]}...」" if len(utterance) > 40 else f"  発話: 「{utterance}」")
        
        predicted = classify_utterance(dimension, train_data, utterance, verbose)
        
        if predicted is not None:
            results.append({
                'true': true_value,
                'predicted': predicted,
                'utterance': utterance
            })
            
            if verbose:
                match = "✓" if true_value == predicted else "✗"
                print(f"  正解: {true_value:+d}, 予測: {predicted:+d} {match}")
    
    if not results:
        print("  分類結果なし")
        return {}
    
    # 評価指標計算
    true_values = [r['true'] for r in results]
    predicted_values = [r['predicted'] for r in results]
    
    # 完全一致
    exact_matches = sum(1 for t, p in zip(true_values, predicted_values) if t == p)
    exact_rate = exact_matches / len(results)
    
    # ±1以内一致
    within_one = sum(1 for t, p in zip(true_values, predicted_values) if abs(t - p) <= 1)
    within_one_rate = within_one / len(results)
    
    # MAE
    mae = sum(abs(t - p) for t, p in zip(true_values, predicted_values)) / len(results)
    
    # 相関係数
    correlation = None
    if len(set(true_values)) > 1 and len(set(predicted_values)) > 1:
        try:
            mean_t = statistics.mean(true_values)
            mean_p = statistics.mean(predicted_values)
            std_t = statistics.stdev(true_values)
            std_p = statistics.stdev(predicted_values)
            if std_t > 0 and std_p > 0:
                n = len(true_values)
                covariance = sum((t - mean_t) * (p - mean_p) for t, p in zip(true_values, predicted_values)) / n
                correlation = covariance / (std_t * std_p)
        except:
            pass
    
    # 値ごとの精度
    accuracy_by_value = {}
    for value in range(-3, 4):
        matches = [(t, p) for t, p in zip(true_values, predicted_values) if t == value]
        if matches:
            correct = sum(1 for t, p in matches if t == p)
            accuracy_by_value[value] = {
                'correct': correct,
                'total': len(matches),
                'rate': correct / len(matches)
            }
    
    # 混同行列
    confusion_matrix = [[0 for _ in range(7)] for _ in range(7)]  # -3〜+3 = 7x7
    for t, p in zip(true_values, predicted_values):
        row = t + 3  # -3 -> 0, +3 -> 6
        col = p + 3
        confusion_matrix[row][col] += 1
    
    # Over/Under prediction
    over_prediction = sum(1 for t, p in zip(true_values, predicted_values) if p > t)
    under_prediction = sum(1 for t, p in zip(true_values, predicted_values) if p < t)
    
    result = {
        'speaker': speaker,
        'dimension': dimension,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'evaluated': len(results),
        'exact_matches': exact_matches,
        'exact_rate': exact_rate,
        'within_one': within_one,
        'within_one_rate': within_one_rate,
        'mae': mae,
        'correlation': correlation,
        'accuracy_by_value': accuracy_by_value,
        'confusion_matrix': confusion_matrix,
        'over_prediction': over_prediction,
        'under_prediction': under_prediction,
        'details': results
    }
    
    # 結果表示
    print(f"\n【結果】")
    print(f"  完全一致率: {exact_rate*100:.1f}% ({exact_matches}/{len(results)})")
    print(f"  ±1以内一致率: {within_one_rate*100:.1f}% ({within_one}/{len(results)})")
    print(f"  MAE: {mae:.2f}")
    if correlation is not None:
        print(f"  相関係数: {correlation:.3f}")
    
    # 予測傾向
    print(f"\n【予測傾向】")
    print(f"  正解: {exact_matches}件 ({exact_matches/len(results)*100:.1f}%)")
    print(f"  高め予測（Over）: {over_prediction}件 ({over_prediction/len(results)*100:.1f}%)")
    print(f"  低め予測（Under）: {under_prediction}件 ({under_prediction/len(results)*100:.1f}%)")
    
    # 混同行列表示
    print(f"\n【混同行列】（行=正解, 列=予測）")
    print("        予測→")
    print("正解↓   " + "  ".join([f"{v:+d}" for v in range(-3, 4)]))
    print("      " + "-" * 28)
    for row_idx, row in enumerate(confusion_matrix):
        true_val = row_idx - 3
        row_str = "  ".join([f"{c:2d}" if c > 0 else " ·" for c in row])
        print(f"  {true_val:+d} |  {row_str}")
    
    print(f"\n【値ごとの精度】")
    for value in range(-3, 4):
        if value in accuracy_by_value:
            acc = accuracy_by_value[value]
            print(f"  {value:+d}: {acc['rate']*100:5.1f}% ({acc['correct']}/{acc['total']})")
    
    return result


def save_results(results: List[Dict], output_dir: str):
    """結果を保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON保存
    json_path = os.path.join(output_dir, 'fewshot_classification_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON保存: {json_path}")
    
    # CSV保存（サマリー）
    csv_path = os.path.join(output_dir, 'fewshot_classification_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Speaker', 'Dimension', 'Train', 'Test', 'Evaluated',
            'Exact_Match', 'Exact_Rate', 'Within1', 'Within1_Rate', 'MAE', 'Correlation'
        ])
        for r in results:
            writer.writerow([
                r['speaker'],
                r['dimension'],
                r['train_size'],
                r['test_size'],
                r['evaluated'],
                r['exact_matches'],
                f"{r['exact_rate']*100:.1f}%",
                r['within_one'],
                f"{r['within_one_rate']*100:.1f}%",
                f"{r['mae']:.2f}",
                f"{r['correlation']:.3f}" if r['correlation'] else '-'
            ])
    print(f"CSV保存: {csv_path}")
    
    # 詳細CSV保存（予測結果）
    for r in results:
        detail_path = os.path.join(output_dir, f"{r['speaker']}_{r['dimension'].lower()}_predictions.csv")
        with open(detail_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['true', 'predicted', 'match', 'utterance'])
            for d in r['details']:
                match = 1 if d['true'] == d['predicted'] else 0
                writer.writerow([d['true'], d['predicted'], match, d['utterance']])
        print(f"詳細CSV保存: {detail_path}")
    
    # 混同行列グラフと予測傾向を保存
    save_confusion_matrices(results, output_dir)
    save_prediction_tendency(results, output_dir)


def save_confusion_matrices(results: List[Dict], output_dir: str):
    """混同行列のヒートマップを保存"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlibが利用できないため、グラフをスキップします")
        return
    
    for r in results:
        cm = np.array(r['confusion_matrix'])
        speaker = r['speaker']
        dimension = r['dimension']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # ヒートマップを描画
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        # カラーバー
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
        
        # 軸ラベル
        labels = [f"{v:+d}" for v in range(-3, 4)]
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix: {speaker} - {dimension}\n'
                     f'Accuracy: {r["exact_rate"]*100:.1f}%, MAE: {r["mae"]:.2f}')
        
        # セルに値を表示
        for i in range(7):
            for j in range(7):
                value = cm[i, j]
                if value > 0:
                    text_color = 'white' if value > cm.max() / 2 else 'black'
                    ax.text(j, i, str(value), ha='center', va='center', color=text_color, fontsize=12)
        
        # 対角線（正解線）を強調
        for i in range(7):
            ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=2))
        
        plt.tight_layout()
        
        # 保存
        graph_path = os.path.join(output_dir, f"{speaker}_{dimension.lower()}_confusion_matrix.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"混同行列グラフ保存: {graph_path}")


def save_prediction_tendency(results: List[Dict], output_dir: str):
    """予測傾向のサマリーをファイルに保存"""
    tendency_path = os.path.join(output_dir, 'prediction_tendency.txt')
    
    with open(tendency_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("予測傾向レポート\n")
        f.write("="*70 + "\n\n")
        
        for r in results:
            speaker = r['speaker']
            dimension = r['dimension']
            evaluated = r['evaluated']
            exact = r['exact_matches']
            over = r['over_prediction']
            under = r['under_prediction']
            
            f.write(f"\n{'-'*50}\n")
            f.write(f"Speaker: {speaker}, Dimension: {dimension}\n")
            f.write(f"{'-'*50}\n\n")
            
            f.write(f"評価サンプル数: {evaluated}\n\n")
            
            f.write("【予測傾向】\n")
            f.write(f"  正解 (Correct):      {exact:3d}件 ({exact/evaluated*100:5.1f}%)\n")
            f.write(f"  高め予測 (Over):     {over:3d}件 ({over/evaluated*100:5.1f}%)\n")
            f.write(f"  低め予測 (Under):    {under:3d}件 ({under/evaluated*100:5.1f}%)\n\n")
            
            # 予測傾向の解釈
            if over > under:
                bias = "高めに予測する傾向（ポジティブバイアス）"
            elif under > over:
                bias = "低めに予測する傾向（ネガティブバイアス）"
            else:
                bias = "偶りなし（バランス）"
            f.write(f"【解釈】{bias}\n")
            
            # 混同行列（テキスト形式）
            f.write(f"\n【混同行列】（行=正解, 列=予測）\n")
            f.write("        予測→\n")
            f.write("正解↓   " + "  ".join([f"{v:+d}" for v in range(-3, 4)]) + "\n")
            f.write("      " + "-"*28 + "\n")
            
            cm = r['confusion_matrix']
            for row_idx, row in enumerate(cm):
                true_val = row_idx - 3
                row_str = "  ".join([f"{c:2d}" if c > 0 else " ·" for c in row])
                f.write(f"  {true_val:+d} |  {row_str}\n")
            
            f.write("\n")
    
    print(f"予測傾向レポート保存: {tendency_path}")
    
    # 予測傾向のグラフも作成
    if HAS_MATPLOTLIB and len(results) > 0:
        save_tendency_graph(results, output_dir)


def save_tendency_graph(results: List[Dict], output_dir: str):
    """予測傾向のグラフを保存"""
    if not HAS_MATPLOTLIB:
        return
    
    # 各結果ごとにグラフを作成
    for r in results:
        speaker = r['speaker']
        dimension = r['dimension']
        
        # 棒グラフデータ
        categories = ['Correct', 'Over\n(Positive bias)', 'Under\n(Negative bias)']
        values = [r['exact_matches'], r['over_prediction'], r['under_prediction']]
        colors = ['#2ecc71', '#e74c3c', '#3498db']  # 緑、赤、青
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)
        
        # 値をバーの上に表示
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}\n({val/r["evaluated"]*100:.1f}%)',
                   ha='center', va='bottom', fontsize=11)
        
        ax.set_ylabel('Count')
        ax.set_title(f'Prediction Tendency: {speaker} - {dimension}\n'
                    f'Total: {r["evaluated"]} samples, Accuracy: {r["exact_rate"]*100:.1f}%')
        ax.set_ylim(0, max(values) * 1.3)
        
        plt.tight_layout()
        
        graph_path = os.path.join(output_dir, f"{speaker}_{dimension.lower()}_tendency.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"予測傾向グラフ保存: {graph_path}")
    
    # 全体比較グラフ（複数結果がある場合）
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results))
        width = 0.25
        
        correct_vals = [r['exact_matches'] for r in results]
        over_vals = [r['over_prediction'] for r in results]
        under_vals = [r['under_prediction'] for r in results]
        
        bars1 = ax.bar(x - width, correct_vals, width, label='Correct', color='#2ecc71')
        bars2 = ax.bar(x, over_vals, width, label='Over (Positive)', color='#e74c3c')
        bars3 = ax.bar(x + width, under_vals, width, label='Under (Negative)', color='#3498db')
        
        ax.set_ylabel('Count')
        ax.set_title('Prediction Tendency Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['speaker']}\n{r['dimension']}" for r in results])
        ax.legend()
        
        plt.tight_layout()
        
        graph_path = os.path.join(output_dir, 'tendency_comparison.png')
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"傾向比較グラフ保存: {graph_path}")


def print_summary(results: List[Dict]):
    """全体サマリーを表示"""
    print("\n" + "=" * 70)
    print("Few-shot分類評価サマリー")
    print("=" * 70)
    
    # 次元別集計
    by_dimension = defaultdict(list)
    for r in results:
        by_dimension[r['dimension']].append(r)
    
    print(f"\n{'次元':<15} {'完全一致':<20} {'±1以内':<20} {'MAE':<10}")
    print("-" * 65)
    
    for dim in ['Intimacy', 'Power', 'TaskOriented']:
        if dim in by_dimension:
            dim_results = by_dimension[dim]
            total_exact = sum(r['exact_matches'] for r in dim_results)
            total_evaluated = sum(r['evaluated'] for r in dim_results)
            total_within = sum(r['within_one'] for r in dim_results)
            avg_mae = statistics.mean(r['mae'] for r in dim_results)
            
            print(f"{dim:<15} {total_exact/total_evaluated*100:>5.1f}% ({total_exact:>3}/{total_evaluated:<3})   "
                  f"{total_within/total_evaluated*100:>5.1f}% ({total_within:>3}/{total_evaluated:<3})   {avg_mae:.2f}")
    
    # 全体
    if results:
        total_exact = sum(r['exact_matches'] for r in results)
        total_evaluated = sum(r['evaluated'] for r in results)
        total_within = sum(r['within_one'] for r in results)
        avg_mae = statistics.mean(r['mae'] for r in results)
        
        print("-" * 65)
        print(f"{'全体':<15} {total_exact/total_evaluated*100:>5.1f}% ({total_exact:>3}/{total_evaluated:<3})   "
              f"{total_within/total_evaluated*100:>5.1f}% ({total_within:>3}/{total_evaluated:<3})   {avg_mae:.2f}")
    
    # LaTeXテーブル
    print("\n" + "-" * 70)
    print("【LaTeX テーブル】")
    print("-" * 70)
    print("""
\\begin{table}[h]
\\centering
\\caption{Few-shot Classification Accuracy}
\\begin{tabular}{llccc}
\\hline
Speaker & Dimension & Exact Match & Within ±1 & MAE \\\\
\\hline""")
    
    for r in results:
        print(f"{r['speaker']} & {r['dimension']} & {r['exact_rate']*100:.1f}\\% ({r['exact_matches']}/{r['evaluated']}) & "
              f"{r['within_one_rate']*100:.1f}\\% & {r['mae']:.2f} \\\\")
    
    print("""\\hline
\\end{tabular}
\\end{table}""")


def main():
    parser = argparse.ArgumentParser(description='Few-shot分類評価')
    parser.add_argument('--data-dir', type=str, 
                       default='evaluation_style/true_data_other_zero',
                       help='正解データディレクトリ')
    parser.add_argument('--dimension', type=str,
                       choices=['Intimacy', 'Power', 'TaskOriented'],
                       help='対象次元')
    parser.add_argument('--all-dimensions', action='store_true',
                       help='全次元を評価')
    parser.add_argument('--speaker', type=str,
                       choices=['asuka', 'shinji'],
                       help='スピーカー')
    parser.add_argument('--all-speakers', action='store_true',
                       help='全スピーカーを評価')
    parser.add_argument('--test-ratio', type=float, default=0.3,
                       help='テストデータの比率（デフォルト: 0.3）')
    parser.add_argument('--seed', type=int, default=42,
                       help='乱数シード')
    parser.add_argument('--output-dir', type=str,
                       default='evaluation_style/fewshot_results',
                       help='結果出力先')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='詳細出力')
    
    args = parser.parse_args()
    
    # API設定
    if HAS_GEMINI:
        set_gemini_key()
    else:
        print("Error: Gemini APIが利用できません")
        return
    
    # 対象の決定
    if args.all_dimensions:
        dimensions = ['Intimacy', 'Power', 'TaskOriented']
    elif args.dimension:
        dimensions = [args.dimension]
    else:
        print("Error: --dimension または --all-dimensions を指定してください")
        return
    
    if args.all_speakers:
        speakers = ['asuka', 'shinji']
    elif args.speaker:
        speakers = [args.speaker]
    else:
        print("Error: --speaker または --all-speakers を指定してください")
        return
    
    print("=" * 70)
    print("Few-shot分類評価")
    print("=" * 70)
    print(f"データディレクトリ: {args.data_dir}")
    print(f"対象次元: {dimensions}")
    print(f"対象スピーカー: {speakers}")
    print(f"テスト比率: {args.test_ratio}")
    print(f"出力先: {args.output_dir}")
    
    # 乱数シード
    random.seed(args.seed)
    
    # 評価実行
    all_results = []
    for speaker in speakers:
        for dimension in dimensions:
            result = evaluate_fewshot(
                data_dir=args.data_dir,
                speaker=speaker,
                dimension=dimension,
                test_ratio=args.test_ratio,
                seed=args.seed,
                verbose=args.verbose
            )
            if result:
                all_results.append(result)
    
    # サマリー表示
    if all_results:
        print_summary(all_results)
        
        # 結果保存
        save_results(all_results, args.output_dir)


if __name__ == '__main__':
    main()
