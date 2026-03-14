#!/usr/bin/env python3
"""
Few-shot学習用の正解データセット生成スクリプト

関係値を注入した実験のプロンプトを抽出し、関係値（-3〜+3）を変化させて
LLMで発話を生成し、正解データセットとして保存します。

使用例:
    # 親密度の正解データを生成
    python3 evaluation_style/generate_fewshot_dataset.py \
        --data-dir out/eva_intimacy_increase \
        --dimension Intimacy \
        --num-prompts 10

    # 全次元の正解データを生成
    python3 evaluation_style/generate_fewshot_dataset.py \
        --data-dir out/eva_intimacy_increase \
        --all-dimensions \
        --num-prompts 10

出力ファイル:
    - asuka_intimacy_fewshot.csv
    - shinji_intimacy_fewshot.csv
    - asuka_power_fewshot.csv
    - shinji_power_fewshot.csv
    - asuka_taskoriented_fewshot.csv
    - shinji_taskoriented_fewshot.csv
    
コマンドラインオプション:
    python3 evaluation_style/generate_fewshot_dataset.py --all-dimensions --num-prompts 10
    
    # 親密度のみ変化させ、他は0固定
    python3 evaluation_style/generate_fewshot_dataset.py \
    --dimension Intimacy --num-prompts 10 --zero-other-dims

    # 全次元を生成（各次元生成時、他次元は0固定）
    python3 evaluation_style/generate_fewshot_dataset.py \
    --all-dimensions --num-prompts 10 --zero-other-dims
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
import random

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
        model_name = os.environ.get('GEMINI_MODEL_NAME', 'gemini-2.5-flash')
        _gemini_model = genai.GenerativeModel(model_name)
    return _gemini_model


def load_prompt_utterance_pairs(data_dir: str) -> List[Dict]:
    """prompt_utterance_pairs.jsonlを読み込む"""
    filepath = os.path.join(data_dir, 'prompt_utterance_pairs.jsonl')
    pairs = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
    return pairs


def extract_relationship_from_prompt(prompt: str) -> Optional[Dict[str, int]]:
    """プロンプトから関係値を抽出する"""
    pattern = r'Intimacy=(-?\d+),?\s*Power=(-?\d+),?\s*TaskOriented=(-?\d+)'
    match = re.search(pattern, prompt)
    if match:
        return {
            'Intimacy': int(match.group(1)),
            'Power': int(match.group(2)),
            'TaskOriented': int(match.group(3))
        }
    return None


def replace_relationship_value(prompt: str, dimension: str, new_value: int, 
                                zero_other_dims: bool = False) -> str:
    """プロンプト内の指定次元の関係値を置換する
    
    Args:
        prompt: 元のプロンプト
        dimension: 変更する次元 ('Intimacy', 'Power', 'TaskOriented')
        new_value: 新しい値 (-3〜+3)
        zero_other_dims: Trueの場合、他の次元を0に固定する
    """
    
    # 現在の値を抽出
    current = extract_relationship_from_prompt(prompt)
    if not current:
        return prompt
    
    # 新しい値で置換（zero_other_dimsがTrueなら他の次元は0）
    if zero_other_dims:
        new_intimacy = new_value if dimension == 'Intimacy' else 0
        new_power = new_value if dimension == 'Power' else 0
        new_task = new_value if dimension == 'TaskOriented' else 0
    else:
        new_intimacy = new_value if dimension == 'Intimacy' else current['Intimacy']
        new_power = new_value if dimension == 'Power' else current['Power']
        new_task = new_value if dimension == 'TaskOriented' else current['TaskOriented']
    
    # 置換パターン
    old_pattern = r'Intimacy=(-?\d+),?\s*Power=(-?\d+),?\s*TaskOriented=(-?\d+)'
    new_string = f'Intimacy={new_intimacy}, Power={new_power}, TaskOriented={new_task}'
    
    return re.sub(old_pattern, new_string, prompt)


def generate_utterance(prompt: str, verbose: bool = False) -> Optional[str]:
    """プロンプトからLLMで発話を生成"""
    if not HAS_GEMINI:
        print("Error: Gemini APIが利用できません")
        return None
    
    try:
        model = get_gemini_model()
        response = run_gemini(model, prompt, max_tokens=200, temperature=0.7)
        
        # 発話部分を抽出（余計な説明を除去）
        # 最初の行または「」内のテキストを取得
        lines = response.strip().split('\n')
        utterance = lines[0].strip()
        
        # 「」で囲まれていれば中身を抽出
        if '「' in utterance and '」' in utterance:
            match = re.search(r'「(.+?)」', utterance)
            if match:
                utterance = match.group(1)
        
        # *で囲まれたアクション説明を除去
        utterance = re.sub(r'\*[^*]+\*', '', utterance).strip()
        
        # [END]タグを除去
        utterance = re.sub(r'\s*\[END\]', '', utterance).strip()
        
        return utterance if utterance else None
        
    except Exception as e:
        if verbose:
            print(f"  生成エラー: {e}")
        return None


def normalize_speaker_name(speaker: str) -> str:
    """スピーカー名を英語の正規化名に変換"""
    speaker_map = {
        '惣流・アスカ・ラングレー': 'asuka',
        'アスカ': 'asuka',
        '碇シンジ': 'shinji',
        'シンジ': 'shinji',
    }
    return speaker_map.get(speaker, speaker.lower().replace(' ', '_').replace('・', '_'))


def generate_fewshot_dataset(
    data_dir: str,
    dimension: str,
    num_prompts: int = 10,
    output_dir: str = None,
    verbose: bool = False,
    zero_other_dims: bool = False
) -> Dict[str, List[Tuple[int, str]]]:
    """指定次元のfew-shotデータセットを生成
    
    Args:
        data_dir: 元データのディレクトリ
        dimension: 対象次元 ('Intimacy', 'Power', 'TaskOriented')
        num_prompts: 抽出するプロンプト数
        output_dir: 出力ディレクトリ
        verbose: 詳細出力
        zero_other_dims: 他の次元を0に固定するかどうか
    
    Returns:
        {speaker: [(value, utterance), ...], ...}
    """
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # データ読み込み
    pairs = load_prompt_utterance_pairs(data_dir)
    if not pairs:
        print(f"データが見つかりません: {data_dir}")
        return {}
    
    # スピーカー別に分類
    speaker_pairs = {}
    for pair in pairs:
        speaker = pair.get('speaker', 'unknown')
        if speaker not in speaker_pairs:
            speaker_pairs[speaker] = []
        # 関係値が含まれるプロンプトのみ
        if extract_relationship_from_prompt(pair.get('prompt', '')):
            speaker_pairs[speaker].append(pair)
    
    print(f"\n発見したスピーカー: {list(speaker_pairs.keys())}")
    
    results = {}
    
    for speaker, sp_pairs in speaker_pairs.items():
        print(f"\n{'='*60}")
        print(f"スピーカー: {speaker}")
        print(f"利用可能プロンプト数: {len(sp_pairs)}")
        print(f"{'='*60}")
        
        if len(sp_pairs) == 0:
            continue
        
        # プロンプトをランダムにサンプリング
        random.shuffle(sp_pairs)
        selected = sp_pairs[:num_prompts]
        
        print(f"選択プロンプト数: {len(selected)}")
        
        # スピーカー名を正規化
        speaker_normalized = normalize_speaker_name(speaker)
        
        # スピーカー別の出力ディレクトリを作成（zero_other_dimsの場合は別ディレクトリ）
        data_subdir = 'true_data_other_zero' if zero_other_dims else 'true_data'
        speaker_output_dir = os.path.join(output_dir, data_subdir, speaker_normalized)
        os.makedirs(speaker_output_dir, exist_ok=True)
        
        # ベースプロンプトを別ファイルに保存
        dim_lower = dimension.lower()
        base_prompts_filename = f"{speaker_normalized}_{dim_lower}_base_prompts.txt"
        base_prompts_path = os.path.join(speaker_output_dir, base_prompts_filename)
        
        with open(base_prompts_path, 'w', encoding='utf-8') as f:
            f.write(f"# {speaker} - {dimension} ベースプロンプト\n")
            f.write(f"# 抽出元: {data_dir}\n")
            f.write(f"# プロンプト数: {len(selected)}\n")
            f.write("=" * 70 + "\n\n")
            for i, pair in enumerate(selected):
                f.write(f"[プロンプト {i+1}]\n")
                f.write(pair.get('prompt', '') + "\n")
                f.write("-" * 50 + "\n\n")
        
        print(f"ベースプロンプト保存: {base_prompts_path}")
        
        # 各関係値（-3〜+3）で発話を生成
        dataset = []
        
        for i, pair in enumerate(selected):
            original_prompt = pair.get('prompt', '')
            print(f"\n[{i+1}/{len(selected)}] ベースプロンプトから生成中...")
            
            for value in range(-3, 4):  # -3, -2, -1, 0, 1, 2, 3
                # 関係値を変更したプロンプトを作成
                modified_prompt = replace_relationship_value(
                    original_prompt, dimension, value, zero_other_dims
                )
                
                if verbose:
                    current = extract_relationship_from_prompt(modified_prompt)
                    print(f"  {dimension}={value:+d}: ", end="")
                
                # 発話を生成
                utterance = generate_utterance(modified_prompt, verbose)
                
                if utterance:
                    dataset.append((value, utterance))
                    if verbose:
                        print(f"「{utterance[:30]}...」" if len(utterance) > 30 else f"「{utterance}」")
                else:
                    if verbose:
                        print("(生成失敗)")
        
        results[speaker] = dataset
        
        # CSVに保存（英語ヘッダー、値のみ）
        csv_filename = f"{speaker_normalized}_{dim_lower}_fewshot.csv"
        csv_path = os.path.join(speaker_output_dir, csv_filename)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([dimension, 'utterance'])
            for value, utterance in dataset:
                writer.writerow([value, utterance])
        
        print(f"\n保存: {csv_path} ({len(dataset)}件)")
    
    return results


def get_data_dir_for_dimension(base_dir: str, dimension: str) -> str:
    """次元に応じたデータディレクトリを取得"""
    dim_to_dir = {
        'Intimacy': 'eva_intimacy_increase',
        'Power': 'eva_power_increase',
        'TaskOriented': 'eva_task_increase'
    }
    subdir = dim_to_dir.get(dimension, 'eva_intimacy_increase')
    return os.path.join(base_dir, subdir)


def main():
    parser = argparse.ArgumentParser(description='Few-shot学習用正解データセット生成')
    parser.add_argument('--data-dir', type=str, default='out',
                       help='ベースデータディレクトリ（デフォルト: out）')
    parser.add_argument('--dimension', type=str, 
                       choices=['Intimacy', 'Power', 'TaskOriented'],
                       help='対象次元')
    parser.add_argument('--all-dimensions', action='store_true',
                       help='全次元のデータを生成')
    parser.add_argument('--num-prompts', type=int, default=10,
                       help='抽出するプロンプト数（デフォルト: 10）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='出力ディレクトリ（デフォルト: スクリプトと同じディレクトリ）')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='詳細出力')
    parser.add_argument('--seed', type=int, default=42,
                       help='乱数シード')
    parser.add_argument('--zero-other-dims', action='store_true',
                       help='対象次元以外の関係値を0に固定する')
    
    args = parser.parse_args()
    
    # 乱数シード設定
    random.seed(args.seed)
    
    # API設定
    if HAS_GEMINI:
        set_gemini_key()
    else:
        print("Error: Gemini APIが利用できません")
        return
    
    # 対象次元の決定
    if args.all_dimensions:
        dimensions = ['Intimacy', 'Power', 'TaskOriented']
    elif args.dimension:
        dimensions = [args.dimension]
    else:
        print("Error: --dimension または --all-dimensions を指定してください")
        return
    
    # 出力ディレクトリ
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Few-shot正解データセット生成")
    print("=" * 70)
    print(f"ベースディレクトリ: {args.data_dir}")
    print(f"対象次元: {dimensions}")
    print(f"プロンプト数: {args.num_prompts}")
    print(f"他次元ゼロ固定: {args.zero_other_dims}")
    print(f"出力先: {output_dir}")
    
    # 生成実行
    all_results = {}
    for dim in dimensions:
        # 次元に応じたデータディレクトリを取得
        data_dir = get_data_dir_for_dimension(args.data_dir, dim)
        
        print(f"\n\n{'#'*70}")
        print(f"# {dim} の正解データセット生成")
        print(f"# データソース: {data_dir}")
        print(f"{'#'*70}")
        
        results = generate_fewshot_dataset(
            data_dir=data_dir,
            dimension=dim,
            num_prompts=args.num_prompts,
            output_dir=output_dir,
            verbose=args.verbose,
            zero_other_dims=args.zero_other_dims
        )
        all_results[dim] = results
    
    # サマリー
    print("\n" + "=" * 70)
    print("生成完了サマリー")
    print("=" * 70)
    
    for dim, results in all_results.items():
        dim_ja = {'Intimacy': '親密度', 'Power': '力関係', 'TaskOriented': '目的指向性'}.get(dim, dim)
        print(f"\n【{dim_ja}】")
        for speaker, dataset in results.items():
            print(f"  {speaker}: {len(dataset)}件")
            # 値ごとの分布
            value_counts = {}
            for value, _ in dataset:
                value_counts[value] = value_counts.get(value, 0) + 1
            dist_str = ", ".join([f"{v:+d}:{c}" for v, c in sorted(value_counts.items())])
            print(f"    分布: {dist_str}")


if __name__ == '__main__':
    main()
