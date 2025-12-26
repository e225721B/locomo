import numpy as np
import json
import time
import sys
import os
import warnings

# Suppress FutureWarning from google.generativeai
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from anthropic import Anthropic

# NOTE:
# 元々このファイルは OpenAI API (Completion / ChatCompletion / Embedding) を利用していましたが、
# ユーザ要望により Gemini API に換装しました。関数名 (get_openai_embedding, run_chatgpt など) は
# 既存コードとの互換性維持のため残し、中身のみ Gemini 実装に置き換えています。
# 将来的には呼び出し元も含め名称統一 (e.g., get_embedding / run_model) を検討してください。

# 直近で成功した Gemini モデル名を記憶して優先的に使う（毎回の失敗→自動選択を避けるため）
_LAST_GOOD_GEMINI_MODEL = None


def get_openai_embedding(texts, model: str = None):
    """Gemini 埋め込み取得 (旧 OpenAI 関数名を互換のため維持)。

    Args:
        texts (List[str]): 入力テキスト群
        model (str, optional): 利用する埋め込みモデル (未指定時は環境変数 GEMINI_EMBED_MODEL → デフォルト models/text-embedding-004)

    Returns:
        np.ndarray: shape = (len(texts), embedding_dim)
    """
    if model is None:
        model = os.environ.get("GEMINI_EMBED_MODEL", "models/text-embedding-004")

    cleaned = [t.replace("\n", " ") if isinstance(t, str) else "" for t in texts]
    vectors = []
    for t in cleaned:
        try:
            # 最新 API: genai.embed_content(model=..., content=...)
            emb = genai.embed_content(model=model, content=t)
            # 返却形式は { 'embedding': [...] } を想定
            vectors.append(emb["embedding"] if isinstance(emb, dict) else emb.embedding)  # SDK バージョン差異吸収
        except Exception as e:
            print(f"Embedding error for one text: {type(e).__name__}: {e}; 0 ベクトルで埋め合わせ")
            vectors.append([0.0])
    # 長さ不一致 (エラー補填 1 次元) を避けるため最大次元にパディング
    max_dim = max(len(v) for v in vectors) if vectors else 0
    padded = [v + [0.0]*(max_dim - len(v)) for v in vectors]
    return np.array(padded)

def set_anthropic_key():
    # 既存コード互換 (未実装)
    pass

def set_gemini_key():
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY が未設定です。export GOOGLE_API_KEY=... を実行してください。")
    genai.configure(api_key=api_key)

def set_openai_key():
    """後方互換のため残存。内部的には Gemini キー設定を呼ぶ。"""
    set_gemini_key()


def run_json_trials(query, num_gen=1, num_tokens_request=1000,
                    model='davinci', use_16k=False, temperature=1.0, wait_time=1,
                    examples=None, input=None, max_retries: int = 10, retry_delay: float = 2.0):
    """LLM から JSON を取得するユーティリティ (Gemini 用・堅牢版)。

    改善点:
      - ``` や ```json などのコードフェンス除去
      - 先頭/末尾に混入したテキストを括弧スタックで JSON 部分抽出
      - partial 失敗時は再プロンプト (最大 max_retries 回)
      - 最後まで失敗した場合は最も長く整形できた JSON 断片で最終パースを試み、それでも失敗なら RuntimeError
      - sys.exit() を避け呼び出し側で例外処理可能に
    """

    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        # ```json / ``` で囲まれている場合
        if s.startswith('```'):
            # 最初の行を除去
            lines = s.splitlines()
            if lines:
                # 先頭 ```json など
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # 末尾 ``` を削除
                if lines and lines[-1].strip().startswith('```'):
                    lines = lines[:-1]
                s = '\n'.join(lines).strip()
        return s

    def _extract_json_substring(raw: str) -> str | None:
        # 最初の { から対応する } まで or [ ... ] をスタック追跡
        first_obj = raw.find('{')
        first_arr = raw.find('[')
        if first_obj == -1 and first_arr == -1:
            return None
        if first_obj == -1:
            start = first_arr
            open_char, close_char = '[', ']'
        elif first_arr == -1 or first_obj < first_arr:
            start = first_obj
            open_char, close_char = '{', '}'
        else:
            start = first_arr
            open_char, close_char = '[', ']'
        stack = []
        for i in range(start, len(raw)):
            c = raw[i]
            if c == open_char:
                stack.append(c)
            elif c == close_char:
                if stack:
                    stack.pop()
                    if not stack:
                        return raw[start:i+1]
        return None  # 閉じが見つからず

    best_fragment = None
    for attempt in range(1, max_retries + 1):
        if examples is not None and input is not None:
            output = run_chatgpt_with_examples(
                query, examples, input,
                num_gen=num_gen, wait_time=wait_time,
                num_tokens_request=num_tokens_request,
                use_16k=use_16k, temperature=temperature
            )
        else:
            output = run_chatgpt(
                query, num_gen=num_gen, wait_time=wait_time, model=model,
                num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature
            )
        if not isinstance(output, str):
            # 複数候補の場合は最初だけを見る (JSON 指示のため)
            output = output[0] if output else ''

        # 不要な 'json' 文字列除去 (ただし単語内部 "json" が鍵になるケースは稀)
        cleaned = output.strip()
        cleaned = _strip_code_fences(cleaned)
        # もし全文が JSON でなければ抽出を試みる
        candidate = cleaned
        if not (candidate.startswith('{') or candidate.startswith('[')):
            ext = _extract_json_substring(candidate)
            if ext:
                candidate = ext.strip()
        else:
            # 途中に余計なテキストが混ざるケース: 最後の閉じ括弧位置までを抽出
            ext = _extract_json_substring(candidate)
            if ext:
                candidate = ext.strip()

        # backticks が残れば再除去
        candidate = _strip_code_fences(candidate)
        # 以前より長ければ記録
        if best_fragment is None or len(candidate) > len(best_fragment):
            best_fragment = candidate

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            print(f"Retrying to avoid JsonDecodeError, trial {attempt} ...")
            # デバッグ出力 (短縮し過ぎない程度)
            snippet = candidate[:1000]
            print(snippet + ('' if len(candidate) <= 1000 else ' ... [truncated]'))
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            else:
                # 最終試行: best_fragment で再挑戦 (括弧補完など軽微な修復)
                repaired = _attempt_repair(best_fragment)
                try:
                    return json.loads(repaired)
                except Exception:
                    raise RuntimeError(f"Failed to parse JSON after {max_retries} attempts: {e}\nLast fragment (truncated 1000 chars):\n{snippet}")


def _attempt_repair(fragment: str) -> str:
    """簡易修復: 括弧数のバランス調整など。"""
    if fragment is None:
        return ''
    s = fragment.strip()
    # 末尾が開き括弧で終わるなどの明らかな未完了パターンを除去
    # 不均衡括弧補完
    opens = s.count('{'); closes = s.count('}')
    if opens > closes:
        s += '}' * (opens - closes)
    opens = s.count('['); closes = s.count(']')
    if opens > closes:
        s += ']' * (opens - closes)
    # 末尾がカンマで終わる場合削る
    while s and s[-1] in ',;':
        s = s[:-1]
    return s


def run_claude(query, max_new_tokens, model_name):

    if model_name == 'claude-sonnet':
        model_name = "claude-3-sonnet-20240229"
    elif model_name == 'claude-haiku':
        model_name = "claude-3-haiku-20240307"

    client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    # print(query)
    message = client.messages.create(
        max_tokens=max_new_tokens,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model_name,
    )
    print(message.content)
    return message.content[0].text


def run_gemini(model, content: str, max_tokens: int = 0, temperature: float = 1.0, candidate_count: int = 1):
    try:
        gen_config = {
            "temperature": temperature,
        }
        if max_tokens:
            gen_config["max_output_tokens"] = max(max_tokens, 100)  # 最低100トークンを確保
        if candidate_count > 1:
            gen_config["candidate_count"] = candidate_count
        
        # Gemini 2.5+ の thinking 機能を無効化（トークン効率のため）
        # SDK バージョンによってはサポートされていないため、try-except でフォールバック
        try:
            from google.generativeai.types import GenerationConfig
            # thinking_config がサポートされているか確認
            test_config = GenerationConfig(thinking_config={"thinking_budget": 0})
            gen_config["thinking_config"] = {"thinking_budget": 0}
        except (TypeError, ImportError):
            # thinking_config 未サポートの場合は無視
            pass
        
        # Safety設定を緩和（ロールプレイ・フィクション用途）
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        response = model.generate_content(content, generation_config=gen_config, safety_settings=safety_settings)
        
        # レスポンスの終了理由をチェック
        if hasattr(response, 'candidates') and response.candidates:
            c0 = response.candidates[0]
            finish_reason = getattr(c0, 'finish_reason', None)
            # finish_reason が MAX_TOKENS や STOP 以外の場合は警告
            if finish_reason and str(finish_reason) not in ('FinishReason.STOP', 'STOP', 'FinishReason.MAX_TOKENS', 'MAX_TOKENS', '1', '2'):
                print(f"[Gemini] Unusual finish_reason: {finish_reason}")
        
        # 複数候補
        if hasattr(response, 'candidates') and response.candidates and candidate_count > 1:
            texts = []
            for c in response.candidates[:candidate_count]:
                part_texts = []
                if hasattr(c, 'content') and getattr(c.content, 'parts', None):
                    for p in c.content.parts:
                        if hasattr(p, 'text'):
                            part_texts.append(p.text)
                texts.append("\n".join(part_texts) if part_texts else getattr(c, 'text', ''))
            return texts
        # 単一
        if hasattr(response, 'text') and response.text:
            return response.text
        # フォールバック: parts 連結
        if hasattr(response, 'candidates') and response.candidates:
            c0 = response.candidates[0]
            if hasattr(c0, 'content') and getattr(c0.content, 'parts', None):
                return "\n".join([p.text for p in c0.content.parts if hasattr(p, 'text')])
        return None
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        return None


def run_chatgpt(query, num_gen=1, num_tokens_request=1000,
                model='chatgpt', use_16k=False, temperature=1.0, wait_time=1):
    """Gemini によるテキスト生成 (旧 run_chatgpt 名称を互換維持)。

    Args:
        query (str): プロンプト (system 相当)
        num_gen (int): 生成候補数
        num_tokens_request (int): 最大トークン (Gemini: max_output_tokens)
        model (str): 呼び出し元が渡すモデル名 (gpt-3.5, gpt-4* など → Gemini にマッピング)
        use_16k (bool): 未使用 (互換保持)
        temperature (float): 生成温度
        wait_time (int): バックオフ初期値 (秒)

    Returns:
        str | List[str]: num_gen=1 なら文字列、>1 なら文字列リストgemini-2.5-flash
    """
    # モデル名 (gemini-3-flash-preview: 高性能モデルを優先)
    gemini_default = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    
    def _normalize_gemini_model_name(name: str) -> str:
        # SDK により "models/" プレフィックスの有無が異なるため吸収
        try:
            return name.replace("models/", "") if name.startswith("models/") else name
        except Exception:
            return name
    # もし gpt 風の名前が来ても Gemini を使う
    if model.startswith("gpt") or model in ("chatgpt", "davinci"):
        target_model = gemini_default
    else:
        # 呼び出し側が既に Gemini 系モデル名を指定した場合を想定
        target_model = model

    set_gemini_key()
    global _LAST_GOOD_GEMINI_MODEL
    primary = _normalize_gemini_model_name(target_model)
    fallbacks = [
        _normalize_gemini_model_name(_LAST_GOOD_GEMINI_MODEL) if _LAST_GOOD_GEMINI_MODEL else None,
        primary,
        _normalize_gemini_model_name("gemini-3-flash-preview"),

    ]
    # None を除去しつつ順序を保持したユニーク化
    seen = set()
    ordered = []
    for m in fallbacks:
        if not m or m in seen:
            continue
        seen.add(m)
        ordered.append(m)
    fallbacks = ordered
    tried = []

    last_err = None
    for model_name in fallbacks:
        if model_name in tried:
            continue
        tried.append(model_name)
        gen_model = genai.GenerativeModel(model_name)
        attempts = 0
        backoff = max(wait_time, 1)
        while attempts < 3:  # 各候補 3 回まで
            try:
                result = run_gemini(gen_model, query, max_tokens=num_tokens_request, temperature=temperature, candidate_count=num_gen)
                if result:
                    # 成功モデルをキャッシュ
                    _LAST_GOOD_GEMINI_MODEL = model_name
                    if num_gen == 1 and isinstance(result, list):
                        return result[0]
                    return result
                raise RuntimeError("Empty response from Gemini")
            except Exception as e:
                last_err = e
                attempts += 1
                print(f"Gemini generation error ({attempts}/3): {type(e).__name__}: {e}; retry in {backoff}s (model={model_name})")
                time.sleep(backoff)
                backoff *= 2
        print(f"Switching model after failures: {model_name} -> next candidate")

    # 最後の手段: ListModels で generateContent 対応のモデルを自動探索
    try:
        print("Listing available Gemini models and trying a compatible one...")
        models = genai.list_models()
        auto_candidates = []
        for m in models:
            name = getattr(m, 'name', '')
            methods = set(getattr(m, 'supported_generation_methods', []) or getattr(m, 'generation_methods', []) or [])
            if not name:
                continue
            if 'generateContent' in methods or 'generateText' in methods:
                # name は "models/xxx" 形式が返ることが多い
                auto_candidates.append(name.replace('models/', ''))
        # 優先: 1.5 系 / flash 系を先に
        auto_candidates = sorted(set(auto_candidates), key=lambda n: (not any(k in n for k in ['1.5', 'flash']), n))
        tried_set = set(tried)
        for ac in auto_candidates:
            if ac in tried_set:
                continue
            print(f"Trying auto-picked model: {ac}")
            gen_model = genai.GenerativeModel(ac)
            result = run_gemini(gen_model, query, max_tokens=num_tokens_request, temperature=temperature, candidate_count=num_gen)
            if result:
                _LAST_GOOD_GEMINI_MODEL = ac
                if num_gen == 1 and isinstance(result, list):
                    return result[0]
                return result
    except Exception as e:
        last_err = e
        print(f"Auto-pick failed: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Gemini generation failed for all candidates (" + ", ".join(fallbacks) + ") and auto-picked models. "
        "Please set GEMINI_MODEL_NAME to one of the models returned by genai.list_models() that supports generateContent, "
        "and/or upgrade google-generativeai (pip install -U google-generativeai).\n"
        f"Last error: {type(last_err).__name__ if last_err else 'Unknown'}: {last_err}"
    )
    

def run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000,
                              use_16k=False, wait_time=1, temperature=1.0):
    """Few-shot 形式を単一テキストにまとめ Gemini に送る実装。"""
    # OpenAI の role 付き messages を 1 本の指示テキストに線形化
    example_blocks = []
    for inp, out in examples:
        example_blocks.append(f"[Example Input]\n{inp}\n[Example Output]\n{out}")
    prompt = (
        f"[Task Instruction]\n{query}\n\n" +
        "\n\n".join(example_blocks) +
        f"\n\n[New Input]\n{input}\n[Answer]\n"
    )
    return run_chatgpt(prompt, num_gen=num_gen, num_tokens_request=num_tokens_request,
                        model='chatgpt', use_16k=use_16k, temperature=temperature, wait_time=wait_time)
