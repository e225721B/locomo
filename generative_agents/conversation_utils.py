import json, re, os
import random
from icrawler.builtin import ImageDownloader
from icrawler.builtin import GoogleImageCrawler
from global_methods import run_chatgpt, run_chatgpt_with_examples


PERSONA_FROM_MSC_PROMPT = (
    "Let's write speaker descriptions from a given set of life attributes. Example:\n\n%s\n\n"
    "Note: Add crucial details in the persona about the person such as their name, age, marital status, gender, job etc. "
    "Add additional details like names of family/friends or specific activities, likes and dislikes, experiences when appropriate.\n\n"
    "For the following attributes, write a persona. Return ONLY a valid JSON object (no prose) with exactly the keys 'persona' and 'name'.\n\n%s\n"
    "Start your answer immediately with '{'.\n"
)


EVENT2QUERY_PROMPT = "Let's write short image search queries in order to find a suitable image for illustrating the given events. Queries should not include names of people, years and other irrelevant details. For example:\n\nInput: A picture of the modern art museum he visited with his grandchildren in Paris in 2018.\nOutput: modern art museum in Paris\n\nInput: A picture of the shared room she and her siblings lived in when she was growing up.\nOutput: cramped room with multiple beds\n\nInput: A photo of the new art supplies Jeremy bought for his upcoming art project with his mentor.\nOutput: new art supplies on a table\n\nInput: A picture of the delicious homemade vegetable smoothie she prepared using fresh produce from her well-organized garden, which she loves to maintain every morning.\n Output: produce garden at home\n\nWrite search queries for the following inputs.\n\n%s\n\nWrite answers in the form of a json list, where each entry is a query."


AGENT_CONV_PROMPT_SESS_1 = "%s\n\n%s は %s と初めて会話します。今日は %s です。あなたは %s になりきり、%s に対して次に言う一言を書いてください。会話を始める場合は、相手の近況を尋ねるか、最近あなたに起きた出来事について話してください。これまでの会話で共有した情報は繰り返さないでください。会話は個人的で、家族・友人・好き嫌い・将来の希望などに触れてください。'last Friday' や 'next month'、'when I was ten years old' のような時間参照や、具体的な場所名を含めてください。返答は 20 語相当以内の短い一文で書いてください。例えば、\n\n%s: 子どもの頃、母がよくパイナップルの誕生日ケーキを焼いてくれて大好きでした。\n\n会話を終えるときは『Bye!』と書いてください。\n\nCONVERSATION:\n\n"

AGENT_CONV_PROMPT_SESS_1_W_EVENTS = """
与えられた PERSONALITY を用いて、会話の次にあなたが言う一言を書いてください。
- 会話を始める場合は、相手のことを尋ねるか、最近あなたに起きた出来事について話してください。
- これまでの会話で共有した情報は繰り返さないでください。
- 'last Friday' や 'next month'、'when I was ten years old' のような時間参照、特定の人物への言及を含めてください。
- 返答は 20 語相当以内の短い一文で書いてください。
- 以前の会話内容へのフォローアップの質問を入れてもかまいません。

PERSONALITY: %s

%s は %s と初めて会話します。今日は %s です。%s の人生で以下の出来事が起きました。
EVENTS: %s

あなたは %s になりきり、%s と親しみのある会話の中でこれらの EVENTS について話してください。%s
"""


AGENT_CONV_PROMPT = "%s\n\n%s は %s と %s に最後に話しました。%s\n\n今日は %s です。あなたは %s になりきり、%s に対して次に言う一言を書いてください。会話を始める場合は、相手の近況を尋ねる、以前の会話のフォローアップをする、または相手が興味を持ちそうな最近の出来事を話してください。これまでに共有した情報は繰り返さないでください。会話は個人的で、家族・友人・好き嫌い・将来の希望などに触れてください。'last Friday' や 'next month'、'when I was ten years old' のような時間参照や、具体的な場所名を含めてください。返答は 20 語相当以内の短い一文で書いてください。例えば、\n\n%s: 子どもの頃、母がよくパイナップルの誕生日ケーキを焼いてくれて大好きでした。\n\n会話を終えるときは『Bye!』と書いてください。\n\nCONVERSATION:\n\n"


AGENT_CONV_PROMPT_W_EVENTS = """
与えられた PERSONALITY を用いて、会話の次にあなたが言う一言を書いてください。
- 会話を始める場合は、相手のことを尋ねるか、最近あなたに起きた出来事について話してください。
- これまでの会話で共有した情報は繰り返さないでください。
- 会話は個人的で、家族・友人・好き嫌い・将来の希望などに触れてください。
- 'last Friday' や 'next month'、'when I was ten years old' のような時間参照、特定の人物への言及を含めてください。
- 返答は 20 語相当以内の短い一文で書いてください。
- 以前の会話内容へのフォローアップの質問を入れてもかまいません。

PERSONALITY: %s

%s は %s と %s に最後に話しました。

%s

今日は %s です。あなたは %s です。前回この相手に会ってから、あなたの人生で以下の出来事が起きました:
%s

これらの EVENTS を会話に活用してください。%s あなたの PERSONALITY に沿って、%s とのこの会話で次に言う一言を書いてください:
"""


AGENT_CONV_PROMPT_W_EVENTS_V2_INIT = """
与えられた PERSONALITY を用いて、会話の次にあなたが言う一言を書いてください。
- 返答は 20 語相当以内の短い一文で書いてください。
- 感情・好き嫌い・願望・人間関係などを扱い、会話は深く個人的な内容にしてください。重要な出来事は具体的に述べてください。
- これまでの会話で共有した情報は繰り返さないでください。
- 'last Friday' や 'next month'、'when I was ten years old' のような時間参照、特定の人物への言及を含めてください。
- ときどき、前回の会話や現在の話題に対するフォローアップ質問をしてください。
- 屋外活動については話さないでください。

PERSONALITY: %s


%s は %s と %s に最後に話しました。今日は %s です。あなたは %s です。 

これまでの会話の要約は次のとおりです。
SUMMARY:
%s

前回この相手に会ってから、あなたの人生で以下の出来事が起きました。
EVENTS:
%s



%s あなたと %s のこの会話で、思慮深い次の一言を書いてください。与えられた EVENTS のみを取り上げ、それがあなたの人生に与える影響を語ってください。EVENTS が否定的なら、動揺や辛さも表現してください。:
"""


AGENT_CONV_PROMPT_W_EVENTS_V2 = """
与えられた PERSONALITY を用いて、会話の次にあなたが言う一言を書いてください。
- 返答は 20 語相当以内の短い一文で書いてください。
- 感情・好き嫌い・願望・人間関係などを扱い、会話は深く個人的な内容にしてください。重要な出来事は具体的に述べてください。
- これまでの会話で共有した情報は繰り返さないでください。
- 'last Friday' や 'next month'、'when I was ten years old' のような時間参照、特定の人物への言及を含めてください。
- ときどき、前回の会話や現在の話題に対するフォローアップ質問をしてください。
- 屋外活動については話さないでください。

PERSONALITY: %s

%s は %s と %s に最後に話しました。今日は %s です。あなたは %s です。 

これまでの会話の要約は次のとおりです。
SUMMARY:
%s

前回この相手に会ってから、あなたの人生で以下の出来事が起きました。
EVENTS:
%s

双方が知っている関連情報は次のとおりです。
RELEVANT_CONTEXT:
%s

%s あなたと %s のこの親密な会話で、思慮深い次の一言を書いてください。与えられた EVENTS のみを取り上げ、それがあなたの人生に与える影響を語ってください。EVENTS が否定的なら、動揺や辛さも表現してください。:
"""


ALIGNMENT_PROMPT = "Let's write whether the given image is relevant to the dialog. Indicate 1 if the image is relevant and 0 if the image is not relevant. For example,\n\nDialog: So Jeremy, how was your day? Anything interesting happen?\nImage Caption: A photo of the garden she planted and cultivated in her backyard with her daughter last year.\nOutput: 0\n\nDialog: Hey Lauri! My day was pretty good. I went to the art museum with my mentor and saw some amazing pieces. How about you? How was your day?\nImage Caption: A selfie of him and his mentor at the museum art exhibit they went to two weeks ago\nOutput: 1\n\nIndicate whether the image is relevant to the dialog for the following dialog and image caption. Output 0 or 1.\n\n"


DIALOG2IMAGE_QUERY_PROMPT = "Let's write short image search queries from textual descriptions of photos shared by a user. Queries should not include names of people, years and other irrelevant details. For example:\n\nInput: That sounds relaxing, Jeremy! As for video game suggestions, have you ever tried \"The Legend of Zelda: Breath of the Wild\"? It's an open-world adventure game that I absolutely love. [shares a photo of Link standing in front of a breathtaking landscape] Have a look at this stunning view!\nOutput: the legend of zelda: breath of wild link landscape\n\nInput: That sounds like such a special memory. Learning how to ride a bike is definitely a milestone. Do you still enjoy biking now? [shares a photo of a scenic bike trail] This is a beautiful bike trail I came across recently. It looks like a peaceful place to ride.\nOutput: scenic bike trail\n\nInput: Yes, we also visited a beautiful sunflower field in Korea. [shares a photo of a vast field of sunflowers] It was such a stunning sight with rows and rows of vibrant yellow flowers stretching as far as the eye could see. It was definitely a highlight of our trip. Have you ever seen a sunflower field before?\n Output: sunflower field korea\n\nWrite search query for the following input.\n\nInput: %s\nOutput: "

CASUAL_DIALOG_PROMPT = "Make the sentence short, less formal, less grandiose and more casual. \n\nInput: %s\nOutput: "


SESSION_SUMMARY_PROMPT = "Previous conversations between %s and %s so far can be summarized as follows: %s. The current time and date are %s. %s and %s just had the following conversation:\n\n%s\n\nSummarize the previous and current conversations between %s and %s in 150 words or less. Include key facts about both speakers and time references.\n\n"


SESSION_SUMMARY_INIT_PROMPT = "Write a concise summary containing key facts mentioned about %s and %s on %s in the following conversation:\n\n%s\n\n"


VISUAL_QUESTION_PROMPT = "{}\n\n{}\n\n{} says, {}, and {}. Write the most natural question or comment {} can include in her response."


def get_msc_persona(args):
    # check if personas exist, else generate persona + summary
    if (os.path.exists(args.agent_a_file) and os.path.exists(args.agent_b_file)) and not args.overwrite_persona:
        return None, None
    else:
        all_personas = json.load(open('./data/msc_personas_all.json'))
        selected_idx = random.choice([idx for idx, d in enumerate(all_personas['train']) if not d["in_dataset"]])
        attributes = all_personas['train'][selected_idx]
        with open('./data/msc_personas_all.json', "w") as f:
            all_personas['train'][selected_idx]["in_dataset"] = 1
            json.dump(all_personas, f, indent=2)
        agent_a = get_persona(args, attributes['Speaker 1'])

        agent_a['persona_summary'] = agent_a['persona']
        agent_a['msc_prompt'] = attributes['Speaker 1']
        agent_b = get_persona(args, attributes['Speaker 2']) # setting the second agent to have age within +/- 5 years of first agent

        agent_b['persona_summary'] = agent_b['persona']
        agent_b['msc_prompt'] = attributes['Speaker 2']
        del agent_a['persona']
        del agent_b['persona']
        print("Agent A Persona: %s" % agent_a['persona_summary'])
        print("Agent B Persona: %s" % agent_b['persona_summary'])
    return agent_a, agent_b


def get_persona(args, attributes, target='human', ref_age=None):
    """MSC 属性からペルソナ JSON を生成 (堅牢リトライ付き)。"""
    task = json.load(open(os.path.join(args.prompt_dir, 'persona_generation_examples.json')))
    persona_examples = [
        task["input_prefix"] + json.dumps(e["input"], indent=2) + '\n' + task["output_prefix"] + e["output"]
        for e in task['examples']
    ]
    input_string = task["input_prefix"] + json.dumps(attributes, indent=2)
    query = PERSONA_FROM_MSC_PROMPT % (persona_examples, input_string)
    if getattr(args, 'lang', 'en') == 'ja':
        query += "\n出力する JSON の 'persona' の値は自然で詳細な日本語で記述してください (キー名は英語のまま)。"

    def extract_first_json(txt: str) -> str | None:
        txt = txt.strip()
        start = txt.find('{')
        if start == -1:
            return None
        stack = []
        for i in range(start, len(txt)):
            c = txt[i]
            if c == '{':
                stack.append(c)
            elif c == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        return txt[start:i+1]
        return None

    last_fragment = None
    for attempt in range(1, 6):
        raw = run_chatgpt(query, num_gen=1, num_tokens_request=800, use_16k=True).strip()
        frag = extract_first_json(raw) or raw
        last_fragment = frag
        try:
            data = json.loads(frag)
            if isinstance(data, dict):
                data = {k.lower(): v for k, v in data.items()}
                # 必須キー検証
                if 'persona' in data and 'name' in data:
                    return data
            # list や str の場合は clean_json_output で再処理
            if isinstance(data, (list, str)):
                cleaned = clean_json_output(raw)
                if isinstance(cleaned, dict):
                    cleaned = {k.lower(): v for k, v in cleaned.items()}
                return cleaned
        except Exception as e:  # JSONDecodeError 他
            print(f"Persona JSON parse retry {attempt}: {type(e).__name__}: {e}")
    # 最終簡易修復
    if last_fragment:
        repaired = last_fragment.rstrip(',;\n ') + '}' if last_fragment.count('{') > last_fragment.count('}') else last_fragment
        try:
            data = json.loads(repaired)
            if isinstance(data, dict):
                data = {k.lower(): v for k, v in data.items()}
            return data
        except Exception:
            pass
    raise RuntimeError(f"Failed to generate persona JSON after retries. Last fragment: {last_fragment[:200] if last_fragment else 'None'}")


def get_datetime_string(input_time='', input_date=''):

    assert input_time or input_date

    if input_date:
        year, month, day = input_date
    if input_time:
        hour, min = input_time
        time_mod = 'am' if hour <= 12 else 'pm'
        hour = hour if hour <= 12 else hour-12
        min = str(min).zfill(2)

    if input_time and not input_date:
        return str(hour) + ':' + min + ' ' + time_mod
    elif input_date and not input_time:
        return day + ' ' + month + ', ' + year
    else:
        return str(hour) + ':' + min + ' ' + time_mod + ' on ' + day + ' ' + month + ', ' + year 


def insert_image(text, events):

    dialog = {"text": text, "raw_text": text}

    if len(events) == 0:
        return dialog
    id_2_event = {e["img_id"]: e for e in events}
    matches = re.findall(r"\[(?i)SHARES [1-9]\]", text)
    for m in matches:
        mid = int(m[-2:-1])
        dialog["text"] = dialog["text"].replace(m, '')
        
        try:
            assert mid in id_2_event, [text, m, mid]
            dialog["img_url"] = id_2_event[mid]["img_url"][0]
            dialog["img_file"] = id_2_event[mid]["img_file"][0]
            dialog["img_id"] = id_2_event[mid]["img_id"]
            dialog["image"] = id_2_event[mid]["image"]
            if "caption" in id_2_event[mid]:
                dialog["caption"] = id_2_event[mid]["caption"]

        except AssertionError:
            print("Did not find %s in events" % str(mid))
            continue

    return dialog


def get_images(query, out_dir, file_offset):
    
    google_crawler = GoogleImageCrawler(downloader_cls=CustomLinkPrinter, storage={'root_dir': out_dir})
    google_crawler.downloader.file_urls = []
    google_crawler.downloader.file_names = []
    google_crawler.crawl(keyword=query, max_num=1, file_idx_offset=file_offset, overwrite=True, filters={'type': 'photo', 'size': '=3024x4032'}) # 'license': 'commercial,modify'
    file_urls =  google_crawler.downloader.file_urls
    file_names = google_crawler.downloader.file_names

    if file_names == []:
        google_crawler = GoogleImageCrawler(downloader_cls=CustomLinkPrinter, storage={'root_dir': out_dir})
        google_crawler.downloader.file_urls = []
        google_crawler.downloader.file_names = []
        google_crawler.crawl(keyword=query, max_num=1, file_idx_offset=file_offset, overwrite=True, filters={'type': 'photo', 'size': '=4032x3024'}) # 'license': 'commercial,modify'
        file_urls =  google_crawler.downloader.file_urls
        file_names = google_crawler.downloader.file_names
    
    return file_urls, file_names


def replace_captions(text, args):

    task = json.load(open(os.path.join(args.prompt_dir, 'image_sharing_examples.json')))
    query = task['prompt']
    examples = []
    for e in task['examples']:
        examples.append([task['input_format'].format(*e["input"]), e["output"]])

    text = text.replace('[END]', '')
    matches = re.findall(r"\[.*\]", text)
    for m in matches:
        if text.replace(m ,'').isspace():
            return ""
        else:
            new_text = run_chatgpt_with_examples(query, examples, m[1:-1], num_gen=1, num_tokens_request=1000, use_16k=False)
            if len(set(text.replace(m, '').split()).intersection(new_text.split())) < 0.5 * len(set(text.replace(m, '').split())):
                text = text.replace(m, '')
            else:
                text = new_text
        break

    return text

def insert_image_response(text):

    matches = re.findall(r"\[.*\]", text)

    image_search_query = None
    m = None
    for m in matches:
        if 'share' in m or 'Share' in m:
            image_search_query = run_chatgpt(DIALOG2IMAGE_QUERY_PROMPT % text, 1, 20, 'chatgpt').strip()
            break
        else:
            text = text.replace(m, '')

    return image_search_query, m


def merge_captions(conv_dir, caption_file):

    captions = json.load(open(caption_file))
    agent_a = json.load(open(os.path.join(conv_dir, 'agent_a.json')))
    agent_b = json.load(open(os.path.join(conv_dir, 'agent_b.json')))

    for c in captions:
        head, img_file_name = os.path.split(c["img_file"])
        head, agent = os.path.split(head)
        head, session_id = os.path.split(head)
        head, conv_id = os.path.split(head)
        # print(agent, session_id, img_file_name)
        if agent == 'a':
            for i, e in enumerate(agent_a['events_%s' % session_id]):
                if e['img_file'][0] == img_file_name:
                    agent_a['events_%s' % session_id][i]["caption"] = c["summary"]
        else:
            for i, e in enumerate(agent_b['events_%s' % session_id]):
                if e['img_file'][0] == img_file_name:
                    agent_b['events_%s' % session_id][i]["caption"] = c["summary"]
    
    with open(os.path.join(conv_dir, 'agent_a_captions.json'), 'w') as f:
        json.dump(agent_a, f, indent=2)
    with open(os.path.join(conv_dir, 'agent_b_captions.json'), 'w') as f:
        json.dump(agent_b, f, indent=2)


def insert_image_in_dialog(session, agent_a_events, agent_b_events, agent_a_name, agent_b_name):

    agent_a_id_2_event = {e["img_id"]: e for e in agent_a_events}
    agent_b_id_2_event = {e["img_id"]: e for e in agent_b_events}

    for i in range(len(session)):
        text = session[i]["text"]
        matches = re.findall(r"\[shares photo [1-9]\]", text)
        for m in matches:
            mid = int(m[-2:-1])
            if session[i]["speaker"] == agent_a_name:

                session[i]["text"] = session[i]["text"].replace(m, '')
                
                if "url" not in session[i]:
                    session[i]["url"] = []
                try:
                    assert mid in agent_a_id_2_event, [text, m, mid]
                    session[i]["url"].append(agent_a_id_2_event[mid]["img_url"][0])
                except AssertionError:
                    continue

            if session[i]["speaker"] == agent_b_name:
                
                session[i]["text"] = session[i]["text"].replace(m, '')

                if "url" not in session[i]:
                    session[i]["url"] = []
                try:
                    assert mid in agent_b_id_2_event
                    session[i]["url"].append(agent_b_id_2_event[mid]["img_url"][0])
                except AssertionError:
                    continue

    return session


def clean_dialog(output, name):

    if output.startswith(name):
        output = output[len(name):]
        output = output.strip()
        if output[0] == ':':
            output = output[1:]
            output = output.strip()
    
    return output


def clean_json_output(output_string):

    print(output_string)

    output_string = output_string.strip()

    if output_string[0] == '[' and output_string[-1] != ']':
        start_index = output_string.index('[')
        end_index = output_string.rindex(']')
        output_string = output_string[start_index:end_index+1]

    if output_string[0] == '{' and output_string[-1] != '}':
        start_index = output_string.index('{')
        end_index = output_string.rindex('}')
        output_string = output_string[start_index:end_index+1]

    # balance brackets in json
    num_start_bracket = len(find_indices(output_string, '{'))
    num_end_bracket = len(find_indices(output_string, '}'))

    if num_start_bracket != num_end_bracket:
        if num_end_bracket < num_start_bracket:
            output_string = output_string + ' '.join(['}']*(num_start_bracket-num_end_bracket))
        if num_start_bracket < num_end_bracket:
            output_string = ' '.join(['{']*(num_end_bracket-num_start_bracket)) + ' ' + output_string

    # balance brackets in json
    num_start_bracket = len(find_indices(output_string, '['))
    num_end_bracket = len(find_indices(output_string, ']'))

    if num_start_bracket != num_end_bracket:
        if num_end_bracket < num_start_bracket:
            output_string = output_string + ' '.join(['[']*(num_start_bracket-num_end_bracket))
        if num_start_bracket < num_end_bracket:
            output_string = ' '.join([']']*(num_end_bracket-num_start_bracket)) + ' ' + output_string

    return json.loads(output_string)


def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices


class CustomLinkPrinter(ImageDownloader):
    
    file_urls = []
    file_names = []

    def get_filename(self, task, default_ext):
        file_idx = self.fetched_num + self.file_idx_offset
        file_url = task['file_url']
        # self.file_urls.append(file_url)
        return '{:04d}.{}'.format(file_idx, default_ext)

    def download(self, task, default_ext, timeout=5, max_retry=3, overwrite=False, **kwargs):
        """Download the image and save it to the corresponding path.

        Args:
            task (dict): The task dict got from ``task_queue``.
            timeout (int): Timeout of making requests for downloading images.
            max_retry (int): the max retry times if the request fails.
            **kwargs: reserved arguments for overriding.
        """
        file_url = task["file_url"]
        task["success"] = False
        task["filename"] = None
        retry = max_retry

        if not overwrite:
            with self.lock:
                self.fetched_num += 1
                filename = self.get_filename(task, default_ext)
                if self.storage.exists(filename):
                    self.logger.info("skip downloading file %s", filename)
                    return
                self.fetched_num -= 1

        while retry > 0 and not self.signal.get("reach_max_num"):
            try:
                response = self.session.get(file_url, timeout=timeout)
            except Exception as e:
                self.logger.error(
                    "Exception caught when downloading file %s, " "error: %s, remaining retry times: %d",
                    file_url,
                    e,
                    retry - 1,
                )
            else:
                if self.reach_max_num():
                    self.signal.set(reach_max_num=True)
                    break
                elif response.status_code != 200:
                    self.logger.error("Response status code %d, file %s", response.status_code, file_url)
                    break
                elif not self.keep_file(task, response, **kwargs):
                    break
                with self.lock:
                    self.fetched_num += 1
                    filename = self.get_filename(task, default_ext)
                self.logger.info("image #%s\t%s", self.fetched_num, file_url)
                self.file_urls.append(file_url)
                self.file_names.append(filename)
                self.storage.write(filename, response.content)
                task["success"] = True
                task["filename"] = filename
                break
            finally:
                retry -= 1

    # def download(self, task, default_ext, timeout=5, max_retry=3, overwrite=False, **kwargs):
    #     file_url = task['file_url']
    #     filename = self.get_filename(task, default_ext)

    #     task['success'] = True
    #     task['filename'] = filename

    #     if not self.signal.get('reach_max_num'):
    #         self.file_urls.append(file_url)
    #         self.file_names.append(filename)

    #     self.fetched_num += 1

    #     if self.reach_max_num():
    #         self.signal.set(reach_max_num=True)

    #     return
