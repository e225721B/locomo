import json
import os, re
import base64

header = """
<!DOCTYPE html>
<html>
<head>
    <title>Chat Example</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        .chat {
            display: flex;
            flex-direction: column;
        }
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 10px;
            font-size: 16px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .sender1 {
            background-color: #e2f0cb;
            align-self: flex-start;
        }
        .sender2 {
            background-color: #b2ebf2;
            align-self: flex-end;
        }
        .date {
            background-color: #ffb6c1;
            align-self: center;
        }
        .message img {
            max-width: 100%;
            height: auto;
            margin-top: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="chat">
    """

speaker_1_div = """
        <div class="message sender1">
            <p>%s</p>
        </div>
"""
    
speaker_1_div_with_image = """
        <div class="message sender1">
            <p>%s</p>
            <img src="%s" alt="Image 1" style="width:300px;">
            <p>%s</p>
        </div>
"""
    
speaker_2_div = """
        <div class="message sender2">
            <p>%s</p>
        </div>
"""

speaker_2_div_with_image = """
        <div class="message sender2">
            <p>%s</p>
            <img src="%s" alt="Image 2" style="width:300px;">
            <p>%s</p>
        </div>
"""

date_time_div = """
            <div class="message date">
                <p> &nbsp; &nbsp;%s&nbsp; &nbsp;</p>
            </div>
"""

rel_div = """
            <div class="message date">
                <p>%s</p>
            </div>
"""

def get_speaker_info(speaker, use_events=False):

    output = ""
    output += "<b>Name</b>: " + speaker["name"] + '<br>'
    # output += "<b>Age</b>: " + str(speaker["age"]) + '<br>'
    # output += "<b>Gender</b>: " + speaker["gender"] + '<br>'
    if 'persona_summary' in speaker:
        output += "<b>Persona</b>: " + speaker["persona_summary"] + '<br>'

    # for k, v in speaker['persona'].items():
    #     if type(v) == list:
    #         value = ', '.join(v)
    #     else:
    #         value = v
    #     output += '<b>' + k + '</b>' + ': ' + value + '<br>'

    # if use_events:
    #     output += '<b>' + 'Events' + '</b>' + '<br>'
    #     for e in speaker['events']:
    #         output += '<b>' + e['date'] + '</b>' + ': ' + e['event'] + '<br>'

    return output

def get_session_events(events):

    output = '<b>' + 'Events' + '</b>' + '<br>'
    for e in events:
        output += '<b>' + e['date'] + '</b>' + ': ' + e['sub-event'] + '<br>'
    return output


def img2base64(image_file_path):
    with open(image_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode("utf-8")


def convert_to_chat_html(speaker_1, speaker_2, outfile="", use_events=False, img_dir=None):

    body = header
    # add persona
    
    body += speaker_1_div % get_speaker_info(speaker_1, use_events=use_events)
    body += speaker_2_div % get_speaker_info(speaker_2, use_events=use_events)

    # add session
    for num in range(1, 50):
        
        if 'session_%s' % num not in speaker_1:
            break
        
        if 'session_%s_date_time' % num in speaker_1:
            date_time_string = speaker_1['session_%s_date_time' % num]
        elif 'session_%s_date' % num in speaker_1:
            date_time_string = speaker_1['session_%s_date' % num]
        else:
            raise ValueError
        
        body += date_time_div % ("Session %s [ %s ]" % (num, date_time_string))

        # Show relationships if available (written on agent_a side)
        rel = speaker_1.get('session_%s_relationships' % num)
        if isinstance(rel, dict):
            # Prefer name-aware block if present
            by = rel.get('by_speaker')
            if isinstance(by, dict) and by:
                lines = []
                for spk, obj in by.items():
                    toward = obj.get('toward', '?')
                    s = obj.get('scores', {})
                    lines.append(f"{spk} → {toward}: intimacy={s.get('intimacy','-')}, power={s.get('power','-')}, social_distance={s.get('social_distance','-')}, trust={s.get('trust','-')}")
                body += rel_div % ("<b>Relationships</b><br>" + "<br>".join(lines))
            else:
                a2b = rel.get('a_to_b', {})
                b2a = rel.get('b_to_a', {})
                text = (
                    "<b>Relationships</b><br>"
                    f"A→B: intimacy={a2b.get('intimacy','-')}, power={a2b.get('power','-')}, social_distance={a2b.get('social_distance','-')}, trust={a2b.get('trust','-')}<br>"
                    f"B→A: intimacy={b2a.get('intimacy','-')}, power={b2a.get('power','-')}, social_distance={b2a.get('social_distance','-')}, trust={b2a.get('trust','-')}"
                )
                body += rel_div % text

    # イベントグラフ機能削除に伴いイベント表示を無効化
    # if 'events_session_%s' % num in speaker_1 and 'events_session_%s' % num in speaker_2:
    #     speaker_1_events = speaker_1['events_session_%s' % num]
    #     speaker_2_events = speaker_2['events_session_%s' % num]
    #     body += speaker_1_div % get_session_events(speaker_1_events)
    #     body += speaker_2_div % get_session_events(speaker_2_events)

        for dialog in speaker_1['session_%s' % num]:
            text = dialog["clean_text"]
            if "img_url" in dialog:
                try:

                    selected_div = speaker_1_div_with_image if dialog["speaker"] == speaker_1["name"] else speaker_2_div_with_image
                    url = dialog["img_url"]
                    if type(url) == list:
                        url = url[0]
                    body += selected_div % (text, url, dialog['caption'])

                    # img_str = img2base64(os.path.join(img_dir, 'session_%s' % num, 'a' if dialog["speaker"] == speaker_1["name"] else 'b', dialog['img_file'][0]))
                    # body += selected_div % (text, 'data:image/png;base64,' + img_str, dialog['caption'])
                    
                except Exception as e:
                    print(e)
                    selected_div = speaker_1_div if dialog["speaker"] == speaker_1["name"] else speaker_2_div
                    body += selected_div % text
            else:
                selected_div = speaker_1_div if dialog["speaker"] == speaker_1["name"] else speaker_2_div
                body += selected_div % text
    body += """
        </div>
    </div>
</body>
</html>
"""
    with open(outfile, 'w') as fhtml:
        fhtml.write(body)

