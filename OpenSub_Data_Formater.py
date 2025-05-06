from datasets import load_dataset
from collections import defaultdict
import json

dataset = load_dataset("open_subtitles", "en-hi", split="train")

def to_chatml_format(convo):
    roles = ["user", "assistant"]
    return {"messages": [
        {"role": roles[i % 2], "content": msg} for i, msg in enumerate(convo)
    ]}

grouped = defaultdict(list)

for item in dataset:
    sid = item['meta']['subtitleId']['en']
    sent_id = item['meta']['sentenceIds']['en'][0]
    text = item['translation']['en']
    
    if text and sid is not None:
        grouped[sid].append((sent_id, text.strip()))

  for sid in grouped:
    grouped[sid] = sorted(grouped[sid], key=lambda x: x[0])

conversations = []

for group in grouped.values():
    lines = [text for _, text in group]
    for i in range(0, len(lines) - 1, 4):  # step size 4
        convo = lines[i:i+4]
        if len(convo) >= 2:
            conversations.append(convo)

      formatted_convos = [to_chatml_format(c) for c in conversations]

with open("opensubs_chatml.jsonl", "w") as f:
    for convo in formatted_convos:
        f.write(json.dumps(convo) + "\n")

for convo in formatted_convos:
    con = convo['messages']
    print(con)
    break
