import json
import json_lines
with open('dev.jsonl', 'r') as f:
    data = [item for item in json_lines.reader(f)]
import jsonlines
import json

with jsonlines.open('dev.jsonl', "r") as rfd:
    with open('dev.json', "w", encoding='utf-8') as wfd:
        for data in rfd:
            json.dump(data, wfd, indent=4, ensure_ascii=False)



