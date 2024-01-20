import json
from opencc import OpenCC
cc = OpenCC('t2s')  # 't2s'表示繁体转简体

root = r'D:\datasets\l-corpus\\'
jsonl_file_path = root + '金庸小说15本.jsonl'
with open(jsonl_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析JSON对象
        json_object = json.loads(line)
        with open(root + json_object['title'] + '.txt', 'w', encoding='utf-8') as f:
            f.write(cc.convert(json_object['text']))
