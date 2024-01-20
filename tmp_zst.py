import zstandard, json
from opencc import OpenCC
import re
reg = re.compile(r'[\r\n]+[\r\n\s\-]*[\r\n]+')
reg2 = re.compile(r'\n[\r\n\s]*')
cc = OpenCC('t2s')  # 't2s'表示繁体转简体

root = r'D:\datasets\novel_cn\\'

with zstandard.open(r'E:\ai\tmp\中文小说_1.jsonl.zst', 'rt', encoding='utf-8') as file:
    for line in file:
        # 解析JSON对象
        json_object = json.loads(line)
        with open(root + json_object['title'] + '.txt', 'w', encoding='utf-8') as f:
            tmp = reg.sub('\n', json_object['text'])
            tmp2 = reg2.sub('\n', tmp).strip()
            tmp3 = cc.convert(tmp2)
            if len(tmp3) < 1600:
                print(json_object)
            else:
                f.write(tmp3)