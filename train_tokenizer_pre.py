from opencc import OpenCC
cc = OpenCC('t2s')  # 't2s'表示繁体转简体
from tqdm import tqdm
import gzip
from h_corpus import Fileset

# 示例
traditional_text = "簡體轉換範例"
simplified_text = cc.convert(traditional_text)
print(f"繁体: {traditional_text}")
print(f"简体: {simplified_text}")

import os

filePath = Fileset(r'D:\datasets\b-corpus\unclear', ext='.txt.gz')

for path in tqdm((filePath[i] for i in range(len(filePath))), desc="Converting"):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        tmp = f.read()
    s_tmp = cc.convert(tmp)
    if s_tmp != tmp:
        print(path)
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(s_tmp)