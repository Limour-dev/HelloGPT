from opencc import OpenCC
cc = OpenCC('t2s')  # 't2s'表示繁体转简体
from tqdm import tqdm

# 示例
traditional_text = "簡體轉換範例"
simplified_text = cc.convert(traditional_text)
print(f"繁体: {traditional_text}")
print(f"简体: {simplified_text}")

import os

filePath = r'D:\datasets\h-corpus'
filePath = [os.path.join(filePath, f) for f in os.listdir(filePath) if f.endswith('txt')]

for path in tqdm(filePath, desc="Converting"):
    with open(path, 'r', encoding='utf-8') as f:
        tmp = f.read()
    s_tmp = cc.convert(tmp)
    if s_tmp != tmp:
        print(path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(s_tmp)