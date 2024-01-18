import jieba
from tqdm import tqdm
import os, pickle
filePath = r'D:\datasets\h-corpus'
filePath = [os.path.join(filePath, f) for f in os.listdir(filePath) if f.endswith('txt')]

# wSet = {}
with open('tmp_wSet.pkl', 'rb') as file:
    wSet = pickle.load(file)

print('start')

for i in tqdm(range(2001, len(filePath)), desc="Converting"):
    path = filePath[i]
    with open(path, 'r', encoding='utf-8') as f:
        tmp = f.read()
    seg_list = jieba.cut(tmp)
    for x in seg_list:
        wSet[x] = wSet.get(x, 0) + 1

    if i%1000 == 0 :
        with open('tmp_wSet.pkl', 'wb') as file:
            pickle.dump(wSet, file)
        print(i)

with open('tmp_wSet.final.pkl', 'wb') as file:
    pickle.dump(wSet, file)


