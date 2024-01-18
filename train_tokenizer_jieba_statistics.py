import pickle
import numpy as np
from itertools import chain
import jieba.posseg as pseg
from tqdm import tqdm

def isname(single_word_string):
    pair_word_list = pseg.lcut(single_word_string)
    for eve_word, cixing in pair_word_list:
        if cixing == "nr":
            return True
    return False


with open('tmp_wSet.final.pkl', 'rb') as file:
    wSet = pickle.load(file)

wSet_1 = {}
wSet_2 = {}
wSet_3 = {}
wSet_4 = {}
wSet_5 = {}
# wSet_n = {}
with open('tmp_wSet_n.pkl', 'rb') as file:
    wSet_n = pickle.load(file)

for k, v in tqdm(wSet.items(), "Converting"):
    # if len(k) > 1 and isname(k):
    #     if v > 1000:
    #         wSet_n[k] = v
    #     continue
    if k in wSet_n:
        continue
    if len(k) == 1:
        if v > 1000:
            wSet_1[k] = v
    elif len(k) == 2:
        if v > 2000:
            wSet_2[k] = v
    elif len(k) == 3:
        if v > 3000:
            wSet_3[k] = v
    elif len(k) == 4:
        if v > 2000:
            wSet_4[k] = v
    else:
        if v > 100:
            wSet_5[k] = v

# tmp = np.array(list(wSet_1.values()))
# np.percentile(tmp,25)
# np.mean(tmp) - np.std(tmp, ddof=1)/np.sqrt(len(tmp))

# with open('tmp_wSet_n.pkl', 'wb') as file:
#     pickle.dump(wSet_n, file)

with open('tmp_jieba.final.txt', 'w', encoding='utf-8') as file:
    for x in chain(wSet_1.keys(), wSet_2.keys(), wSet_3.keys(), wSet_4.keys(), wSet_5.keys()):
        for i in range(int(wSet[x]**(0.33))+1):
            file.write(x)
            file.write('§')
    for x in wSet_n.keys():
        file.write(x)
        file.write('§')