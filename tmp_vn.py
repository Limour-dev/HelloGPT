import json
from opencc import OpenCC
cc = OpenCC('t2s')  # 't2s'表示繁体转简体
from h_corpus import Fileset
import unicodedata

def fullwidth_to_halfwidth(input_str):
    return ''.join([unicodedata.normalize('NFKC', char) for char in input_str])

def clearT(s):
    s = cc.convert(fullwidth_to_halfwidth(s))
    return s

a = r'D:\datasets\v-corpus'
b = r'D:\datasets\tmp' + '\\'

# nSet = {}
with open('tmp_nSet.json', 'r', encoding='utf-8') as f:
    json_str = f.read()
nSet = json.loads(json_str)

# with open('tmp_v_33036.txt', 'r', encoding='utf-8') as f:
#     tmp_v = [x.rstrip().split('\t') for x in f]
#
# for v,k in tmp_v:
#     nSet[k] = v

# for k in nSet:
#     tmp = k.replace(' ', '')
#     tmp = tmp.replace('・', '＆')
#     tmp = tmp.replace('＋', '＆')
#     tmp = cc.convert(fullwidth_to_halfwidth(tmp))
#     nSet[k] = tmp

# for k, v in nSet.items():
#     v = v.replace('·', '＆')
#     nSet[k] = v

a = Fileset(a, ext='.tsv')
for i in range(len(a)):
    save = []
    with open(a[i], 'r', encoding='utf-8') as f:
        tmp = next(f).rstrip().split('\t')
        # # print(tmp)
        # if not('Name' in tmp) :
#         #     print(a[i])
#         # continue
        idx_n = tmp.index('Name')
        idx_d = tmp.index('Dialogue')
        idx_v = tmp.index('Voice') if 'Voice' in tmp else idx_n
        for line in f:
            tmp = line.rstrip().split('\t')
            if len(tmp) < idx_d + 1:
                print(tmp)
                continue
            n = tmp[idx_n]
            n = nSet.get(n, n).strip()
            v = tmp[idx_v].strip()
            d = tmp[idx_d].strip()
            if not d:
                print(a[i])
                break
            if n == '' and v != '':
                n = v
            n = n.replace('：', ':')
            if n == '' and v == '':
                n = '旁白'
            save.append(clearT(n) + '：' + clearT(d))
    if save:
        with open(b + f'{i}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(save))
#             break
#             # print(tmp[idx_n], tmp[idx_d])
# #             nSet[tmp[idx_n]] = nSet.get(tmp[idx_n],0) + 1
#
# json_str = json.dumps(nSet, indent=2,  ensure_ascii=False)
# with open('tmp_nSet.json', 'w', encoding='utf-8') as f:
#     f.write(json_str)
