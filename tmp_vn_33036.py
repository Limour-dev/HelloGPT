def any_in(l, s: str):
    return any(x in s for x in l)
from opencc import OpenCC
cc = OpenCC('t2s')  # 't2s'表示繁体转简体
import unicodedata
def fullwidth_to_halfwidth(input_str):
    return ''.join([unicodedata.normalize('NFKC', char) for char in input_str])

def not_Dia(s: str):
    if any_in(['.', '…', 'し', 'ん', '！？', '！', '―', '—', '『', '？','好好', '一下', '一会'], s):
        return False
    elif any_in('那没谁~～，不然我就是很想危险抱着你的么。问适傍已经制服另而如此间还聊请确您别在这用嘴把遇到说道出去、所以为对摇无会打疼让摸答被得掉迎解握害', s):
        return False
    elif any_in(['羡慕', '谢谢', '刚刚', '等等'], s):
        return False
    elif any_in('噫叹哐呸啾噗咳咚哒嘛哟烦哼唉咔嘿呦喂哇嗯啊呢呀哎啥唔呜哈呃嗒哔哦吗吧诶了哪个呼咬啃嚼吃过糊', s):
        return False
    elif any_in('0123456789', s):
        return False
    elif any_in('左上前东南今明昨点安回良', s) and any_in('来右下后西北早晚天有好', s) :
        return False
    elif len(s.strip()) == 1 and any_in('有早中晚好切', s):
        return False
    else:
        return True


n = []
# d = []
tmp_a = []
tmp_b = ''
with open(r'D:\datasets\v-corpus\v33036_zh.tsv', 'r', encoding='utf-8') as f:
    tmp = next(f).rstrip().split('\t')
    idx_n = tmp.index('Name')
    idx_v = tmp.index('Voice')
    idx_d = tmp.index('Dialogue')
    for line in f:
        tmp = line.rstrip().split('\t')
        if not tmp[idx_v].startswith('v'):
            tmp_b = tmp[idx_d]
            continue
        if tmp[idx_n] not in n:
            tmp_a.append(tmp_b + '\t')
            n.append(tmp[idx_n])
            tmp_a.append(tmp[idx_n] + '\n')
        # if not tmp[idx_v].startswith('v') and not_Dia(tmp[idx_d]):
        #     tmp = tmp[idx_d]
        #     tmp = cc.convert(fullwidth_to_halfwidth(tmp))
        #     if tmp not in d:
        #         d.append(tmp)

with open('tmp_v_33036.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(tmp_a))
