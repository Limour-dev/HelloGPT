from tqdm import tqdm
import gzip
from h_corpus import Fileset

filePath = Fileset(r'D:\datasets\b-corpus\科幻小说\华语科幻', ext='.txt.gz')
for path in tqdm((filePath[i] for i in range(len(filePath))), desc="Converting"):
    try:
        with gzip.open(path, 'rt') as f:
            tmp = f.read()
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(tmp)
    except:
        pass