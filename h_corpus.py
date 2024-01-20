import os
import gzip

def get_all_files_in_directory(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        root = root[len(directory):]
        if root.startswith('\\') or root.startswith('/'):
            root = root[1:]
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

class Fileset(list):
    def __init__(self, path, ext='', _read=None):
        if isinstance(path, str):
            self.root = path
            self.extend(f for f in get_all_files_in_directory(self.root) if f.endswith(ext))
            self._read = _read

    def __getitem__(self, index):
        if isinstance(index, int):  # index是索引
            if self._read:
                return self._read(os.path.join(self.root, super().__getitem__(index)))
            else:
                return os.path.join(self.root, super().__getitem__(index))
        else:  # index是切片
            fileset = Fileset(None)
            fileset.root = self.root
            fileset._read = self._read
            fileset.extend(super().__getitem__(index))
            return fileset


from tokenizer import tokenizer
token_eos = 2


def readOne(filePath):
    retn = []
    with gzip.open(filePath, 'rt', encoding='utf-8') if filePath.endswith('.gz') else open(filePath, encoding='utf-8') as f:
        for line in f:
            retn += tokenizer.encode(line).ids
    retn.append(token_eos)
    return retn


class Hcorpus():
    def __init__(self, path, ext='txt', fileset_idx=0, fileset_sub_idx=0):
        self.fileset = Fileset(path, ext, readOne)
        self.fileset_idx = fileset_idx
        self.fileset_sub_idx = fileset_sub_idx
        if self.fileset_sub_idx < 0:  # 再读上一个太复杂了，直接放弃
            self.fileset_sub_idx = 0
        if self.fileset_idx >= len(self.fileset):
            self.fileset_idx = 0
        self.cache = self.fileset[self.fileset_idx]
        self.fileset_idx += 1
        self.cache_idx = self.fileset_sub_idx

    def __call__(self, size=512):
        while len(self.cache) < self.cache_idx + size:
            if self.fileset_idx >= len(self.fileset):
                self.fileset_idx = 0
            self.fileset_sub_idx = self.cache_idx - len(self.cache)
            self.cache = self.cache[self.cache_idx:] + self.fileset[self.fileset_idx]
            self.cache_idx = 0
            self.fileset_idx += 1
        retn = self.cache[self.cache_idx:self.cache_idx + size]
        self.cache_idx += size
        self.fileset_sub_idx += size
        return retn

    def __repr__(self):
        return f"Hcorpus(r'{self.fileset.root}', fileset_idx={self.fileset_idx-1}, fileset_sub_idx={self.fileset_sub_idx})"

if __name__ == '__main__':
    # filePath = r'D:\datasets\h-corpus'
    # tmp = Hcorpus(filePath)
    # for i in range(10):
    #     tmp2 = tmp()
    #     tmp3 = tokenizer.decode(tmp2)
    #     print(tmp2, '\n', tmp3, '\n', len(tmp3)/len(tmp2))
    # print(tmp)
    # print(tokenizer.decode(tmp()))
    # tmp = Hcorpus(r'D:\datasets\h-corpus', fileset_idx=1, fileset_sub_idx=132)
    # print(tokenizer.decode(tmp()))
    # print(tmp)
    filePath = r'D:\datasets\h-corpus'
    tmp = Hcorpus(filePath, ext='.txt.gz')  # pigz *
    # 2..20 | ForEach-Object {cd "D:\datasets\h-corpus\h-corpus-s$($_.ToString('D2'))"; pigz *.txt}
    print(tokenizer.decode(tmp()))

