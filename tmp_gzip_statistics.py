import struct, os
from h_corpus import Fileset
import shutil
def get_original_file_size(gzip_file_path):
    with open(gzip_file_path, 'rb') as file:
        # 获取文件末尾的原始大小信息
        file.seek(-8, 2)  # 从文件末尾回退8个字节
        info = file.read(8)

    # 解析原文件大小信息
    crc32, original_size = struct.unpack("<II", info)

    return original_size

def get_file_size(file_path):
    # 获取文件大小
    size = os.path.getsize(file_path)
    return size

def get_gz_ratio(file_path):
    return get_file_size(file_path)/get_original_file_size(file_path)

tmp = Fileset(r'D:\datasets\h-s-corpus')
tmp = [tmp[i] for i in range(len(tmp)) if get_gz_ratio(tmp[i]) > 0.53]

for i, file_name in enumerate(tmp):
    shutil.move(file_name, r'D:\datasets\h-no')