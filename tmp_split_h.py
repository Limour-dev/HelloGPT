import os
import shutil

# 源文件夹路径
source_folder = r'D:\datasets\h-corpus'

# 目标文件夹前缀
target_folder_prefix = r'D:\datasets\h-corpus-s'

# 获取源文件夹中的所有文件列表
all_files = os.listdir(source_folder)

# 计算每个目标文件夹应包含的文件数量
files_per_folder = len(all_files) // 20

# 创建目标文件夹
for i in range(1, 21):
    target_folder = f'{target_folder_prefix}{i:02d}'
    os.makedirs(target_folder, exist_ok=True)

# 分配文件到目标文件夹
for i, file_name in enumerate(all_files):
    target_folder_index = i // files_per_folder + 1
    target_folder = f'{target_folder_prefix}{target_folder_index:02d}'
    shutil.move(os.path.join(source_folder, file_name), os.path.join(target_folder, file_name))

print("文件已平均分配到二十个文件夹中。")
