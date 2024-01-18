import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from h_corpus import Hcorpus
from model import HelloGPT
import torch.utils.benchmark as benchmark
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import gc
import pickle

def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


ReStart = False

if ReStart:  # 清空日志
    import os


    def delete_all_files(directory_path):
        # 获取目录下的所有文件
        file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        # 删除每个文件
        for file_name in file_list:
            file_path = os.path.join(directory_path, file_name)
            try:
                os.remove(file_path)
                print(f"文件 {file_name} 已删除")
            except Exception as e:
                print(f"无法删除文件 {file_name}: {e}")


    delete_all_files('logs')

writer = SummaryWriter('logs')  # tensorboard --logdir logs


# for i in range(100):
#     writer.add_scalar('test', i**0.5, i)


def get_batch(size=512, bsz=8):
    x = []
    y = []
    for i in range(bsz):
        tmp = data(size + 1)
        x.append(tmp[:size])
        y.append(tmp[1:])
    # print(data)
    return torch.tensor(x).to(device), torch.tensor(y).to(device)


# with torch.no_grad():
#     tmp, _ = get_batch(size=384, bsz=4)
#     for i in range(10):
#         print(f"The no_mask=False runs in {benchmark_torch_function_in_microseconds(model, tmp, no_mask=False):.3f} microseconds")
#         torch.cuda.empty_cache()
#         gc.collect()
#     for i in range(10):
#         print(f"The no_mask=True runs in {benchmark_torch_function_in_microseconds(model, tmp, no_mask=True):.3f} microseconds")
#         torch.cuda.empty_cache()
#         gc.collect()

if ReStart:
    model = HelloGPT(n_layers=8, max_seq_len=768)  # 载入模型
    data = Hcorpus(r'D:\datasets\h-corpus')  # 载入数据
    epoch = 0  # 初始化循环位置
else:
    with open('tmp_training.pkl', 'rb') as file:
        epoch = pickle.load(file)  # 读取 epoch 位置
        tmp_fileset_idx = pickle.load(file)  # 读取 data 位置
        tmp_fileset_sub_idx = pickle.load(file)
    # 恢复数据位置
    data = Hcorpus(r'D:\datasets\h-corpus', fileset_idx=tmp_fileset_idx-1, fileset_sub_idx=tmp_fileset_sub_idx)
    model = torch.load(f'tmp_model_{epoch}.pth')  # 恢复模型
    print(f'start from epoch: {epoch}   data: {data}')

model.to(device)
train_parameters = set(filter(lambda p: p.requires_grad, model.parameters()))  # 需要训练的参数

## 初始化训练器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(train_parameters, lr=6e-4)  # Adam 优化器
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)  # 余弦退火学习率
torch.manual_seed(1337)  # 魔术随机种子

total_loss = 0
print_iter = 20
save_iter = 5000
for epoch in range(epoch + 1, 100001):
    optimizer.zero_grad(set_to_none=True)  # 清空梯度，节省显存
    x, y = get_batch(size=384, bsz=4)  # x 是训练语料 y 是 x 移动了一位，当做预测目标
    y_ = model(x)  # 通过 x 预测的 y
    loss = criterion(y_.view(-1, 32765), y.view(-1))  # 计算损失
    loss.backward()  # 反向传播梯度
    torch.nn.utils.clip_grad_norm_(train_parameters, 0.5)  # 梯度裁剪，减轻过拟合
    optimizer.step()  # 通过梯度优化训练参数
    scheduler.step()  # 计算下一步的学习率
    total_loss += loss  # 累计损失

    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    writer.add_scalar('loss', loss, epoch)
    if epoch % print_iter == 0:
        print(data)
        print(f'epoch: {epoch}  lr: {scheduler.get_last_lr()[0]:.4e} loss: {total_loss / print_iter:.4e}')
        writer.add_scalar('total_loss', total_loss / print_iter, epoch)
        total_loss = 0

    if epoch % save_iter == 0:
        optimizer.zero_grad(set_to_none=True)  # 清空梯度，节省显存
        with open('tmp_training.pkl', 'wb') as file:
            pickle.dump(epoch, file)  # 保存 epoch 位置
            pickle.dump(data.fileset_idx, file)  # 保存 data 位置
            pickle.dump(data.fileset_sub_idx, file)
        torch.save(model, f'tmp_model_{epoch}.pth')  # 保存模型
        print(f'save to tmp_model_{epoch}.pth')

writer.close()
