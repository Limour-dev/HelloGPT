# 检测CUDA是否安装正确并能被Pytorch检测
import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available()) # 查看CUDA是否可用
print(torch.cuda.device_count()) # 查看可用的CUDA数量
print(torch.version.cuda) # 查看CUDA的版本号

# 检测能否调用CUDA加速
a = torch.Tensor(5,3)
a = a.cuda()
print(a)