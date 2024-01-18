## 配置环境

+ [相关代码 Github 地址](https://github.com/Limour-dev/HelloGPT)

```powershell
cd E:\GPT
conda install mamba -c conda-forge
mamba create -n HelloGPT pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
conda activate HelloGPT
conda install numpy transformers tiktoken tensorboard sentencepiece-python jieba emoji -c conda-forge
pip install opencc-python-reimplemented -i https://pypi.tuna.tsinghua.edu.cn/simple
python test_cuda.py
python test_SPDA.py
D:\vscode\Code.exe
```

```python
# 检测CUDA是否安装正确并能被Pytorch检测
import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available()) # 查看CUDA是否可用
print(torch.cuda.device_count()) # 查看可用的CUDA数量
print(torch.version.cuda) # 查看CUDA的版本号

# 检测能否调用CUDA加速
a = torch.Tensor(5,3)
a = a.cuda()
print(a)
```

```powershell
tree /f .\Qwen-1B8\
# E:\GPT\QWEN-1B8
#     qwen.tiktoken
#     tokenization_qwen.py
#     tokenizer_config.json
python test_tokenizer.py
# [108386, 6313]
```

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./Qwen-1B8", trust_remote_code=True)
print(tokenizer.encode('你好！'))
```

## 准备数据集

+ 下载 [h-corpus-2023](https://huggingface.co/collections/Limour/r18-novels-galgame-6598f16894cadc9cdcb3f3ab) 

+ 直接使用 Qwen1B8 的 Tokenizer

```python
import os

from transformers import AutoTokenizer


class Fileset(list):
    def __init__(self, path, ext='', _read=None):
        if isinstance(path, str):
            self.root = path
            self.extend(f for f in os.listdir(self.root) if f.endswith(ext))
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

    def getFileName(self, index):
        fname, ext = os.path.splitext(super().__getitem__(index))
        return fname


tokenizer = AutoTokenizer.from_pretrained("./Qwen-1B8", trust_remote_code=True)
token_eos = 151643


def readOne(filePath):
    retn = []
    with open(file=filePath, encoding='utf-8') as f:
        for line in f:
            retn += tokenizer.encode(line)
    retn.append(token_eos)
    return retn


class Hcorpus():
    def __init__(self, path, ext='txt'):
        self.fileset = Fileset(path, ext, readOne)
        self.fileset_idx = 0
        self.cache = []
        self.cache_idx = 0

    def __call__(self, size=512):
        while len(self.cache) < self.cache_idx + size:
            if self.fileset_idx >= len(self.fileset):
                self.fileset_idx = 0
            self.cache = self.cache[self.cache_idx:] + self.fileset[self.fileset_idx]
            self.cache_idx = 0
            self.fileset_idx += 1
        retn = self.cache[self.cache_idx:self.cache_idx + size]
        self.cache_idx += size
        return retn
```

## 定义模型

### 预期结构

```text
HelloGPT(
  (tok_embeddings): Embedding(32765, 768)
  (rotary_emb): RotaryEmbedding(heads_dim=64, max_seq_len=2048)
  (layers): ModuleList(
    (0-11): 12 x Decoder(
      (ln1): RMSNorm(hidden_size=768, eps=1e-06)
      (attn): Attention(
        (q_proj): Linear(in_features=768, out_features=768, bias=False)
        (k_proj): Linear(in_features=768, out_features=768, bias=False)
        (v_proj): Linear(in_features=768, out_features=768, bias=False)
        (o_proj): Linear(in_features=768, out_features=768, bias=False)
      )
      (ln2): RMSNorm(hidden_size=768, eps=1e-06)
      (mlp): MLP(
        (gate_proj): Linear(in_features=768, out_features=1536, bias=False)
        (up_proj): Linear(in_features=768, out_features=1536, bias=False)
        (down_proj): Linear(in_features=1536, out_features=768, bias=False)
      )
    )
  )
  (norm): RMSNorm(hidden_size=768, eps=1e-06)
  (ln2): Linear(in_features=768, out_features=32765, bias=False)
)
```

### 定义 Embedding

+ 直接使用 Qwen1B8 的 Embedding
+ 这个 Tokenizer 和 Embedding 属实有点大了，有时间还是自己弄一个小的吧

```python
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained(r"E:\ai\qwen", trust_remote_code=True).eval()
torch.save(model.base_model.wte.state_dict(), 'qwen1b8.base_model.wte.pth')
```

```python
wte = nn.Embedding(151936, 2048)
wte.load_state_dict(torch.load(r'D:\models\qwen1b8.base_model.wte.pth'))
wte.to(device)
wte.requires_grad = False # filter(lambda p: p.requires_grad, model.parameters())
if __name__ == '__main__':
    context_tokens = [108386, 6313] # tokenizer.encode('你好！')
    input_ids = torch.tensor([context_tokens]).to(device)
    inputs_embeds = wte(input_ids)
```

### 定义 Decoder

#### 定义 RMSnorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight
```

#### 定义 rotary_emb

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, heads_dim: int, max_pos: int, device=device, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, heads_dim, 2).float().to(device) / heads_dim))
        t = torch.arange(max_pos, device=device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # 外积
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数，模 1，角度 freqs

    def forward(self, start_pos: int, seqlen: int):
        local_freqs_cis =self.freqs_cis[start_pos: start_pos + seqlen]
        def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
            ndim = x.ndim
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])
            shape = (d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape))
            return freqs_cis.view(*shape)
        def rotary_emb(xq, xk, freqs_cis=local_freqs_cis):
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)
        return rotary_emb
```

#### 定义 Attention

```python
class Attention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, rotary_emb):
        bsz, seqlen, hidden_size = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)

        q, k = rotary_emb(q, k)

        q = q.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, hidden_size)
        return self.o_proj(output)
```

#### 定义 MLP

```python
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        intermediate_size = int(2 * hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        intermediate_states = self.up_proj(x)
        return self.down_proj(gate * intermediate_states)
```

#### 组装 Decoder

```python
class Decoder(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.ln1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, n_heads)
        self.ln2 = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size)

    def forward(self, x, rotary_emb):
        x = x + self.attn(self.ln1(x), rotary_emb)
        return x + self.mlp(self.ln2(x))
```

### 组装模型

```python
class HelloGPT(nn.Module):
    def __init__(self, vocab_size=16382, hidden_size=768, n_heads=12, max_seq_len=512, n_layers=12):
        super().__init__()
        # hidden_size > 8.33 * ln(vocab_size)
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.rotary_emb = RotaryEmbedding(hidden_size // n_heads, max_seq_len * 2)
        self.rotary_emb.requires_grad = False
        self.layers = nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(Decoder(hidden_size, n_heads))
        self.norm = RMSNorm(hidden_size)
        self.ln2 = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        _bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)

        # 预计算，减少每一层的重复计算
        rotary_emb = self.rotary_emb(0, seqlen)
        for layer in self.layers:
            h = layer(h, rotary_emb)

        h = self.norm(h)
        h = self.ln2(h)
        return h.float()
```

## 修改 Tokenizer

### 训练 Tokenizer

+ [tokenizer 包的文档](https://huggingface.co/docs/tokenizers/quicktour)

+ 繁体转换成简体：[train_tokenizer_pre.py](https://github.com/Limour-dev/HelloGPT/blob/main/train_tokenizer_pre.py)

+ 获取常用 emoji：[tmp_emoji.py](https://github.com/Limour-dev/HelloGPT/blob/main/tmp_emoji.py)

+ 分词统计词频：[tokenizer_jieba.py](https://github.com/Limour-dev/HelloGPT/blob/main/train_tokenizer_jieba.py)

+ 区分词性并构造 BPE 语料：[train_tokenizer_jieba_statistics.py](https://github.com/Limour-dev/HelloGPT/blob/main/train_tokenizer_jieba_statistics.py)

+ 训练 BPE 模型：[train_tokenizer.py](https://github.com/Limour-dev/HelloGPT/blob/main/train_tokenizer.py)

### 修改 Hcorpus

```python
from tokenizer import tokenizer
token_eos = 2
def readOne(filePath):
    retn = []
    with open(file=filePath, encoding='utf-8') as f:
        for line in f:
            retn += tokenizer.encode(line).ids
    retn.append(token_eos)
    return retn
```

### 修改模型

+ 修改后的 [model.py](https://github.com/Limour-dev/HelloGPT/blob/main/model.py)

## 训练模型

### 数据载入

```python
def get_batch(size=512, bsz=8):
    x = []
    y = []
    for i in range(bsz):
        tmp = data(size+1)
        x.append(tmp[:size])
        y.append(tmp[1:])
    return torch.tensor(x).to(device), torch.tensor(y).to(device)
```

### 模型载入

```python
model = HelloGPT(n_layers=8, max_seq_len=768)
model.to(device)
```

### 训练模型

```python
if ReStart:
    model = HelloGPT(n_layers=8, max_seq_len=768)  # 载入模型
    data = Hcorpus(r'D:\datasets\h-corpus')  # 载入数据
    epoch = 0  # 初始化环位置

model.to(device)
train_parameters = set(filter(lambda p: p.requires_grad, model.parameters()))  # 需要训练的参数

## 初始化训练器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(train_parameters, lr=6e-4)  # Adam 优化器
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)  # 余弦退火学习率
torch.manual_seed(1337)  # 魔术随机种子

total_loss = 0
print_iter = 5
save_iter = 20
for epoch in range(epoch + 1, 101):
    optimizer.zero_grad(set_to_none=True)  # 清空梯度，节省显存
    x, y = get_batch(size=384, bsz=4)  # x 是训练语料 y 是 x 移动了一位，当做预测目标
    y_ = model(x)  # 通过 x 预测的 y
    loss = criterion(y_.view(-1, 32765), y.view(-1))  # 计算损失
    loss.backward()  # 反向传播梯度
    torch.nn.utils.clip_grad_norm_(train_parameters, 0.5)  # 梯度裁剪，减轻过拟合
    optimizer.step()  # 通过梯度优化训练参数
    scheduler.step()  # 计算下一步的学习率
    total_loss += loss  # 累计损失

    if epoch % print_iter == 0:
        print(data)
        print(f'epoch: {epoch}  lr: {scheduler.get_last_lr()[0]:.4e} loss: {total_loss / print_iter:.4e}')
        total_loss = 0


```
