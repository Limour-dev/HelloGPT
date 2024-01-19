## 预期结构
```txt
HelloGPT(
  (tok_embeddings): Embedding(32765, 768)
  (rotary_emb): RotaryEmbedding(head_dim=64, max_seq_len=1024)
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
## 配置环境
+ [安装conda](https://limour-blog.github.io/-ji-lu--an-zhuang-conda-bing-geng-huan-qing-hua-yuan)
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
## 准备数据
+ 下载 [h-corpus-2023](https://huggingface.co/collections/Limour/r18-novels-galgame-6598f16894cadc9cdcb3f3ab) 
```python
import os

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


from tokenizer import tokenizer
token_eos = 2


def readOne(filePath):
    retn = []
    with open(file=filePath, encoding='utf-8') as f:
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
```
## 训练Tokenizer
+ [tokenizer 包的文档](https://huggingface.co/docs/tokenizers/quicktour)
+ 繁体转换成简体：[train_tokenizer_pre.py](https://github.com/Limour-dev/HelloGPT/blob/main/train_tokenizer_pre.py)
+ 获取常用 emoji：[tmp_emoji.py](https://github.com/Limour-dev/HelloGPT/blob/main/tmp_emoji.py)
+ 分词统计词频：[tokenizer_jieba.py](https://github.com/Limour-dev/HelloGPT/blob/main/train_tokenizer_jieba.py)
+ 区分词性并构造 BPE 语料：[train_tokenizer_jieba_statistics.py](https://github.com/Limour-dev/HelloGPT/blob/main/train_tokenizer_jieba_statistics.py)
+ 训练 BPE 模型：[train_tokenizer.py](https://github.com/Limour-dev/HelloGPT/blob/main/train_tokenizer.py)
+ 最终训练好的 BPE 模型：[HelloBPE.tokenizer.json](https://github.com/Limour-dev/HelloGPT/blob/main/HelloBPE.tokenizer.json)
```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("HelloBPE.tokenizer.json")
```
## 定义模型
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
#### 定义 RoPE
```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, device=device, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.set_max_seq_len(max_seq_len, device, theta)

    def set_max_seq_len(self, max_seq_len: int, device=device, theta: float = 10000.0):
        self.max_seq_len = max_seq_len
        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim))
        t = torch.arange(max_seq_len, device=device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # 外积
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数，模 1，角度 freqs
        self.freqs_cis.requires_grad = False  # filter(lambda p : p.requires_grad, model.parameters())

    def rotary_emb(self, x):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x_ * self.local_freqs_cis).flatten(3)
        return x_out.type_as(x)

    def forward(self, start_pos: int, seqlen: int):
        self.local_freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen].view(1, seqlen, 1, -1)  # cacheKV 相关，可忽略
        self.local_freqs_cis.requires_grad = False
        return self.rotary_emb
```
#### 定义 Attention
```python
class Attention(nn.Module):
    def __init__(self, hidden_size, n_heads, cacheKV, max_batch_size, max_seq_len, device=device):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, rotary_emb, start_pos=0, mask=None, is_causal=True):
        bsz, seqlen, hidden_size = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)

        q = rotary_emb(q)
        k = rotary_emb(k)

        q = q.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        k = k.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        v = v.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

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
    def __init__(self, hidden_size, n_heads, cacheKV, max_batch_size, max_seq_len):
        super().__init__()
        self.ln1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, n_heads, cacheKV, max_batch_size, max_seq_len)
        self.ln2 = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size)

    def forward(self, x, rotary_emb, start_pos, mask=None, is_causal=True):
        x = x + self.attn(self.ln1(x), rotary_emb, start_pos, mask, is_causal)
        return x + self.mlp(self.ln2(x))
```
### 组装模型
```python
class HelloGPT(nn.Module):
    def __init__(self, vocab_size=32765, hidden_size=768, n_heads=12, max_seq_len=1024, n_layers=12, cacheKV=False, max_batch_size=1):
        super().__init__()
        # hidden_size > 8.33 * ln(vocab_size)
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.rotary_emb = RotaryEmbedding(hidden_size // n_heads, max_seq_len * 2)
        self.rotary_emb.requires_grad = False
        self.layers = nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(Decoder(hidden_size, n_heads, cacheKV, max_batch_size, max_seq_len))
        self.norm = RMSNorm(hidden_size)
        self.ln2 = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, start_pos=0, no_mask=True):
        _bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)

        # 预计算，减少每一层的重复计算
        rotary_emb = self.rotary_emb(start_pos, seqlen)
        for layer in self.layers:
            h = layer(h, rotary_emb, start_pos)

        h = self.norm(h)
        h = self.ln2(h)
        return h.float()
```
## 训练模型

### 数据载入
```python
data = Hcorpus(r'D:\datasets\h-corpus')
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
## 初始化训练器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(train_parameters, lr=6e-4)  # Adam 优化器
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)  # 余弦退火学习率
torch.manual_seed(1337)  # 魔术随机种子

total_loss = 0
print_iter = 20
for epoch in range(1, 100001):
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

### 保存读取
```python
with open('tmp_training.pkl', 'rb') as file:
    epoch = pickle.load(file)  # 读取 epoch 位置
    tmp_fileset_idx = pickle.load(file)  # 读取 data 位置
    tmp_fileset_sub_idx = pickle.load(file)
# 恢复数据位置
data = Hcorpus(r'D:\datasets\h-corpus', fileset_idx=tmp_fileset_idx-1, fileset_sub_idx=tmp_fileset_sub_idx)
model = torch.load(f'tmp_model_{epoch}.pth')  # 恢复模型
print(f'start from epoch: {epoch}   data: {data}')

save_iter = 5000
for epoch in range(1, 100001):
    pass
    if epoch % save_iter == 0:
        optimizer.zero_grad(set_to_none=True)  # 清空梯度，节省显存
        with open('tmp_training.pkl', 'wb') as file:
            pickle.dump(epoch, file)  # 保存 epoch 位置
            pickle.dump(data.fileset_idx, file)  # 保存 data 位置
            pickle.dump(data.fileset_sub_idx, file)
        torch.save(model, f'tmp_model_{epoch}.pth')  # 保存模型
        print(f'save to tmp_model_{epoch}.pth')
```
### 可视化
```python
writer = SummaryWriter('logs')  # tensorboard --logdir logs
for epoch in range(1, 100001):
    pass
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    writer.add_scalar('loss', loss, epoch)
    if epoch % print_iter == 0:
        pass
        writer.add_scalar('total_loss', total_loss / print_iter, epoch)
writer.close()
```

## 附加 streaming_llm
```python
class RotaryEmbedding(nn.Module):
    pass
    def inverse_rotary_emb(self, x):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x_ * self.local_freqs_cis_inverse).flatten(3)
        return x_out.type_as(x)

    def inverse_forward(self, start_pos: int, seqlen: int):
        self.local_freqs_cis_inverse = self.freqs_cis[start_pos: start_pos + seqlen].view(1, seqlen, 1, -1)  # cacheKV 相关，可忽略
        self.local_freqs_cis_inverse = self.local_freqs_cis_inverse.conj()  # 乘上共轭就旋转回去了
        self.local_freqs_cis.requires_grad = False
        return self.inverse_rotary_emb

class Attention(nn.Module):
    pass
    def forward(self, hidden_states, rotary_emb, start_pos=0, mask=None, is_causal=True):
        pass
        if self.cacheKV:  # cacheKV 相关，可忽略
            self.cache_k[:bsz, start_pos: start_pos + seqlen] = k
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = v
            k = self.cache_k[:bsz, : start_pos + seqlen]
            v = self.cache_v[:bsz, : start_pos + seqlen]

    def streaming_llm(self, start_pos, seqlen, to_pos, inverse_rotary_emb, rotary_emb, bsz):
        k = self.cache_k[:bsz, start_pos: start_pos + seqlen]
        v = self.cache_v[:bsz, start_pos: start_pos + seqlen]
        k = inverse_rotary_emb(k)
        k = rotary_emb(k)
        self.cache_k[:bsz, to_pos: to_pos + seqlen] = k
        self.cache_v[:bsz, to_pos: to_pos + seqlen] = v

class HelloGPT(nn.Module):
    pass
    def streaming_llm(self, start_pos, seqlen, to_pos, max_batch_size=1):
        rotary_emb = self.rotary_emb(to_pos, seqlen)
        inverse_rotary_emb = self.rotary_emb.inverse_forward(start_pos, seqlen)
        for layer in self.layers:
            layer.attn.streaming_llm(start_pos, seqlen, to_pos, inverse_rotary_emb, rotary_emb, max_batch_size)
```
