import torch
import torch.nn.functional as F
from torch import nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight

    def __repr__(self):
        return f'RMSNorm(hidden_size={self.hidden_size}, eps={self.eps})'


class RotaryEmbedding(nn.Module):
    def __init__(self, heads_dim: int, max_seq_len: int, device=device, theta: float = 10000.0):
        super().__init__()
        self.heads_dim = heads_dim
        self.max_seq_len = max_seq_len
        freqs = 1.0 / (theta ** (torch.arange(0, heads_dim, 2).float().to(device) / heads_dim))
        t = torch.arange(max_seq_len, device=device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # 外积
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数，模 1，角度 freqs
        self.freqs_cis.requires_grad = False  # filter(lambda p : p.requires_grad, model.parameters())
        self.local_freqs_cis = None

    def reshape_for_broadcast(self, x: torch.Tensor):
        ndim = x.ndim
        shape = (d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape))
        self.local_freqs_cis = self.local_freqs_cis.view(*shape)

    def rotary_emb(self, xq, xk):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if len(self.local_freqs_cis.shape) == 2:
            self.reshape_for_broadcast(xq_)
        xq_out = torch.view_as_real(xq_ * self.local_freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * self.local_freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, start_pos: int, seqlen: int):
        self.local_freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]  # cacheKV 相关，可忽略
        self.local_freqs_cis.requires_grad = False
        return self.rotary_emb

    def __repr__(self):
        return f'RotaryEmbedding(heads_dim={self.heads_dim}, max_seq_len={self.max_seq_len})'


class Attention(nn.Module):
    def __init__(self, hidden_size, n_heads, cacheKV, max_batch_size, max_seq_len, device=device):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.cacheKV = cacheKV
        if cacheKV:  # cacheKV 相关，可忽略
            self.cache_k = torch.zeros(max_batch_size, max_seq_len, self.n_heads, self.head_dim).to(device)
            self.cache_v = torch.zeros(max_batch_size, max_seq_len, self.n_heads, self.head_dim).to(device)

    def forward(self, hidden_states, rotary_emb, start_pos=0, mask=None, is_causal=True):
        bsz, seqlen, hidden_size = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)

        q, k = rotary_emb(q, k)

        if self.cacheKV:  # cacheKV 相关，可忽略
            self.cache_k[:bsz, start_pos: start_pos + seqlen] = k
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = v
            k = self.cache_k[:bsz, : start_pos + seqlen]
            v = self.cache_v[:bsz, : start_pos + seqlen]

        q = q.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        k = k.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        v = v.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        # print(is_causal, mask is None)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, hidden_size)
        return self.o_proj(output)


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


def getMask(seqlen, type, cacheKV, start_pos, device=device):
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    if not cacheKV:
        return mask.type(type)
    else:  # cacheKV 相关，可忽略，(seqlen, cache_len + seqlen)
        return torch.hstack([torch.zeros((seqlen, start_pos), device=device), mask]).type(type)


class HelloGPT(nn.Module):
    def __init__(self, vocab_size=32765, hidden_size=768, n_heads=12, max_seq_len=1024, n_layers=12, cacheKV=False, max_batch_size=1):
        super().__init__()
        # hidden_size > 8.33 * ln(vocab_size)
        self.cacheKV = cacheKV  # cacheKV 相关，可忽略
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
        mask = None if no_mask or seqlen <= 1 else getMask(seqlen, h.type(), self.cacheKV, start_pos)
        is_causal = no_mask and (seqlen > 1)  # 似乎 SDPA 比预计算 mask 还快
        for layer in self.layers:
            h = layer(h, rotary_emb, start_pos, mask, is_causal)

        h = self.norm(h)
        h = self.ln2(h)
        return h.float()


if __name__ == '__main__':
    from h_corpus import Hcorpus
    data = Hcorpus(r'D:\datasets\h-corpus')
    context_tokens = data()
    # context_tokens = [0,2,32764]
    input_ids = torch.tensor([context_tokens]).to(device)
    tmp = HelloGPT()
    tmp.to(device)
    tmp2 = tmp(input_ids)