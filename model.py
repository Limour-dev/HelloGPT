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

    def inverse_rotary_emb(self, x):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x_ * self.local_freqs_cis_inverse).flatten(3)
        return x_out.type_as(x)

    def inverse_forward(self, start_pos: int, seqlen: int):
        self.local_freqs_cis_inverse = self.freqs_cis[start_pos: start_pos + seqlen].view(1, seqlen, 1, -1)  # cacheKV 相关，可忽略
        self.local_freqs_cis_inverse = self.local_freqs_cis_inverse.conj()  # 乘上共轭就旋转回去了
        self.local_freqs_cis.requires_grad = False
        return self.inverse_rotary_emb

    def __repr__(self):
        return f'RotaryEmbedding(head_dim={self.head_dim}, max_seq_len={self.max_seq_len})'


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

        q = rotary_emb(q)
        k = rotary_emb(k)

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

    def streaming_llm(self, start_pos, seqlen, to_pos, inverse_rotary_emb, rotary_emb, bsz):
        k = self.cache_k[:bsz, start_pos: start_pos + seqlen]
        v = self.cache_v[:bsz, start_pos: start_pos + seqlen]
        k = inverse_rotary_emb(k)
        k = rotary_emb(k)
        self.cache_k[:bsz, to_pos: to_pos + seqlen] = k
        self.cache_v[:bsz, to_pos: to_pos + seqlen] = v


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
        self.rotary_emb = RotaryEmbedding(hidden_size // n_heads, max_seq_len)
        self.rotary_emb.requires_grad = False
        self.layers = nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(Decoder(hidden_size, n_heads, cacheKV, max_batch_size, max_seq_len))
        self.norm = RMSNorm(hidden_size)
        self.ln2 = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, start_pos=0, no_mask=True, com_mask=None):
        _bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)

        # 预计算，减少每一层的重复计算
        rotary_emb = self.rotary_emb(start_pos, seqlen)
        mask = com_mask if no_mask or seqlen <= 1 else getMask(seqlen, h.type(), self.cacheKV, start_pos)
        is_causal = no_mask and (seqlen > 1)  # 似乎 SDPA 比预计算 mask 还快
        for layer in self.layers:
            h = layer(h, rotary_emb, start_pos, mask, is_causal)

        h = self.norm(h)
        h = self.ln2(h)
        return h.float()

    def streaming_llm(self, start_pos, seqlen, to_pos, max_batch_size=1):
        rotary_emb = self.rotary_emb(to_pos, seqlen)
        inverse_rotary_emb = self.rotary_emb.inverse_forward(start_pos, seqlen)
        for layer in self.layers:
            layer.attn.streaming_llm(start_pos, seqlen, to_pos, inverse_rotary_emb, rotary_emb, max_batch_size)


if __name__ == '__main__':
    from h_corpus import Hcorpus
    data = Hcorpus(r'D:\datasets\h-corpus')
    context_tokens = data()
    # context_tokens = [0,2,32764]
    input_ids = torch.tensor([context_tokens]).to(device)
    tmp = HelloGPT()
    tmp.to(device)
    with torch.no_grad():
        tmp2 = tmp(input_ids)
        tmp3 = torch.rand(1, 10, 12, 64, device=device)
        tmp4 = RotaryEmbedding(64, 12)
        tmp7 = tmp3.float()
        tmp5 = tmp4(0, 10)
        tmp8 = tmp4.inverse_forward(0, 10)
        for i in range(1000):
            tmp6 = tmp5(tmp7)
            tmp7 = tmp8(tmp6)
            if not torch.allclose(tmp3, tmp7, atol=1e-4):
                print(i)  # 800+
                break
            print(torch.allclose(tmp3, tmp6, atol=1e-4), torch.allclose(tmp3, tmp7, atol=1e-4), torch.equal(tmp3, tmp7))