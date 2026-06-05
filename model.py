import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 256
    num_decoders: int = 6
    q_heads: int = 8
    kv_heads: int = 8
    vocab_size: int = 1024
    eps: float = 1e-5
    seq_len: int = 128
    dropout: float = 0.1

class InputEmbedding(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x)     # output.shape = (batch_size, seq_len, d_model)

class RMSNorm(nn.Module):

    # RMSNorm(x) = γ ⊙ (x / sqrt(mean(x^2) + ε))

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weights = nn.Parameter(torch.ones(config.d_model))     # weight vector is shared across all tokens, all positions, and all batches

    def forward(self, x):     # x.shape = (batch_size, seq_len, d_model)
        ms = x.pow(2).mean(dim=-1, keepdim=True)     # ms.shape = (batch_size, seq_len, 1)
        x = x * torch.rsqrt(ms + self.eps)     # x.shape = (batch_size, seq_len, d_model)
        return self.weights * x     # output.shape = (batch_size, seq_len, d_model)

class RoPE(nn.Module):

    # θ_i = 1 / (theta_base^(i / pairs))
    # φ(pos, i) = pos × θ_i

    # For each dimension pair (x_even, x_odd):
    # |x_even'|   | cos(φ)  -sin(φ) | |x_even|
    # |       | = |                 | |      |
    # |x_odd' |   | sin(φ)   cos(φ) | |x_odd |

    def __init__(self, d_head, seq_len, theta_base=10000.0):
        super().__init__()
        assert d_head % 2 == 0
        self.d_head = d_head
        self.pairs = self.d_head // 2     
        i_pairs = torch.arange(self.pairs, dtype=torch.float32)
        theta = 1.0 / (theta_base ** (i_pairs / self.pairs))     
        positions = torch.arange(seq_len, dtype=torch.float32)    
        phi = torch.outer(positions, theta)
        self.register_buffer("cos_phi", torch.cos(phi), persistent=False)
        self.register_buffer("sin_phi", torch.sin(phi), persistent=False)

    def forward(self, x):
        seq_len = x.size(2)
        cos = self.cos_phi[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_phi[:seq_len].unsqueeze(0).unsqueeze(0)
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd  = x_even * sin + x_odd * cos
        x_rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return x_rotated.flatten(-2)
    
class GQA(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.q_heads = config.q_heads
        self.kv_heads = config.kv_heads
        self.seq_len = config.seq_len
        self.dropout = config.dropout
        assert self.d_model % self.q_heads == 0
        assert self.q_heads % self.kv_heads == 0
        self.d_head = self.d_model // self.q_heads
        self.group_size = self.q_heads // self.kv_heads     # number of query heads sharing one KV head
        self.scale = self.d_head ** -0.5
        self.Wq = nn.Linear(self.d_model, self.q_heads * self.d_head, bias=False)     # Wq.weight.shape = (q_heads * d_head, d_model)
        self.Wk = nn.Linear(self.d_model, self.kv_heads * self.d_head, bias=False)
        self.Wv = nn.Linear(self.d_model, self.kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)     # Wo
        self.attn_dropout = nn.Dropout(self.dropout)     # applied to attention probabilities after softmax
        self.res_dropout = nn.Dropout(self.dropout)     # applied to the output vectors produced by the attention block
        self.rope = RoPE(self.d_head, self.seq_len)
        self.register_buffer("mask", torch.tril(torch.ones(self.seq_len, self.seq_len)))     # causal mask

    def split_heads(self, x, num_heads):

        # (batch_size, seq_len, num_heads * d_head) -> (batch_size, num_heads, seq_len, d_head)

        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, num_heads, self.d_head)
        return x.transpose(1, 2)

    def merge_heads(self, x):

        # (batch_size, q_heads, seq_len, d_head) -> (batch_size, seq_len, d_model)

        batch_size, num_heads, seq_len, d_head = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * d_head)

    def forward(self, x):

        batch_size, seq_len, d_model = x.shape

        # QKV projections
        Q = self.Wq(x)     # Q.shape = (batch_size, seq_len, q_heads * d_head)
        K = self.Wk(x)     # K.shape = (batch_size, seq_len, kv_heads * d_head)
        V = self.Wv(x)     # V.shape = (batch_size, seq_len, kv_heads * d_head)

        # split heads
        Q = self.split_heads(Q, self.q_heads)     # Q.shape = (batch_size, q_heads, seq_len, d_head)
        K = self.split_heads(K, self.kv_heads)     # K.shape = (batch_size, kv_heads, seq_len, d_head)
        V = self.split_heads(V, self.kv_heads)     # V.shape = (batch_size, kv_heads, seq_len, d_head)

        # apply RoPE
        Q = self.rope(Q)
        K = self.rope(K)

        # expand KV heads for GQA
        K = K.repeat_interleave(self.group_size, dim=1)     # K.shape = (batch_size, q_heads, seq_len, d_head)
        V = V.repeat_interleave(self.group_size, dim=1)     # V.shape = (batch_size, q_heads, seq_len, d_head)

        # attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))     # scores.shape = (batch_size, q_heads, seq_len, seq_len)
        scores = scores * self.scale

        # causal masking
        causal_mask = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # softmax
        attn = F.softmax(scores.float(), dim=-1)
        attn = attn.type_as(scores)
        attn = self.attn_dropout(attn)

        # weighted sum
        out = torch.matmul(attn, V)     # (batch_size, q_heads, seq_len, d_head)

        # merge heads
        out = self.merge_heads(out)     # (batch_size, seq_len, d_model)

        # output projection
        out = self.out_proj(out)
        out = self.res_dropout(out)

        return out
    
class SwiGLU(nn.Module):

    # output = Wdown(Wu(input) * SiLU(Wg(input)))

    def __init__(self, config):
        super().__init__()
        d_ff = int(4 * config.d_model)     # d_ff = 4 * d_model (as per original transformer)
        self.w_g = nn.Linear(config.d_model, d_ff, bias=False)     # Wg.weight.shape = (d_ff, d_model)
        self.w_u = nn.Linear(config.d_model, d_ff, bias=False)     # Wu.weight.shape = (d_ff, d_model)
        self.w_d = nn.Linear(d_ff, config.d_model, bias=False)     # Wd.weight.shape = (d_model, d_ff)

    def forward(self, x):     # x.shape = (batch_size, seq_len, d_model)
        gate = F.silu(self.w_g(x))     # gate.shape = (batch_size, seq_len, d_ff)
        up = self.w_u(x)     # up.shape = (batch_size, seq_len, d_ff)
        return self.w_d(gate * up)     # out.shape = (batch_size, seq_len, d_model)
    
class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config)
        self.attn = GQA(config)
        self.ffn_norm = RMSNorm(config)
        self.ffn = SwiGLU(config)

    def forward(self, x):     # x.shape = (batch_size, seq_len, d_model)
        h = x + self.attn(self.attn_norm(x))     # h.shape = (batch_size, seq_len, d_model)
        out = h + self.ffn(self.ffn_norm(h))     # out.shape = (batch_size, seq_len, d_model)
        return out
    
class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = InputEmbedding(config)
        self.decoders = nn.ModuleList([
            TransformerBlock(config)
            for i in range(config.num_decoders)
        ])
        self.norm = RMSNorm(config)
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )
        self.lm_head.weight = self.embedding.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, tokens):
        x = self.embedding(tokens)
        for decoder in self.decoders:
            x = decoder(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
