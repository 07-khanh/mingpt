import torch
import math
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0, "Token embedding dimensionality must be divisible by number of heads."
    self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)

    self.register_buffer(
        "bias",
        torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1, 1, config.block_size, config.block_size),
        persistent=False
    )

    self.n_embd = config.n_embd
    self.n_head = config.n_head

  def forward(self, x):
    B, T, C = x.shape
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # (B, T, d)

    q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, H, T, d_k)
    k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

    attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))  # (B, H, T, T)
    attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    attn = self.attn_dropout(attn)
    y = attn @ v # (B, H, T, d)
    y = y.transpose(1, 2).contiguous().view(B, T, C)

    y = self.resid_dropout(self.c_proj(y))
    return y

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    x = self.c_fc(x) # (B, T, 4d)
    x = self.gelu(x)
    x = self.c_proj(x) # (B, T, d)
    x = self.dropout(x)
    return x

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50304
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768
  dropout: float = 0.1
  bias: bool = False


class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    assert config.vocab_size is not None
    assert config.block_size is not None

    self.config = config

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        drop = nn.Dropout(config.dropout),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(config.n_embd)
    ))

    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.lm_head.weight = self.transformer.wte.weight # weight tying

    self.apply(self._init_weights)

    # special init for the residual projections
    for name, param in self.named_parameters():
      if name.endswith("c_proj.weight"):
        torch.nn.init.normal_(param, mean=0.0, std=0.02 * 1.0/math.sqrt(2*config.n_layer))

    # report number of params
    print(f"Number of parameters: {self.get_num_params()}")

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def get_num_params(self):
    param_num = sum(p.numel() for p in self.parameters())
    return param_num

  def forward(self, idx, targets=None):
    device = idx.device
    _, T = idx.shape # (B, T)
    assert T <= self.config.block_size, f"Cannot forward sequence of size {T}. Max block size is only {self.config.block_size}"
    pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T)

    tok_emb = self.transformer.wte(idx) # (B, T, d)
    pos_emb = self.transformer.wpe(pos) # (B, T, d)
    x = self.transformer.drop(tok_emb+pos_emb)

    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)

    if targets is not None:
      logits = self.lm_head(x) # (B, T, V)
      loss = F.cross_entropy(
          logits.view(-1, logits.shape[-1]),
          targets.view(-1),
          ignore_index=-1
      )
    else:
      logits = self.lm_head(x[:, -1, :]) # (B, V) - only use the last pos
      loss = None

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tok, temperature=1.0, top_k=None):
    assert temperature > 0
    for _ in range(max_new_tok):
      idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
      logits, _ = self(idx_cond)
      logits = logits / temperature

      if top_k is not None:
        assert top_k >= 1
        v, _ = torch.topk(logits, min(top_k, logits.shape[-1])) # (B, k)
        logits[logits < v[:, [-1]]] = -float('Inf')

      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.cat((idx, idx_next), dim=-1)
    return idx

