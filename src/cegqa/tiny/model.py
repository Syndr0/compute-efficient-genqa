"""P5 - A small decoder-only Transformer (GPT-style, pre-LayerNorm) written from scratch."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    vocab_size: int = 8192
    max_seq_len: int = 512
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 scaled init for residual projections
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return {"total": total, "non_embedding": total - emb, "embedding": emb}

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, eos_id: int) -> list[int]:
        """Greedy decoding for a single sequence (shape [1, T])."""
        out = []
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.max_seq_len:])[:, -1]
            nxt = int(logits.argmax(-1))
            if nxt == eos_id:
                break
            out.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]], device=idx.device)], dim=1)
        return out
