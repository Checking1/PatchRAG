import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """
    Copied from SRSNet and reused inside CRAFT to keep the module local.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TimeStepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super(TimeStepEmbedding, self).__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        device = t.device
        emb_scale = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if emb.shape[1] < self.d_model:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return self.proj(emb[:, : self.d_model])


class PatchProjector(nn.Module):
    def __init__(
        self,
        patch_len: int,
        input_dim: int,
        d_model: int,
        dropout: float,
        max_positions: int = 5000,
    ):
        super(PatchProjector, self).__init__()
        self.patch_len = patch_len
        self.input_dim = input_dim
        self.proj = nn.Linear(patch_len * input_dim, d_model, bias=False)
        self.pos = PositionalEmbedding(d_model, max_len=max_positions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch, num_patches, patch_len, channels = patches.shape
        flat = patches.reshape(batch, num_patches, patch_len * channels)
        tokens = self.proj(flat)
        tokens = tokens + self.pos(tokens)
        return self.dropout(tokens)


def patchify_sequence(
    x: torch.Tensor,
    patch_len: int,
    stride: int,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """
    Convert [B, L, C] to [B, N, P, C] with right padding when needed.
    """
    _, seq_len, _ = x.shape
    if seq_len <= patch_len:
        pad_len = patch_len - seq_len
    else:
        remainder = (seq_len - patch_len) % stride
        pad_len = 0 if remainder == 0 else stride - remainder

    if pad_len > 0:
        x = F.pad(x.transpose(1, 2), (0, pad_len), mode=pad_mode).transpose(1, 2)

    patches = x.unfold(dimension=1, size=patch_len, step=stride)
    return patches.permute(0, 1, 3, 2).contiguous()


def patchify_future(x: torch.Tensor, patch_len: int) -> Tuple[torch.Tensor, int]:
    _, seq_len, _ = x.shape
    remainder = seq_len % patch_len
    pad_len = 0 if remainder == 0 else patch_len - remainder
    if pad_len > 0:
        x = F.pad(x.transpose(1, 2), (0, pad_len), mode="replicate").transpose(1, 2)
    patches = x.unfold(dimension=1, size=patch_len, step=patch_len)
    patches = patches.permute(0, 1, 3, 2).contiguous()
    return patches, pad_len


def unpatchify_future(patches: torch.Tensor, channels: int, pad_len: int) -> torch.Tensor:
    batch, num_patches, patch_len, _ = patches.shape
    future = patches.reshape(batch, num_patches * patch_len, channels)
    if pad_len > 0:
        future = future[:, :-pad_len, :]
    return future
