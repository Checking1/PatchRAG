import torch
from torch import nn

from CRAFT.layers.embeddings import (
    PatchProjector,
    TimeStepEmbedding,
    patchify_future,
    unpatchify_future,
)


class DiffusionScheduler(nn.Module):
    def __init__(self, num_steps: int, beta_start: float, beta_end: float):
        super(DiffusionScheduler, self).__init__()
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        # Precompute SNR = ᾱ_t / (1 - ᾱ_t) for Min-SNR-γ loss weighting
        snr = alpha_bars / (1.0 - alpha_bars)
        self.register_buffer("snr", snr)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars.index_select(0, t).view(-1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def training_sample(self, x0: torch.Tensor):
        batch = x0.shape[0]
        t = torch.randint(0, self.num_steps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        return xt, t, noise

    def get_snr_weights(self, t: torch.Tensor, gamma: float) -> torch.Tensor:
        """Min-SNR-γ per-sample loss weights: min(SNR(t), γ) / SNR(t)."""
        snr_t = self.snr.index_select(0, t)
        return torch.clamp(snr_t, max=gamma) / snr_t

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars.index_select(0, t).view(-1, 1, 1)
        return (xt - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

    def eps_from_x0(self, xt: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
        """反推等效 eps：已知 x0_pred，供 x0-prediction 模式的 DDIM step 使用。"""
        alpha_bar_t = self.alpha_bars.index_select(0, t).view(-1, 1, 1)
        return (xt - alpha_bar_t.sqrt() * x0_pred) / (1.0 - alpha_bar_t).sqrt()

    def ddim_step(
        self, xt: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        x0 = self.predict_x0(xt, t, eps)
        alpha_bar_prev = self.alpha_bars.index_select(0, t_prev.clamp(min=0)).view(
            -1, 1, 1
        )
        x_prev = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps
        mask = (t_prev >= 0).view(-1, 1, 1)
        return torch.where(mask, x_prev, x0)


class DiffusionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super(DiffusionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        layer_idx: int = 0,
        num_layers: int = 1,
        use_rma: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # RMA: 前半段注入 fused_tokens（cond 后半段），后半段注入 query_tokens（cond 前半段）
        # cond 形状为 [B, 2S, d]，前 S 为 typed_query_tokens，后 S 为 typed_fused_tokens
        if use_rma and cond.shape[1] % 2 == 0:
            half = cond.shape[1] // 2
            if layer_idx < num_layers // 2:
                kv = cond[:, half:, :]   # 前半层：fused_tokens（检索重构信号）
            else:
                kv = cond[:, :half, :]   # 后半层：query_tokens（语义精化）
        else:
            kv = cond
        sa_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + sa_out)
        ca_out, _ = self.cross_attn(x, kv, kv)
        x = self.norm2(x + ca_out)
        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Norm conditioned on timestep embedding (DiT-style).

    Given a timestep embedding t_emb of shape [B, D], produces per-sample
    scale (gamma) and shift (beta) that modulate the standard LayerNorm
    output: AdaLN(x) = gamma * LN(x) + beta.
    """

    def __init__(self, d_model: int):
        super(AdaLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, d_model * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: [B, D] -> scale, shift: [B, 1, D]
        scale, shift = self.proj(t_emb).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class AdaLNDiffusionBlock(nn.Module):
    """DiffusionBlock with AdaLN: each sub-layer's LayerNorm is modulated by
    the timestep embedding, so every block receives fresh t information."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super(AdaLNDiffusionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = AdaLayerNorm(d_model)
        self.norm2 = AdaLayerNorm(d_model)
        self.norm3 = AdaLayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
        layer_idx: int = 0,
        num_layers: int = 1,
        use_rma: bool = False,
    ) -> torch.Tensor:
        if use_rma and cond.shape[1] % 2 == 0:
            half = cond.shape[1] // 2
            if layer_idx < num_layers // 2:
                kv = cond[:, half:, :]
            else:
                kv = cond[:, :half, :]
        else:
            kv = cond
        sa_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + sa_out, t_emb)
        ca_out, _ = self.cross_attn(x, kv, kv)
        x = self.norm2(x + ca_out, t_emb)
        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out, t_emb)


class ConditionalDiffusionDenoiser(nn.Module):
    def __init__(
        self,
        pred_len: int,
        input_dim: int,
        patch_len: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
        use_adaln: bool = False,
        use_rma: bool = False,
    ):
        super(ConditionalDiffusionDenoiser, self).__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.use_adaln = use_adaln
        self.use_rma = use_rma
        self.num_layers = num_layers
        self.patch_projector = PatchProjector(patch_len, input_dim, d_model, dropout)
        self.time_embed = TimeStepEmbedding(d_model)
        block_cls = AdaLNDiffusionBlock if use_adaln else DiffusionBlock
        self.blocks = nn.ModuleList(
            [block_cls(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(d_model, patch_len * input_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        patches, pad_len = patchify_future(x_t, self.patch_len)
        tokens = self.patch_projector(patches)
        t_emb = self.time_embed(t)
        tokens = tokens + t_emb.unsqueeze(1)
        if self.use_adaln:
            for idx, block in enumerate(self.blocks):
                tokens = block(
                    tokens, cond, t_emb,
                    layer_idx=idx, num_layers=self.num_layers, use_rma=self.use_rma,
                )
        else:
            for idx, block in enumerate(self.blocks):
                tokens = block(
                    tokens, cond,
                    layer_idx=idx, num_layers=self.num_layers, use_rma=self.use_rma,
                )
        noise_patches = self.output_proj(tokens).reshape(
            tokens.shape[0], tokens.shape[1], self.patch_len, self.input_dim
        )
        return unpatchify_future(noise_patches, self.input_dim, pad_len)
