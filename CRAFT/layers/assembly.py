import torch
from torch import nn


class FragmentAssemblyEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        max_retrieval_k: int,
        max_patch_positions: int,
    ):
        super(FragmentAssemblyEncoder, self).__init__()
        self.retrieval_rank_embed = nn.Embedding(max_retrieval_k, d_model)
        self.patch_pos_embed = nn.Embedding(max_patch_positions, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        fragment_tokens: torch.Tensor,
        retrieval_rank: torch.Tensor,
        patch_position: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        rank_tokens = self.retrieval_rank_embed(retrieval_rank.long())
        patch_tokens = self.patch_pos_embed(patch_position.long())
        fragment_states = self.encoder(fragment_tokens + rank_tokens + patch_tokens)
        cross_out, _ = self.cross_attn(query_tokens, fragment_states, fragment_states)
        return self.norm(query_tokens + cross_out)
