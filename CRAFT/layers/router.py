import math

import torch
from torch import nn
import torch.nn.functional as F


class FragmentRouter(nn.Module):
    """
    Slot-wise fragment selection followed by SRS-style reorder.

    The router builds a dense candidate score matrix between query slots and candidate
    fragments, selects one fragment per slot, then reorders the selected sequence with
    a second scorer. Straight-through rescaling keeps the selected path differentiable
    in the same spirit as SRSNet's select/shuffle stages.
    """

    def __init__(self, d_model: int, top_m: int, dropout: float):
        super(FragmentRouter, self).__init__()
        self.top_m = top_m
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.fragment_proj = nn.Linear(d_model, d_model, bias=False)
        self.score_scale = 1.0 / math.sqrt(d_model)
        self.reorder_scorer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def _straight_through_scale(scores: torch.Tensor) -> torch.Tensor:
        scaled_scores = scores.clone()
        non_zero_mask = scaled_scores != 0
        if non_zero_mask.any():
            inv = (1.0 / scaled_scores[non_zero_mask]).detach()
            scaled_scores[non_zero_mask] = scaled_scores[non_zero_mask] * inv
        return scaled_scores

    @staticmethod
    def _gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        view_shape = list(indices.shape) + [1] * max(values.dim() - 2, 0)
        expand_shape = list(indices.shape) + list(values.shape[2:])
        gather_index = indices.view(*view_shape).expand(*expand_shape)
        return torch.gather(values, dim=1, index=gather_index)

    def forward(
        self,
        query_tokens: torch.Tensor,
        fragment_tokens: torch.Tensor,
        fragment_patches: torch.Tensor,
        retrieval_rank: torch.Tensor,
        patch_position: torch.Tensor,
        candidate_index: torch.Tensor,
        source_memory_index: torch.Tensor,
    ) -> dict:
        batch, num_slots, _ = query_tokens.shape
        num_fragments = fragment_tokens.shape[1]
        if num_fragments == 0:
            raise ValueError("Fragment library is empty.")

        projected_query = F.normalize(self.query_proj(query_tokens), dim=-1)
        projected_fragments = F.normalize(self.fragment_proj(fragment_tokens), dim=-1)
        selection_scores = (
            torch.matmul(projected_query, projected_fragments.transpose(1, 2)) * self.score_scale
        )

        selected_scores, selection_indices = torch.max(selection_scores, dim=-1)
        selected_tokens = self._gather_candidates(fragment_tokens, selection_indices)
        selected_patches = self._gather_candidates(fragment_patches, selection_indices)
        selected_rank = self._gather_candidates(retrieval_rank, selection_indices)
        selected_patch_pos = self._gather_candidates(patch_position, selection_indices)
        selected_candidate_index = self._gather_candidates(candidate_index, selection_indices)
        selected_source_memory_index = self._gather_candidates(
            source_memory_index, selection_indices
        )

        selection_weights = self._straight_through_scale(selected_scores)
        selected_tokens = selected_tokens * selection_weights.unsqueeze(-1)
        selected_patches = selected_patches * selection_weights.unsqueeze(-1).unsqueeze(-1)

        reorder_scores = self.reorder_scorer(selected_tokens).squeeze(-1)
        reorder_indices = torch.argsort(reorder_scores, dim=-1, descending=True)
        target_slot_index = torch.arange(num_slots, device=query_tokens.device, dtype=torch.long)
        target_slot_index = target_slot_index.view(1, num_slots).expand(batch, -1)

        reordered_tokens = self._gather_candidates(selected_tokens, reorder_indices)
        reordered_patches = self._gather_candidates(selected_patches, reorder_indices)
        reordered_rank = self._gather_candidates(selected_rank, reorder_indices)
        reordered_patch_pos = self._gather_candidates(selected_patch_pos, reorder_indices)
        reordered_candidate_index = self._gather_candidates(
            selected_candidate_index, reorder_indices
        )
        reordered_source_memory_index = self._gather_candidates(
            selected_source_memory_index, reorder_indices
        )
        reordered_target_slot_index = self._gather_candidates(target_slot_index, reorder_indices)
        reordered_selected_scores = self._gather_candidates(selected_scores, reorder_indices)
        reordered_selection_weights = self._gather_candidates(selection_weights, reorder_indices)
        reordered_reorder_scores = self._gather_candidates(reorder_scores, reorder_indices)

        reorder_weights = self._straight_through_scale(reordered_reorder_scores)
        reordered_tokens = reordered_tokens * reorder_weights.unsqueeze(-1)
        reordered_patches = reordered_patches * reorder_weights.unsqueeze(-1).unsqueeze(-1)
        route_weights = reordered_selection_weights * reorder_weights

        return {
            "selection_scores": selection_scores,
            "selection_indices": selection_indices,
            "selected_scores": reordered_selected_scores,
            "reorder_scores": reordered_reorder_scores,
            "selected_tokens": reordered_tokens,
            "selected_patches": reordered_patches,
            "retrieval_rank": reordered_rank,
            "patch_position": reordered_patch_pos,
            "candidate_index": reordered_candidate_index,
            "selected_candidate_index": reordered_candidate_index,
            "source_memory_index": reordered_source_memory_index,
            "target_slot_index": reordered_target_slot_index,
            "reorder_indices": reorder_indices,
            "route_weights": route_weights,
        }
