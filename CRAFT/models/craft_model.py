from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from CRAFT.layers.assembly import FragmentAssemblyEncoder
from CRAFT.layers.diffusion import (
    ConditionalDiffusionDenoiser,
    DiffusionScheduler,
)
from CRAFT.layers.embeddings import PatchProjector, patchify_sequence
from CRAFT.layers.router import FragmentRouter
from CRAFT.utils.retrieval_cache import (
    batch_window_hashes,
    build_dtw_views,
    compute_top_dtw_shortlist,
    extract_target_sequence,
)


def _compute_num_patches(seq_len: int, patch_len: int, stride: int) -> int:
    if seq_len <= patch_len:
        return 1
    remainder = (seq_len - patch_len) % stride
    pad_len = 0 if remainder == 0 else stride - remainder
    return ((seq_len + pad_len - patch_len) // stride) + 1


class QueryConditionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        patch_len: int,
        stride: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
        max_positions: int = 5000,
    ):
        super(QueryConditionEncoder, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projector = PatchProjector(
            patch_len, input_dim, d_model, dropout, max_positions=max_positions
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor):
        patches = patchify_sequence(x, self.patch_len, self.stride)
        tokens = self.projector(patches)
        tokens = self.encoder(tokens)
        global_repr = tokens.mean(dim=1)
        return global_repr, tokens


class FragmentEncoder(nn.Module):
    def __init__(
        self,
        patch_len: int,
        input_dim: int,
        d_model: int,
        dropout: float,
        max_positions: int,
    ):
        super(FragmentEncoder, self).__init__()
        self.projector = PatchProjector(
            patch_len, input_dim, d_model, dropout, max_positions=max_positions
        )

    def forward(self, fragment_patches: torch.Tensor) -> torch.Tensor:
        batch, num_fragments, patch_len, channels = fragment_patches.shape
        return self.projector(
            fragment_patches.reshape(batch, num_fragments, patch_len, channels)
        )


class CRAFTModel(nn.Module):
    def __init__(self, config):
        super(CRAFTModel, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.input_dim = config.enc_in
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.memory_fragment_stride = getattr(config, "memory_fragment_stride", 1)
        self.top_k = config.top_k
        self.top_m = config.top_m
        self.coarse_recall_k = max(
            getattr(config, "coarse_recall_k", max(config.top_k, 16)), config.top_k
        )
        self.dtw_window = getattr(config, "dtw_window", None)
        self.dtw_downsample_stride = max(
            1, getattr(config, "dtw_downsample_stride", self.patch_len)
        )
        self.reranker_hidden_dim = getattr(
            config, "reranker_hidden_dim", config.d_model
        )
        self.reranker_loss_weight = getattr(config, "reranker_loss_weight", 0.1)
        self.snr_gamma = getattr(config, "snr_gamma", None)
        self.x0_loss_weight = float(getattr(config, "x0_loss_weight", 0.0))
        self.prediction_target = getattr(config, "prediction_target", "eps")
        self.reranker_temperature = float(
            getattr(config, "reranker_temperature", 1.0)
        )
        self.reranker_target_temperature = float(
            getattr(config, "reranker_target_temperature", 1.0)
        )
        self.use_predictive_reranker = bool(
            getattr(config, "use_predictive_reranker", False)
        )
        self.use_reranker_fragment_weight = bool(
            getattr(config, "use_reranker_fragment_weight", False)
        )
        self.dtw_normalize_ts = bool(getattr(config, "dtw_normalize_ts", False))
        self.dtw_target_channel_index = int(
            getattr(config, "dtw_target_channel_index", 0)
        )
        self.num_query_patches = _compute_num_patches(
            self.seq_len, self.patch_len, self.stride
        )
        self.num_memory_fragments = _compute_num_patches(
            self.seq_len, self.patch_len, self.memory_fragment_stride
        )
        self.max_patch_positions = max(self.num_memory_fragments + 2, 2)
        self.max_fragment_tokens = max(self.top_k * self.num_memory_fragments + 2, 5000)

        self.query_encoder = QueryConditionEncoder(
            input_dim=self.input_dim,
            patch_len=config.patch_len,
            stride=config.stride,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            num_layers=config.num_query_layers,
            dropout=config.dropout,
        )
        self.fragment_encoder = FragmentEncoder(
            patch_len=config.patch_len,
            input_dim=self.input_dim,
            d_model=config.d_model,
            dropout=config.dropout,
            max_positions=self.max_fragment_tokens,
        )
        self.fragment_router = FragmentRouter(
            d_model=config.d_model,
            top_m=config.top_m,
            dropout=config.dropout,
        )
        self.assembly_encoder = FragmentAssemblyEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            num_layers=config.num_assembly_layers,
            dropout=config.dropout,
            max_retrieval_k=max(config.top_k, 2),
            max_patch_positions=self.max_patch_positions,
        )
        self.reranker = nn.Sequential(
            nn.Linear(config.d_model * 4 + 1, self.reranker_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.reranker_hidden_dim, 1),
        )
        self.use_retrieval_quality_gate = bool(
            getattr(config, "use_retrieval_quality_gate", False)
        )
        if self.use_retrieval_quality_gate:
            # Quality gate: per-slot sigmoid gate that measures retrieval credibility.
            # Input: [query_tok, reconstructed_tok, |query - reconstructed|] → scalar gate
            # Bias initialized to 0 → sigmoid(0)=0.5, preserving retrieval signal at init.
            self.quality_head = nn.Sequential(
                nn.Linear(config.d_model * 3, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, 1),
            )
            # Initialize final layer bias to 0 for sigmoid(0)=0.5 at training start
            nn.init.zeros_(self.quality_head[-1].bias)
            nn.init.zeros_(self.quality_head[-1].weight)
        else:
            self.fusion_alpha = nn.Parameter(
                torch.ones(self.num_query_patches, config.d_model) * 2.0
            )
        self.token_type_embed = nn.Embedding(2, config.d_model)
        self.scheduler = DiffusionScheduler(
            num_steps=config.num_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
        self.use_adaln = bool(getattr(config, "use_adaln", False))
        self.use_rma = bool(getattr(config, "use_rma", False))
        self.denoiser = ConditionalDiffusionDenoiser(
            pred_len=config.pred_len,
            input_dim=self.input_dim,
            patch_len=config.patch_len,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            num_layers=config.num_diffusion_layers,
            dropout=config.dropout,
            use_adaln=self.use_adaln,
            use_rma=self.use_rma,
        )
        self.ddim_steps = config.ddim_steps
        self.memory_bank_contexts = None
        self.memory_bank_keys = None
        self.memory_bank_future_targets = None
        self.memory_bank_context_hashes: List[str] = []
        self.memory_hash_to_index: Dict[str, int] = {}
        self.dtw_cache_payload: Dict[str, Any] = {}
        self.dtw_cache_path: Optional[str] = None
        self.dtw_bank_views: Optional[np.ndarray] = None
        self.last_retrieval_debug: Optional[Dict[str, Any]] = None
        self.max_memory_chunks = config.max_memory_chunks

    def set_memory_bank(
        self,
        contexts: torch.Tensor,
        future_targets: Optional[torch.Tensor] = None,
        context_hashes: Optional[List[str]] = None,
        dtw_cache_payload: Optional[Dict[str, Any]] = None,
        dtw_cache_path: Optional[str] = None,
    ):
        self.memory_bank_contexts = contexts
        self.memory_bank_keys = self._encode_memory_bank(contexts)
        self.memory_bank_future_targets = (
            future_targets.detach().cpu() if future_targets is not None else None
        )
        if context_hashes is None:
            context_hashes = batch_window_hashes(contexts)
        self.memory_bank_context_hashes = list(context_hashes)
        self.memory_hash_to_index = {
            context_hash: index
            for index, context_hash in enumerate(self.memory_bank_context_hashes)
        }
        self.dtw_cache_payload = dtw_cache_payload or {}
        self.dtw_cache_payload.setdefault("entries", {})
        self.dtw_cache_payload.setdefault(
            "context_hashes", list(self.memory_bank_context_hashes)
        )
        dtw_views = self.dtw_cache_payload.get("dtw_views")
        if isinstance(dtw_views, torch.Tensor):
            dtw_views = dtw_views.detach().cpu().numpy()
        if dtw_views is None:
            dtw_views = build_dtw_views(
                contexts.detach().cpu(),
                self.dtw_downsample_stride,
                target_channel_index=self.dtw_target_channel_index,
            )
            self.dtw_cache_payload["dtw_views"] = dtw_views
        self.dtw_bank_views = np.asarray(dtw_views, dtype=np.float32)
        self.dtw_cache_path = dtw_cache_path
        self.last_retrieval_debug = None

    def get_last_retrieval_debug(self) -> Optional[Dict[str, Any]]:
        return self.last_retrieval_debug

    def _encode_memory_bank(self, contexts: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        keys = []
        with torch.no_grad():
            for start in range(0, contexts.shape[0], self.max_memory_chunks):
                chunk = contexts[start : start + self.max_memory_chunks]
                chunk_keys, _ = self.query_encoder(chunk)
                keys.append(F.normalize(chunk_keys, dim=-1))
        if was_training:
            self.train()
        return torch.cat(keys, dim=0)

    def _gather_memory_tensor(
        self, memory_tensor: torch.Tensor, retrieval_indices: torch.Tensor
    ) -> torch.Tensor:
        expanded_indices = retrieval_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            memory_tensor.shape[-2],
            memory_tensor.shape[-1],
        )
        memory_expand = memory_tensor.unsqueeze(0).expand(
            retrieval_indices.shape[0], -1, -1, -1
        )
        return torch.gather(memory_expand, dim=1, index=expanded_indices)

    def _gather_cpu_memory_tensor(
        self,
        memory_tensor: torch.Tensor,
        retrieval_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        cpu_indices = retrieval_indices.detach().cpu()
        gathered = memory_tensor[cpu_indices]
        return gathered.to(device)

    def _lookup_cached_shortlist(
        self, query_hash: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        entry = self.dtw_cache_payload.get("entries", {}).get(query_hash)
        if entry is None:
            return None, None
        indices = np.asarray(entry["recall_indices"], dtype=np.int64)
        distances = np.asarray(entry["recall_distances"], dtype=np.float32)
        return indices, distances

    def _compute_dtw_shortlist(
        self, query_context: torch.Tensor, query_hashes: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[int]]]:
        if self.dtw_bank_views is None:
            raise ValueError("DTW cache views are not prepared.")

        recall_indices = []
        recall_distances = []
        source_indices: List[Optional[int]] = []
        recall_k = max(1, min(self.coarse_recall_k, self.dtw_bank_views.shape[0]))

        for batch_index, query_hash in enumerate(query_hashes):
            memory_index = self.memory_hash_to_index.get(query_hash)
            source_indices.append(memory_index)
            cached_indices, cached_distances = self._lookup_cached_shortlist(query_hash)
            if cached_indices is None or cached_distances is None:
                query_view = extract_target_sequence(
                    query_context[batch_index].detach().cpu().numpy(),
                    self.dtw_target_channel_index,
                )
                exclude_index = (
                    memory_index
                    if memory_index is not None and self.dtw_bank_views.shape[0] > 1
                    else None
                )
                cached_indices, cached_distances = compute_top_dtw_shortlist(
                    query_view,
                    self.dtw_bank_views,
                    top_k=recall_k,
                    window=self.dtw_window,
                    exclude_index=exclude_index,
                    normalize_ts=self.dtw_normalize_ts,
                )
                self.dtw_cache_payload.setdefault("entries", {})[query_hash] = {
                    "memory_index": memory_index,
                    "recall_indices": cached_indices.tolist(),
                    "recall_distances": cached_distances.tolist(),
                }

            recall_indices.append(cached_indices)
            recall_distances.append(cached_distances)

        recall_indices_tensor = torch.as_tensor(
            np.stack(recall_indices, axis=0),
            device=query_context.device,
            dtype=torch.long,
        )
        recall_distances_tensor = torch.as_tensor(
            np.stack(recall_distances, axis=0),
            device=query_context.device,
            dtype=query_context.dtype,
        )
        return recall_indices_tensor, recall_distances_tensor, source_indices

    def _rerank_shortlist(
        self,
        query_global: torch.Tensor,
        recall_indices: torch.Tensor,
        recall_distances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_keys = self.memory_bank_keys[recall_indices]
        query_expand = query_global.unsqueeze(1).expand_as(candidate_keys)
        normalized_distances = recall_distances / (
            recall_distances.mean(dim=-1, keepdim=True) + 1e-6
        )
        features = torch.cat(
            [
                query_expand,
                candidate_keys,
                torch.abs(query_expand - candidate_keys),
                query_expand * candidate_keys,
                (-normalized_distances).unsqueeze(-1),
            ],
            dim=-1,
        )
        reranker_scores = self.reranker(features).squeeze(-1)
        top_k = min(self.top_k, reranker_scores.shape[-1])
        top_scores, top_positions = torch.topk(reranker_scores, k=top_k, dim=-1)
        top_indices = torch.gather(recall_indices, dim=1, index=top_positions)
        return reranker_scores, top_indices, top_scores

    def _build_candidate_condition(
        self, candidate_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Build a lightweight condition tensor from a single candidate context
        for use in predictive reranker utility estimation.

        The output shape must match what prepare_condition returns:
          - quality_gate mode:  [B, S, d]  (residual injection; gate=1 since no
                                            retrieval discrepancy for a standalone ctx)
          - legacy mode:        [B, 2*S, d] (concat of typed_query + typed_fused)

        In quality_gate mode the candidate itself acts as both query and
        reconstructed source, so the retrieval delta is zero and the condition
        reduces to the candidate's own token representation.
        """
        _, cand_tokens = self.query_encoder(candidate_context)
        # Both quality-gate and legacy modes produce [B, 2S, d] condition.
        # For a standalone candidate context, delta=0 so fused=query regardless of gate.
        typed_q = cand_tokens + self.token_type_embed.weight[0].view(1, 1, -1)
        typed_f = cand_tokens + self.token_type_embed.weight[1].view(1, 1, -1)
        return torch.cat([typed_q, typed_f], dim=1)  # [B, 2*S, d]

    def _compute_reranker_loss(
        self,
        future: Optional[torch.Tensor],
        recall_indices: torch.Tensor,
        reranker_scores: torch.Tensor,
        x_t: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if future is None or self.memory_bank_future_targets is None:
            return None

        if self.use_predictive_reranker and x_t is not None and self.training:
            # ── Predictive utility: measure denoiser prediction quality for
            #    each recall candidate individually, reusing the same (x_t, t, noise)
            #    drawn from the current future. This directly measures "how useful
            #    is this context for the denoiser" rather than "how similar is its future".
            # Only run during training — validation uses fast proxy utility instead,
            # since we only need a stable early-stopping signal, not exact reranker loss.
            R = recall_indices.shape[1]
            utilities = []
            for i in range(R):
                # gather the i-th candidate context for every item in the batch
                # recall_indices[:, i]: [B]
                cand_ctx = self._gather_cpu_memory_tensor(
                    self.memory_bank_contexts,
                    recall_indices[:, i].unsqueeze(1),  # [B, 1]
                    x_t.device,
                ).squeeze(1)  # [B, L, C]

                cand_condition = self._build_candidate_condition(cand_ctx)  # [B, 2S, d]

                with torch.no_grad():
                    # denoiser is called without grad here: we only want utility
                    # values as target labels, not to back-prop through them.
                    # Gradients flow through reranker_scores → log_probs only.
                    pred_eps_i = self.denoiser(x_t, t, cand_condition)

                # per-sample diffusion loss for this candidate: [B]
                u_i = -F.mse_loss(pred_eps_i, noise, reduction="none").mean(dim=(1, 2))
                utilities.append(u_i)

            utility = torch.stack(utilities, dim=1)  # [B, R]
        else:
            # ── Proxy utility: negative MSE between candidate stored future and
            #    current true future. Fast, no extra forward pass.
            candidate_futures = self._gather_cpu_memory_tensor(
                self.memory_bank_future_targets,
                recall_indices,
                reranker_scores.device,
            )
            utility = -((candidate_futures - future.unsqueeze(1)) ** 2).mean(dim=(2, 3))

        target_probs = torch.softmax(
            utility / max(self.reranker_target_temperature, 1e-6), dim=-1
        )
        log_probs = torch.log_softmax(
            reranker_scores / max(self.reranker_temperature, 1e-6), dim=-1
        )
        return -(target_probs * log_probs).sum(dim=-1).mean()

    def _build_retrieval_debug(
        self,
        query_context: torch.Tensor,
        query_hashes: List[str],
        dtw_recall_indices: torch.Tensor,
        dtw_recall_distances: torch.Tensor,
        reranker_scores: torch.Tensor,
        retrieval_indices: torch.Tensor,
        retrieval_scores: torch.Tensor,
        retrieved_contexts: torch.Tensor,
    ) -> Dict[str, Any]:
        return {
            "query_hashes": list(query_hashes),
            "query_context": query_context.detach().cpu().numpy(),
            "dtw_recall_indices": dtw_recall_indices.detach().cpu().numpy(),
            "dtw_recall_distances": dtw_recall_distances.detach().cpu().numpy(),
            "reranker_scores": reranker_scores.detach().cpu().numpy(),
            "reranker_topk_indices": retrieval_indices.detach().cpu().numpy(),
            "reranker_topk_scores": retrieval_scores.detach().cpu().numpy(),
            "selected_contexts": retrieved_contexts.detach().cpu().numpy(),
        }

    def _retrieve(
        self,
        query_context: torch.Tensor,
        future: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ):
        if self.memory_bank_keys is None or self.memory_bank_contexts is None:
            raise ValueError("Memory bank has not been prepared.")

        query_global, query_tokens = self.query_encoder(query_context)
        query_global = F.normalize(query_global, dim=-1)
        query_hashes = batch_window_hashes(query_context)
        dtw_recall_indices, dtw_recall_distances, _ = self._compute_dtw_shortlist(
            query_context, query_hashes
        )
        reranker_scores, retrieval_indices, retrieval_scores = self._rerank_shortlist(
            query_global, dtw_recall_indices, dtw_recall_distances
        )
        retrieved_contexts = self._gather_memory_tensor(
            self.memory_bank_contexts, retrieval_indices
        )
        reranker_loss = self._compute_reranker_loss(
            future, dtw_recall_indices, reranker_scores,
            x_t=x_t, t=t, noise=noise,
        )
        retrieval_debug = self._build_retrieval_debug(
            query_context=query_context,
            query_hashes=query_hashes,
            dtw_recall_indices=dtw_recall_indices,
            dtw_recall_distances=dtw_recall_distances,
            reranker_scores=reranker_scores,
            retrieval_indices=retrieval_indices,
            retrieval_scores=retrieval_scores,
            retrieved_contexts=retrieved_contexts,
        )
        retrieval_debug["reranker_loss"] = (
            None
            if reranker_loss is None
            else float(reranker_loss.detach().cpu().item())
        )
        self.last_retrieval_debug = retrieval_debug
        return {
            "query_global": query_global,
            "query_tokens": query_tokens,
            "retrieval_indices": retrieval_indices,
            "retrieved_contexts": retrieved_contexts,
            "dtw_recall_indices": dtw_recall_indices,
            "dtw_recall_distances": dtw_recall_distances,
            "reranker_scores": reranker_scores,
            "reranker_topk_scores": retrieval_scores,
            "reranker_loss": reranker_loss,
        }

    def _build_fragment_library(
        self, retrieved_contexts: torch.Tensor, retrieval_indices: torch.Tensor
    ):
        batch, top_k, _, channels = retrieved_contexts.shape
        flat_contexts = retrieved_contexts.reshape(batch * top_k, self.seq_len, channels)
        fragment_patches = patchify_sequence(
            flat_contexts, self.patch_len, self.memory_fragment_stride
        )
        num_fragments = fragment_patches.shape[1]
        fragment_patches = fragment_patches.reshape(
            batch, top_k * num_fragments, self.patch_len, channels
        )

        retrieval_rank = torch.arange(
            top_k, device=retrieved_contexts.device, dtype=torch.long
        ).view(1, top_k, 1)
        retrieval_rank = retrieval_rank.expand(batch, top_k, num_fragments).reshape(
            batch, top_k * num_fragments
        )

        patch_position = torch.arange(
            num_fragments, device=retrieved_contexts.device, dtype=torch.long
        ).view(1, 1, num_fragments)
        patch_position = patch_position.expand(batch, top_k, num_fragments).reshape(
            batch, top_k * num_fragments
        )

        candidate_index = torch.arange(
            top_k * num_fragments, device=retrieved_contexts.device, dtype=torch.long
        ).view(1, -1).expand(batch, -1)

        source_memory_index = retrieval_indices.unsqueeze(-1).expand(
            -1, -1, num_fragments
        ).reshape(batch, top_k * num_fragments)

        return {
            "fragment_patches": fragment_patches,
            "retrieval_rank": retrieval_rank,
            "patch_position": patch_position,
            "candidate_index": candidate_index,
            "source_memory_index": source_memory_index,
        }

    def _fuse_views(
        self, query_tokens: torch.Tensor, reconstructed_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.fusion_alpha[: query_tokens.shape[1]]).unsqueeze(0)
        fused_tokens = gate * query_tokens + (1.0 - gate) * reconstructed_tokens
        return fused_tokens, gate

    def _compute_quality_gate(
        self, query_tokens: torch.Tensor, reconstructed_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a per-slot quality gate in [0, 1] that measures how credible
        the retrieved reconstruction is for each query slot.

        High gate (→1): reconstructed is close to / coherent with query,
                        retrieval delta is trustworthy → inject it fully.
        Low gate (→0):  large discrepancy between query and reconstructed,
                        retrieval may have failed → suppress the delta.

        Input features per slot: [query, reconstructed, |query - reconstructed|]
        Initialized at gate=0.5 (zero weight + zero bias in final Linear).
        """
        diff = (query_tokens - reconstructed_tokens).abs()
        features = torch.cat([query_tokens, reconstructed_tokens, diff], dim=-1)
        return torch.sigmoid(self.quality_head(features))  # [B, S, 1]

    def prepare_condition(
        self,
        query_context: torch.Tensor,
        future: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> dict:
        retrieval_output = self._retrieve(
            query_context, future=future, x_t=x_t, t=t, noise=noise
        )
        query_global = retrieval_output["query_global"]
        query_tokens = retrieval_output["query_tokens"]
        retrieval_indices = retrieval_output["retrieval_indices"]
        retrieved_contexts = retrieval_output["retrieved_contexts"]
        fragment_library = self._build_fragment_library(retrieved_contexts, retrieval_indices)
        fragment_tokens = self.fragment_encoder(fragment_library["fragment_patches"])

        if self.use_reranker_fragment_weight and self.training:
            # ── Inject reranker context scores as per-fragment multiplicative weights.
            # This opens a direct gradient path:
            #   diffusion_loss → denoiser → condition → fragment_tokens
            #   → fragment_weights → reranker_scores → reranker parameters
            #
            # Shape accounting:
            #   reranker_topk_scores: [B, top_k]   (raw logits, not softmax)
            #   fragment_tokens:      [B, top_k * P, d_model]  P = num_fragments per context
            #
            # We use softmax-and-scale so that the average weight is 1.0:
            #   w_i = softmax(s_i) * top_k
            # High-ranked contexts get weight > 1, low-ranked < 1, preserving
            # total token energy while creating a learnable soft selection.
            top_scores = retrieval_output["reranker_topk_scores"]  # [B, top_k]
            top_k_actual = top_scores.shape[1]
            # Per-context soft weight, mean = 1
            ctx_weights = torch.softmax(top_scores, dim=-1) * top_k_actual  # [B, top_k]
            # Expand to per-fragment: each context owns P consecutive fragments
            P = fragment_tokens.shape[1] // top_k_actual
            # [B, top_k] → [B, top_k, 1] → [B, top_k, P] → [B, top_k * P]
            frag_weights = ctx_weights.unsqueeze(-1).expand(-1, -1, P).reshape(
                fragment_tokens.shape[0], top_k_actual * P
            )
            # Multiply: [B, top_k*P, d_model] * [B, top_k*P, 1]
            fragment_tokens = fragment_tokens * frag_weights.unsqueeze(-1)

        router_output = self.fragment_router(
            query_tokens=query_tokens,
            fragment_tokens=fragment_tokens,
            fragment_patches=fragment_library["fragment_patches"],
            retrieval_rank=fragment_library["retrieval_rank"],
            patch_position=fragment_library["patch_position"],
            candidate_index=fragment_library["candidate_index"],
            source_memory_index=fragment_library["source_memory_index"],
        )
        reconstructed_tokens = self.assembly_encoder(
            fragment_tokens=router_output["selected_tokens"],
            retrieval_rank=router_output["retrieval_rank"],
            patch_position=router_output["patch_position"],
            query_tokens=query_tokens,
        )

        if self.use_retrieval_quality_gate:
            # ── Quality Gating: content-aware fusion replacing fixed fusion_alpha.
            #
            # The gate measures per-slot retrieval credibility:
            #   retrieval_delta = reconstructed - query           # what retrieval adds
            #   quality_gate    = σ(MLP([q, r, |q-r|]))          # [B, S, 1], ∈ (0,1)
            #   fused           = query + gate * retrieval_delta  # [B, S, d]
            #
            # Key design choice: condition shape stays [B, 2S, d] (same as legacy),
            # so denoiser retains full expressiveness of both query and fused views.
            # Only the fusion mechanism changes: fixed per-slot alpha → input-adaptive gate.
            #
            # Gate semantics:
            #   high gate (→1): retrieval is coherent, inject delta fully → fused≈reconstructed
            #   low  gate (→0): retrieval diverges from query → fused degrades to pure query
            retrieval_delta = reconstructed_tokens - query_tokens  # [B, S, d]
            quality_gate = self._compute_quality_gate(
                query_tokens, reconstructed_tokens
            )  # [B, S, 1]
            fused_tokens = query_tokens + quality_gate * retrieval_delta  # [B, S, d]
            fusion_gate = quality_gate.squeeze(-1)  # [B, S] for debug
        else:
            fused_tokens, fusion_gate = self._fuse_views(query_tokens, reconstructed_tokens)

        typed_query_tokens = query_tokens + self.token_type_embed.weight[0].view(1, 1, -1)
        typed_fused_tokens = fused_tokens + self.token_type_embed.weight[1].view(1, 1, -1)
        condition = torch.cat([typed_query_tokens, typed_fused_tokens], dim=1)  # [B, 2S, d]

        return {
            "condition": condition,
            "query_global": query_global,
            "query_tokens": query_tokens,
            "reconstructed_tokens": reconstructed_tokens,
            "fused_tokens": fused_tokens,
            "fusion_gate": fusion_gate,
            "router": router_output,
            "retrieval_indices": retrieval_indices,
            "fragment_library": fragment_library,
            "retrieval": retrieval_output,
        }

    def compute_loss(self, query_context: torch.Tensor, future: torch.Tensor) -> dict:
        # Sample (x_t, t, noise) first so they can be reused by the predictive
        # reranker utility computation inside _retrieve → _compute_reranker_loss.
        x_t, t, noise = self.scheduler.training_sample(future)
        condition_dict = self.prepare_condition(
            query_context, future=future, x_t=x_t, t=t, noise=noise
        )
        condition = condition_dict["condition"]

        if self.prediction_target == "x0":
            # ── x0-prediction 模式 ────────────────────────────────────
            # denoiser 直接预测干净序列 x0（即 future），损失在原始空间计算。
            # snr_gamma 加权仍可使用：高噪声步 SNR 低，权重小，让模型专注低噪声步。
            # x0_loss_weight 在此模式下无意义（主损失已经是 x0 MSE），保持为零忽略。
            pred_x0 = self.denoiser(x_t, t, condition)
            if self.snr_gamma is not None:
                per_sample_mse = F.mse_loss(
                    pred_x0, future, reduction="none"
                ).mean(dim=(1, 2))
                snr_weights = self.scheduler.get_snr_weights(t, self.snr_gamma)
                diffusion_loss = (snr_weights * per_sample_mse).mean()
            else:
                diffusion_loss = F.mse_loss(pred_x0, future)
            x0_loss = diffusion_loss.new_zeros(())
            pred_out = pred_x0
        else:
            # ── ε-prediction 模式（默认，向后兼容）────────────────────
            pred_noise = self.denoiser(x_t, t, condition)
            if self.snr_gamma is not None:
                per_sample_mse = F.mse_loss(
                    pred_noise, noise, reduction="none"
                ).mean(dim=(1, 2))
                snr_weights = self.scheduler.get_snr_weights(t, self.snr_gamma)
                diffusion_loss = (snr_weights * per_sample_mse).mean()
            else:
                diffusion_loss = F.mse_loss(pred_noise, noise)
            x0_loss = diffusion_loss.new_zeros(())
            if self.x0_loss_weight > 0:
                x0_pred = self.scheduler.predict_x0(x_t, t, pred_noise)
                x0_loss = F.mse_loss(x0_pred, future)
            pred_out = pred_noise

        reranker_loss = condition_dict["retrieval"].get("reranker_loss")
        if reranker_loss is None:
            reranker_loss = diffusion_loss.new_zeros(())
        loss = (
            diffusion_loss
            + self.x0_loss_weight * x0_loss
            + self.reranker_loss_weight * reranker_loss
        )
        return {
            "loss": loss,
            "diffusion_loss": diffusion_loss,
            "x0_loss": x0_loss,
            "reranker_loss": reranker_loss,
            "pred_noise": pred_out,
            "noise": noise,
            "condition": condition,
        }

    def _ddim_loop(
        self, condition: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Run a single DDIM sampling chain from random noise to x0."""
        sample = torch.randn(
            batch_size, self.pred_len, self.input_dim, device=device
        )
        steps = torch.linspace(
            self.scheduler.num_steps - 1,
            0,
            steps=min(self.ddim_steps, self.scheduler.num_steps),
            device=device,
        ).long()

        for idx, step in enumerate(steps):
            t = torch.full(
                (batch_size,), int(step.item()), device=device, dtype=torch.long
            )
            t_prev_value = -1 if idx == len(steps) - 1 else int(steps[idx + 1].item())
            t_prev = torch.full_like(t, t_prev_value)
            if self.prediction_target == "x0":
                pred_x0 = self.denoiser(sample, t, condition)
                eps = self.scheduler.eps_from_x0(sample, t, pred_x0)
                sample = self.scheduler.ddim_step(sample, t, t_prev, eps)
            else:
                pred_noise = self.denoiser(sample, t, condition)
                sample = self.scheduler.ddim_step(sample, t, t_prev, pred_noise)
        return sample

    def sample(
        self, query_context: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        condition_dict = self.prepare_condition(query_context)
        condition = condition_dict["condition"]
        B = query_context.shape[0]
        device = query_context.device

        if num_samples <= 1:
            return self._ddim_loop(condition, B, device)

        # Multi-sample ensemble: run K independent DDIM chains from different
        # random noise, then average the predictions.  The condition (retrieval
        # + encoding + fusion) is computed once and reused across all chains.
        accum = torch.zeros(B, self.pred_len, self.input_dim, device=device)
        for _ in range(num_samples):
            accum += self._ddim_loop(condition, B, device)
        return accum / num_samples
