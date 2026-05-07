import copy
import json
import math
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import optim

from CRAFT.models.craft_model import CRAFTModel
from CRAFT.utils.retrieval_cache import (
    batch_window_hashes,
    build_cache_key,
    build_dtw_cache_metadata,
    build_dtw_cache_payload,
    build_dtw_views,
    build_query_shortlist_entries,
    get_dtw_cache_path,
    get_retrieval_artifact_dir,
    hash_string_sequence,
    load_dtw_cache,
    save_dtw_cache,
    validate_dtw_cache,
)
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.utils import (
    EarlyStopping,
    forecasting_data_provider,
    train_val_split,
)
from ts_benchmark.utils.get_device import get_device
from ts_benchmark.utils.get_file_name import get_unique_file_suffix


MODEL_HYPER_PARAMS = {
    "d_model": 256,
    "d_ff": 512,
    "n_heads": 4,
    "num_query_layers": 2,
    "num_assembly_layers": 2,
    "num_diffusion_layers": 4,
    "patch_len": 24,
    "stride": 24,
    "memory_fragment_stride": 1,
    "top_k": 4,
    "top_m": 8,
    "retrieval_pool_size": None,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.0001,
    "lradj": "constant",
    "num_epochs": 100,
    "num_workers": 0,
    "patience": 10,
    "norm": True,
    "loss": "MSE",
    "num_diffusion_steps": 200,
    "ddim_steps": 50,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "max_memory_chunks": 2048,
    "subtract_last": False,
    "affine": True,
    "coarse_recall_k": 8,
    "dtw_window": None,
    "dtw_downsample_stride": 1,
    "dtw_normalize_ts": False,
    "dtw_target_channel_index": 0,
    "reranker_hidden_dim": 256,
    "reranker_loss_weight": 0.1,
    "reranker_temperature": 1.0,
    "reranker_target_temperature": 1.0,
    "save_retrieval_artifacts": True,
    "retrieval_artifact_max_batches": 4,
    "use_grad_clip": False,
    "grad_clip_norm": 1.0,
    "use_predictive_reranker": False,
    "use_reranker_fragment_weight": False,
    "use_retrieval_quality_gate": False,
    "use_forecast_val": False,
    "num_inference_samples": 1,
    "snr_gamma": None,
    "x0_loss_weight": 0.0,
    "use_adaln": False,
    "use_rma": False,
    "prediction_target": "eps",
    "retrieval_dropout_prob": 0.0,
    "cfg_scale": 1.0,
    "cfg_drop_query": False,
}


class CRAFT(DeepForecastingModelBase):
    """
    Cross-sample Retrieval and Assembly of Fragments for Time-series.
    """

    def __init__(self, **kwargs):
        super(CRAFT, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        self.memory_bank_contexts = None
        self.memory_bank_future_targets = None
        self.memory_bank_context_hashes = None
        self.memory_hash_to_index = None
        self.dtw_cache_payload = None
        self.dtw_cache_path = None
        self.retrieval_artifact_dir = None
        self.retrieval_artifact_counter = 0

    @property
    def model_name(self):
        return "CRAFT"

    def _init_model(self):
        return CRAFTModel(self.config)

    def _build_memory_bank_tensors(self, data: pd.DataFrame):
        values = data.values.astype(np.float32)
        total_len = values.shape[0]
        sample_len = self.config.seq_len + self.config.pred_len
        if total_len < sample_len:
            raise ValueError("Training data is shorter than one context/future pair.")

        contexts = []
        futures = []
        for start in range(total_len - sample_len + 1):
            mid = start + self.config.seq_len
            end = mid + self.config.pred_len
            contexts.append(values[start:mid])
            futures.append(values[mid:end])

        contexts = torch.tensor(np.stack(contexts), dtype=torch.float32)
        futures = torch.tensor(np.stack(futures), dtype=torch.float32)

        if (
            self.config.retrieval_pool_size is not None
            and contexts.shape[0] > self.config.retrieval_pool_size
        ):
            indices = torch.linspace(
                0,
                contexts.shape[0] - 1,
                steps=self.config.retrieval_pool_size,
                dtype=torch.long,
            )
            contexts = contexts.index_select(0, indices)
            futures = futures.index_select(0, indices)

        context_hashes = batch_window_hashes(contexts)
        return contexts, futures, context_hashes

    def _collect_context_windows(self, values: np.ndarray) -> np.ndarray:
        total_len = values.shape[0]
        if total_len < self.config.seq_len:
            return np.empty((0, self.config.seq_len, values.shape[1]), dtype=np.float32)
        windows = [
            values[start : start + self.config.seq_len]
            for start in range(total_len - self.config.seq_len + 1)
        ]
        if not windows:
            return np.empty((0, self.config.seq_len, values.shape[1]), dtype=np.float32)
        return np.asarray(windows, dtype=np.float32)

    def _build_valid_query_contexts(self, valid_data: Optional[pd.DataFrame]) -> np.ndarray:
        if valid_data is None or len(valid_data) < self.config.seq_len:
            return np.empty(
                (0, self.config.seq_len, self.memory_bank_contexts.shape[-1]),
                dtype=np.float32,
            )
        return self._collect_context_windows(valid_data.values.astype(np.float32))

    def _build_forecast_query_contexts(self, series: pd.DataFrame) -> np.ndarray:
        values = series.values.astype(np.float32)
        return self._collect_context_windows(values)

    def build_query_contexts_from_array(self, values: np.ndarray) -> np.ndarray:
        return self._collect_context_windows(values.astype(np.float32))

    def prepare_query_context_array(
        self, values: np.ndarray, exog_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        prepared = values.astype(np.float32)
        if exog_values is not None:
            prepared = np.concatenate((prepared, exog_values.astype(np.float32)), axis=-1)
        if self.config.norm:
            original_shape = prepared.shape
            prepared = self.scaler.transform(prepared.reshape(-1, prepared.shape[-1])).reshape(
                original_shape
            )
        return prepared.astype(np.float32)

    def _ensure_full_dtw_cache(self, query_contexts: np.ndarray) -> None:
        memory_query_hashes = list(self.memory_bank_context_hashes)
        extra_query_hashes = (
            batch_window_hashes(query_contexts) if len(query_contexts) > 0 else []
        )
        all_query_hashes = list(dict.fromkeys(memory_query_hashes + extra_query_hashes))
        query_digest = hash_string_sequence(all_query_hashes)
        metadata = build_dtw_cache_metadata(
            self.memory_bank_contexts,
            pred_len=self.config.pred_len,
            recall_k=self.config.coarse_recall_k,
            dtw_window=self.config.dtw_window,
            dtw_downsample_stride=self.config.dtw_downsample_stride,
            query_digest=query_digest,
            dtw_normalize_ts=getattr(self.config, "dtw_normalize_ts", False),
            dtw_target_channel_index=getattr(
                self.config, "dtw_target_channel_index", 0
            ),
        )
        cache_key = build_cache_key(metadata)
        cache_path = get_dtw_cache_path(cache_key)
        payload = load_dtw_cache(cache_path)
        valid, _ = validate_dtw_cache(
            payload,
            metadata,
            required_query_hashes=all_query_hashes,
        )
        if valid:
            self.dtw_cache_payload = payload
            self.dtw_cache_path = cache_path
            return

        dtw_views = build_dtw_views(
            self.memory_bank_contexts,
            self.config.dtw_downsample_stride,
            target_channel_index=getattr(self.config, "dtw_target_channel_index", 0),
        )
        existing_entries = payload.get("entries", {}) if valid else {}
        entries = build_query_shortlist_entries(
            self.memory_bank_contexts.detach().cpu().numpy(),
            dtw_views,
            recall_k=self.config.coarse_recall_k,
            dtw_window=self.config.dtw_window,
            dtw_downsample_stride=self.config.dtw_downsample_stride,
            memory_hash_to_index=self.memory_hash_to_index,
            query_hashes=self.memory_bank_context_hashes,
            existing_entries=existing_entries,
            dtw_normalize_ts=getattr(self.config, "dtw_normalize_ts", False),
            dtw_target_channel_index=getattr(
                self.config, "dtw_target_channel_index", 0
            ),
        )
        if len(query_contexts) > 0:
            entries = build_query_shortlist_entries(
                query_contexts,
                dtw_views,
                recall_k=self.config.coarse_recall_k,
                dtw_window=self.config.dtw_window,
                dtw_downsample_stride=self.config.dtw_downsample_stride,
                memory_hash_to_index=self.memory_hash_to_index,
                existing_entries=entries,
                dtw_normalize_ts=getattr(self.config, "dtw_normalize_ts", False),
                dtw_target_channel_index=getattr(
                    self.config, "dtw_target_channel_index", 0
                ),
            )

        payload = build_dtw_cache_payload(
            metadata=metadata,
            context_hashes=all_query_hashes,
            entries=entries,
        )
        payload["dtw_views"] = dtw_views
        save_dtw_cache(cache_path, payload)
        self.dtw_cache_payload = payload
        self.dtw_cache_path = cache_path

    def precompute_query_shortlists(self, query_contexts: np.ndarray) -> None:
        if self.memory_bank_contexts is None:
            raise ValueError("Memory bank is not built.")
        if query_contexts is None:
            query_contexts = np.empty(
                (0, self.config.seq_len, self.memory_bank_contexts.shape[-1]),
                dtype=np.float32,
            )
        self._ensure_full_dtw_cache(query_contexts)

    def _refresh_memory_bank(self, device: torch.device):
        if self.memory_bank_contexts is None:
            raise ValueError("Memory bank is not built.")
        future_targets = self.memory_bank_future_targets
        self.model.set_memory_bank(
            self.memory_bank_contexts.to(device),
            future_targets=future_targets,
            context_hashes=self.memory_bank_context_hashes,
            dtw_cache_payload=self.dtw_cache_payload,
            dtw_cache_path=self.dtw_cache_path,
        )

    def _save_retrieval_artifact(self, debug_info):
        if (
            not getattr(self.config, "save_retrieval_artifacts", False)
            or debug_info is None
            or self.retrieval_artifact_dir is None
            or self.retrieval_artifact_counter
            >= getattr(self.config, "retrieval_artifact_max_batches", 0)
        ):
            return

        suffix = get_unique_file_suffix().replace(".csv", "")
        artifact_base = os.path.join(
            self.retrieval_artifact_dir,
            f"retrieval_{self.retrieval_artifact_counter:03d}{suffix}",
        )
        np.savez_compressed(
            artifact_base + ".npz",
            query_context=debug_info["query_context"],
            dtw_recall_indices=debug_info["dtw_recall_indices"],
            dtw_recall_distances=debug_info["dtw_recall_distances"],
            reranker_scores=debug_info["reranker_scores"],
            reranker_topk_indices=debug_info["reranker_topk_indices"],
            reranker_topk_scores=debug_info["reranker_topk_scores"],
            selected_contexts=debug_info["selected_contexts"],
        )
        summary = {
            "query_hashes": debug_info["query_hashes"],
            "reranker_loss": debug_info.get("reranker_loss"),
            "dtw_cache_path": self.dtw_cache_path,
            "topk_shape": list(np.asarray(debug_info["reranker_topk_indices"]).shape),
        }
        with open(artifact_base + ".json", "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        self.retrieval_artifact_counter += 1

    def _validate_diffusion(self, valid_data_loader) -> float:
        if valid_data_loader is None:
            return np.nan

        self.model.eval()
        device = get_device()
        total_loss = []
        with torch.no_grad():
            for input_data, target, _, _ in valid_data_loader:
                input_data = input_data.to(device)
                future = target[:, -self.config.pred_len :, :].to(device)
                loss_dict = self.model.compute_loss(input_data, future)
                total_loss.append(loss_dict["loss"].item())

        self.model.train()
        return float(np.mean(total_loss)) if total_loss else np.nan

    def _validate_forecast_mse(self, valid_data_loader) -> float:
        """
        Compute validation MSE using full DDIM inference (deterministic sampling
        from pure noise → x0), matching the actual test-time evaluation metric.

        This aligns early stopping with the true forecasting objective, avoiding
        the mismatch where noise-prediction loss (random t) and forecast MSE
        can move in opposite directions during training.
        """
        if valid_data_loader is None:
            return np.nan

        self.model.eval()
        device = get_device()
        total_se = []
        with torch.no_grad():
            for input_data, target, _, _ in valid_data_loader:
                input_data = input_data.to(device)
                true_future = target[:, -self.config.pred_len :, :].to(device)
                pred_future = self.model.sample(input_data)
                # mean over (T, C) per sample, then collect
                se = (pred_future - true_future).pow(2).mean(dim=(1, 2))
                total_se.append(se)

        self.model.train()
        if not total_se:
            return np.nan
        return float(torch.cat(total_se).mean().item())

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "CRAFT":
        if covariates is None:
            covariates = {}

        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            train_valid_data = pd.concat([train_valid_data, exog_data], axis=1)

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, self.config.seq_len
        )

        self.scaler.fit(train_data.values)
        if self.config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )
            if valid_data is not None:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )

        (
            self.memory_bank_contexts,
            self.memory_bank_future_targets,
            self.memory_bank_context_hashes,
        ) = self._build_memory_bank_tensors(train_data)
        self.memory_hash_to_index = {
            context_hash: index
            for index, context_hash in enumerate(self.memory_bank_context_hashes)
        }
        valid_query_contexts = self._build_valid_query_contexts(valid_data)
        self._ensure_full_dtw_cache(valid_query_contexts)
        self.model = self._init_model()

        _, train_data_loader = forecasting_data_provider(
            train_data,
            self.config,
            timeenc=1,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )
        valid_data_loader = None
        if valid_data is not None:
            _, valid_data_loader = forecasting_data_provider(
                valid_data,
                self.config,
                timeenc=1,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
            )

        device = get_device()
        self.model.to(device)
        self._refresh_memory_bank(device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.early_stopping = EarlyStopping(patience=self.config.patience)
        self.check_point = copy.deepcopy(self.model.state_dict())

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("----------------------------------------------------------", self.model_name)
        print(f"Total trainable parameters: {total_params}")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            self._refresh_memory_bank(device)

            for input_data, target, _, _ in train_data_loader:
                input_data = input_data.to(device)
                future = target[:, -self.config.pred_len :, :].to(device)

                optimizer.zero_grad()
                loss_dict = self.model.compute_loss(input_data, future)
                loss_dict["loss"].backward()
                if getattr(self.config, "use_grad_clip", False):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=getattr(self.config, "grad_clip_norm", 1.0),
                    )
                optimizer.step()

            if getattr(self.config, "use_forecast_val", True):
                valid_loss = self._validate_forecast_mse(valid_data_loader)
            else:
                valid_loss = self._validate_diffusion(valid_data_loader)
            if not math.isnan(valid_loss):
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    self.check_point = copy.deepcopy(self.model.state_dict())
                if self.early_stopping.early_stop:
                    break
            else:
                self.check_point = copy.deepcopy(self.model.state_dict())

            if self.config.lradj == "type1":
                lr = self.config.lr * (0.5 ** epoch)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

        if self.check_point is not None:
            self.model.load_state_dict(self.check_point)
            self._refresh_memory_bank(device)

        self._save_checkpoint()
        return self

    def _save_checkpoint(self) -> None:
        """Save model weights and scaler to disk for reproducible testing."""
        ckpt_dir = os.path.join(
            os.path.abspath(os.path.join(__file__, "..", "..")),
            "result",
            "_checkpoints",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        suffix = get_unique_file_suffix().replace(".csv", "")
        ckpt_path = os.path.join(ckpt_dir, f"craft{suffix}.pth")
        payload = {
            "model_state_dict": self.model.state_dict(),
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "config": {
                k: v
                for k, v in vars(self.config).items()
                if not k.startswith("_")
            },
            "memory_bank_contexts": self.memory_bank_contexts.cpu(),
            "memory_bank_context_hashes": self.memory_bank_context_hashes,
        }
        torch.save(payload, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    def _sample_chunks(
        self, input_tensor: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        if horizon > self.config.pred_len:
            raise ValueError(
                "Offline DTW precompute only supports horizon <= pred_len in CRAFT."
            )
        num_samples = getattr(self.config, "num_inference_samples", 1)
        chunk = self.model.sample(input_tensor, num_samples=num_samples)
        self._save_retrieval_artifact(self.model.get_last_retrieval_debug())
        return chunk[:, :horizon, :]

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        if covariates is None:
            covariates = {}

        series_dim = series.shape[-1]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            series = pd.concat([series, exog_data], axis=1)

        if self.check_point is not None:
            self.model.load_state_dict(self.check_point)

        if self.config.norm:
            series = pd.DataFrame(
                self.scaler.transform(series.values),
                columns=series.columns,
                index=series.index,
            )

        if len(series) < self.config.seq_len:
            raise ValueError("The input series is shorter than seq_len.")

        device = get_device()
        self.model.to(device)
        self._refresh_memory_bank(device)
        self.model.eval()
        self.retrieval_artifact_counter = 0
        self.retrieval_artifact_dir = get_retrieval_artifact_dir(
            f"forecast{get_unique_file_suffix().replace('.csv', '')}"
        )

        context = torch.tensor(
            series.values[-self.config.seq_len :],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            answer = self._sample_chunks(context, horizon)

        answer = answer.squeeze(0).cpu().numpy()
        if self.config.norm:
            answer = self.scaler.inverse_transform(answer)
        return answer[:, :series_dim]

    def batch_forecast(self, horizon: int, batch_maker, **kwargs) -> np.ndarray:
        if self.check_point is not None:
            self.model.load_state_dict(self.check_point)
        if self.model is None:
            raise ValueError("Model not trained. Call forecast_fit() first.")

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]

        covariates = input_data.get("covariates", {}) or {}
        exog_data = covariates.get("exog")
        input_np = self.prepare_query_context_array(input_np, exog_data)

        series_dim = input_data["input"].shape[-1]

        device = get_device()
        self.model.to(device)
        self._refresh_memory_bank(device)
        self.model.eval()
        self.retrieval_artifact_counter = 0
        self.retrieval_artifact_dir = get_retrieval_artifact_dir(
            f"batch_forecast{get_unique_file_suffix().replace('.csv', '')}"
        )

        with torch.no_grad():
            context = torch.tensor(input_np, dtype=torch.float32, device=device)
            answer = self._sample_chunks(context, horizon).cpu().numpy()

        if self.config.norm:
            answer = self.scaler.inverse_transform(
                answer.reshape(-1, answer.shape[-1])
            ).reshape(answer.shape)
        return answer[:, :, :series_dim]
