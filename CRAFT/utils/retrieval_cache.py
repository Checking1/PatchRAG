import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from dtaidistance import dtw

from ts_benchmark.common.constant import ROOT_PATH


CACHE_VERSION = 3
DTW_BACKEND = "baseline_target_only_dtaidistance"


def _to_numpy_float32(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    return np.ascontiguousarray(array)


def _to_numpy_float64(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    return np.ascontiguousarray(array)


def stable_window_hash(window: Any) -> str:
    array = _to_numpy_float32(window)
    hasher = hashlib.sha1()
    hasher.update(str(tuple(array.shape)).encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def batch_window_hashes(windows: Any) -> List[str]:
    array = _to_numpy_float32(windows)
    if array.ndim < 2:
        raise ValueError("Expected batched windows with shape [N, ...].")
    return [stable_window_hash(array[idx]) for idx in range(array.shape[0])]


def hash_string_sequence(values: Sequence[str]) -> str:
    hasher = hashlib.sha1()
    for value in values:
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def tensor_digest(value: Any) -> str:
    array = _to_numpy_float32(value)
    hasher = hashlib.sha1()
    hasher.update(str(tuple(array.shape)).encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def normalize_sequence(sequence: Any) -> np.ndarray:
    array = _to_numpy_float64(sequence).reshape(-1)
    mean = np.mean(array)
    std = np.std(array)
    if std < 1e-8:
        return array - mean
    return (array - mean) / std


def extract_target_sequence(sequence: Any, target_channel_index: int = 0) -> np.ndarray:
    array = _to_numpy_float64(sequence)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError("Expected sequence with shape [L] or [L, C].")
    if target_channel_index < 0 or target_channel_index >= array.shape[1]:
        raise IndexError("target_channel_index is out of range for DTW target extraction.")
    return np.ascontiguousarray(array[:, target_channel_index])


def downsample_sequence(sequence: Any, stride: int) -> np.ndarray:
    return extract_target_sequence(sequence)


def build_dtw_views(
    contexts: Any,
    stride: int,
    target_channel_index: int = 0,
) -> np.ndarray:
    array = _to_numpy_float32(contexts)
    if array.ndim != 3:
        raise ValueError("Expected contexts with shape [N, L, C].")
    views = [extract_target_sequence(array[idx], target_channel_index) for idx in range(array.shape[0])]
    return np.stack(views, axis=0).astype(np.float64)


def dtw_distance(
    query_view: Any,
    candidate_view: Any,
    window: Optional[int] = None,
    normalize_ts: bool = False,
) -> float:
    query = extract_target_sequence(query_view)
    candidate = extract_target_sequence(candidate_view)
    if query.shape[0] != candidate.shape[0]:
        return np.inf
    if normalize_ts:
        query = normalize_sequence(query)
        candidate = normalize_sequence(candidate)
    kwargs = {} if window is None else {"window": int(window)}
    return float(dtw.distance(query, candidate, **kwargs))


def compute_top_dtw_shortlist(
    query_view: Any,
    bank_views: Any,
    top_k: int,
    window: Optional[int] = None,
    exclude_index: Optional[int] = None,
    normalize_ts: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    bank = _to_numpy_float64(bank_views)
    if bank.ndim != 2:
        raise ValueError("Expected bank views with shape [N, L].")

    scored: List[Tuple[int, float]] = []
    for index in range(bank.shape[0]):
        if exclude_index is not None and index == exclude_index:
            continue
        distance = dtw_distance(
            query_view,
            bank[index],
            window=window,
            normalize_ts=normalize_ts,
        )
        scored.append((index, distance))

    if not scored:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    scored.sort(key=lambda item: item[1])
    top_items = scored[:top_k]
    indices = np.asarray([item[0] for item in top_items], dtype=np.int64)
    distances = np.asarray([item[1] for item in top_items], dtype=np.float32)
    return indices, distances


def build_query_shortlist_entries(
    query_contexts: Any,
    bank_views: Any,
    *,
    recall_k: int,
    dtw_window: Optional[int],
    dtw_downsample_stride: int,
    memory_hash_to_index: Optional[Dict[str, int]] = None,
    query_hashes: Optional[Sequence[str]] = None,
    existing_entries: Optional[Dict[str, Dict[str, Any]]] = None,
    dtw_normalize_ts: bool = False,
    dtw_target_channel_index: int = 0,
) -> Dict[str, Dict[str, Any]]:
    contexts = _to_numpy_float32(query_contexts)
    if contexts.size == 0:
        return {} if existing_entries is None else dict(existing_entries)
    if contexts.ndim != 3:
        raise ValueError("Expected query contexts with shape [N, L, C].")

    if query_hashes is None:
        query_hashes = batch_window_hashes(contexts)
    query_views = build_dtw_views(
        contexts,
        dtw_downsample_stride,
        target_channel_index=dtw_target_channel_index,
    )
    entries = {} if existing_entries is None else dict(existing_entries)
    bank = _to_numpy_float64(bank_views)
    if bank.ndim != 2:
        raise ValueError("Expected bank views with shape [N, L].")
    bank_size = bank.shape[0]

    for index, query_hash in enumerate(query_hashes):
        if query_hash in entries:
            continue
        memory_index = None
        if memory_hash_to_index is not None:
            memory_index = memory_hash_to_index.get(query_hash)
        exclude_index = memory_index if memory_index is not None and bank_size > 1 else None
        shortlist_size = max(
            1,
            min(recall_k, bank_size - (1 if exclude_index is not None else 0)),
        )
        recall_indices, recall_distances = compute_top_dtw_shortlist(
            query_views[index],
            bank,
            top_k=shortlist_size,
            window=dtw_window,
            exclude_index=exclude_index,
            normalize_ts=dtw_normalize_ts,
        )
        entries[query_hash] = {
            "memory_index": memory_index,
            "recall_indices": recall_indices.tolist(),
            "recall_distances": recall_distances.tolist(),
        }
    return entries


def build_dtw_cache_metadata(
    contexts: Any,
    *,
    pred_len: int,
    recall_k: int,
    dtw_window: Optional[int],
    dtw_downsample_stride: int,
    query_digest: str,
    dtw_normalize_ts: bool = False,
    dtw_target_channel_index: int = 0,
) -> Dict[str, Any]:
    array = _to_numpy_float32(contexts)
    return {
        "version": CACHE_VERSION,
        "dtw_backend": DTW_BACKEND,
        "memory_digest": tensor_digest(array),
        "memory_size": int(array.shape[0]),
        "seq_len": int(array.shape[1]),
        "channels": int(array.shape[2]),
        "pred_len": int(pred_len),
        "recall_k": int(recall_k),
        "dtw_window": None if dtw_window is None else int(dtw_window),
        "dtw_downsample_stride": int(dtw_downsample_stride),
        "dtw_normalize_ts": bool(dtw_normalize_ts),
        "dtw_target_channel_index": int(dtw_target_channel_index),
        "query_digest": query_digest,
    }


def build_cache_key(metadata: Dict[str, Any]) -> str:
    serialized = json.dumps(metadata, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def get_dtw_cache_path(cache_key: str) -> str:
    cache_dir = os.path.join(ROOT_PATH, "result", "_cache", "craft_dtw")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"dtw_cache_{cache_key}.pt")


def load_dtw_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return None
    return torch.load(cache_path, map_location="cpu")


def validate_dtw_cache(
    payload: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
    required_query_hashes: Optional[Sequence[str]] = None,
) -> Tuple[bool, str]:
    if payload is None:
        return False, "missing"
    payload_metadata = payload.get("metadata")
    if payload_metadata is None:
        return False, "metadata_missing"

    fields = [
        "version",
        "dtw_backend",
        "memory_digest",
        "memory_size",
        "seq_len",
        "channels",
        "pred_len",
        "recall_k",
        "dtw_window",
        "dtw_downsample_stride",
        "dtw_normalize_ts",
        "dtw_target_channel_index",
        "query_digest",
    ]
    for field in fields:
        if payload_metadata.get(field) != metadata.get(field):
            return False, f"{field}_mismatch"

    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return False, "entries_missing"

    memory_size = int(metadata["memory_size"])
    recall_k = int(metadata["recall_k"])
    query_hashes = payload.get("context_hashes")
    if query_hashes is not None and len(query_hashes) != len(set(query_hashes)):
        return False, "context_hashes_duplicated"

    for query_hash in required_query_hashes or []:
        entry = entries.get(query_hash)
        if entry is None:
            return False, "entry_missing"
        recall_indices = entry.get("recall_indices")
        recall_distances = entry.get("recall_distances")
        if not isinstance(recall_indices, list) or not isinstance(recall_distances, list):
            return False, "entry_format_invalid"
        if len(recall_indices) != len(recall_distances):
            return False, "entry_length_mismatch"
        if not recall_indices:
            return False, "entry_empty"
        if len(recall_indices) > recall_k:
            return False, "entry_too_large"
        if any(index < 0 or index >= memory_size for index in recall_indices):
            return False, "entry_index_out_of_range"
    return True, "ok"


def save_dtw_cache(cache_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temp_path = cache_path + ".tmp"
    torch.save(payload, temp_path)
    os.replace(temp_path, cache_path)


def build_dtw_cache_payload(
    metadata: Dict[str, Any],
    context_hashes: Sequence[str],
    entries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "metadata": metadata,
        "context_hashes": list(context_hashes),
        "entries": entries,
    }


def get_retrieval_artifact_dir(run_tag: str) -> str:
    artifact_dir = os.path.join(ROOT_PATH, "result", "_artifacts", "craft_retrieval", run_tag)
    os.makedirs(artifact_dir, exist_ok=True)
    return artifact_dir
