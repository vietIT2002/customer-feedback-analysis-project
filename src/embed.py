"""Compute sentence embeddings, with device auto-detection and optional cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_device(requested: str) -> str:
    """Map 'auto'/'cuda'/'cpu' to an actually-available device."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def embed_texts(texts: list[str], cfg: dict) -> np.ndarray:
    """Return embeddings for `texts`.

    If `cache_path` is set and matches the current input count, the cached
    array is reused; otherwise embeddings are recomputed and saved.
    """
    from sentence_transformers import SentenceTransformer

    cache_path = cfg.get("cache_path")
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape[0] == len(texts):
                print(f"[embed] reusing cached embeddings: {cache_path}")
                return cached
            print("[embed] cache size mismatch; recomputing")

    device = resolve_device(cfg.get("device", "auto"))
    print(f"[embed] model={cfg['model_name']} device={device}")
    model = SentenceTransformer(cfg["model_name"], device=device)

    embeddings = model.encode(
        texts,
        batch_size=cfg.get("batch_size", 32),
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"[embed] saved embeddings -> {cache_path} shape={embeddings.shape}")

    return embeddings
