"""Build and fit the BERTopic model (UMAP -> HDBSCAN -> BERTopic)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build_model(cfg, embedding_model=None):
    """Construct a BERTopic model from the umap/hdbscan/bertopic config blocks."""
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from hdbscan import HDBSCAN
    from umap import UMAP

    u = cfg.umap
    umap_model = UMAP(
        n_components=u["n_components"],
        n_neighbors=u["n_neighbors"],
        min_dist=u["min_dist"],
        metric=u["metric"],
        random_state=u["random_state"],
    )

    h = cfg.hdbscan
    hdbscan_model = HDBSCAN(
        min_cluster_size=h["min_cluster_size"],
        min_samples=h["min_samples"],
        metric=h["metric"],
        cluster_selection_method=h["cluster_selection_method"],
        prediction_data=True,
    )

    b = cfg.bertopic
    mmr = MaximalMarginalRelevance(diversity=b.get("mmr_diversity", 0.1))

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=mmr,
        language=b.get("language", "multilingual"),
        verbose=True,
    )


def fit(model, texts: list[str], embeddings: np.ndarray, cfg):
    """Fit BERTopic; optionally refine words with KeyBERTInspired.

    Returns (topics, probs). Reuses precomputed `embeddings` so we don't encode
    twice.
    """
    topics, probs = model.fit_transform(texts, embeddings=embeddings)

    if cfg.bertopic.get("use_keybert", True):
        from bertopic.representation import KeyBERTInspired

        model.update_topics(texts, representation_model=KeyBERTInspired())

    return topics, probs


def silhouette(reduced_embeddings: np.ndarray, clusters) -> float | None:
    """Silhouette score, guarding the degenerate cases that crash sklearn."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(clusters)
    n_labels = len(set(labels.tolist()))
    if n_labels < 2 or n_labels >= len(labels):
        return None
    try:
        return float(silhouette_score(reduced_embeddings, labels))
    except Exception:
        return None


def save(model, model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    # serialization="safetensors" keeps the artifact portable and small.
    model.save(str(model_dir), serialization="safetensors", save_ctfidf=True)
    print(f"[topic_model] saved -> {model_dir}")
