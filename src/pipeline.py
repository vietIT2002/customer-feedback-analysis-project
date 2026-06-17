"""End-to-end pipeline CLI.

Usage:
    python -m src.pipeline --config config.yaml
    python -m src.pipeline --config config.yaml --no-summarize
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import embed, ingest, preprocess, summarize, topic_model
from .config import Config


def run(config_path: str, do_summarize: bool | None = None) -> None:
    cfg = Config.load(config_path)
    out_dir = cfg.output_dir

    # 1. Ingest -------------------------------------------------------------
    df = ingest.load_feedback(cfg.data)
    print(f"[ingest] {len(df)} rows from {cfg.data['path']}")

    # 2. Preprocess (clean -> tokenize -> stopwords -> dedup) ---------------
    df = preprocess.preprocess(df, cfg.preprocess)
    texts = df["clean"].tolist()
    print(f"[preprocess] {len(texts)} rows after cleaning/dedup")
    if not texts:
        raise SystemExit("No documents left after preprocessing; check the input data.")

    # 3. Embed --------------------------------------------------------------
    embeddings = embed.embed_texts(texts, cfg.embedding)
    print(f"[embed] shape={embeddings.shape}")

    # 4. Topic model --------------------------------------------------------
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(
        cfg.embedding["model_name"],
        device=embed.resolve_device(cfg.embedding.get("device", "auto")),
    )
    model = topic_model.build_model(cfg, embedding_model=embedding_model)
    topics, _ = topic_model.fit(model, texts, embeddings, cfg)

    score = topic_model.silhouette(model.umap_model.embedding_, topics)
    n_topics = len([t for t in set(topics) if t != -1])
    n_noise = sum(1 for t in topics if t == -1)
    print(f"[topic_model] {n_topics} topics, {n_noise} noise docs, silhouette={score}")

    topic_model.save(model, cfg.output["model_dir"])

    # 5. Summarize ----------------------------------------------------------
    if do_summarize is None:
        do_summarize = cfg.summarizer.get("enabled", True)

    if do_summarize:
        summarizer = summarize.make_summarizer(cfg.summarizer)
        result_df = summarizer.summarize_topics(model)
    else:
        # Keyword-only output when summarization is skipped.
        info = model.get_topic_info()
        result_df = info[info.Topic != -1][["Topic", "Name"]].rename(
            columns={"Name": "Keywords"}
        )

    csv_path = out_dir / cfg.output["topics_csv"]
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[output] wrote {len(result_df)} topics -> {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer feedback topic analysis")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--summarize", dest="summarize", action="store_true", help="Force LLM summaries"
    )
    group.add_argument(
        "--no-summarize",
        dest="summarize",
        action="store_false",
        help="Skip LLM summaries (keywords only)",
    )
    parser.set_defaults(summarize=None)
    args = parser.parse_args()

    if not Path(args.config).exists():
        raise SystemExit(f"Config not found: {args.config}")
    run(args.config, do_summarize=args.summarize)


if __name__ == "__main__":
    main()
