"""Vietnamese text preprocessing.

Correct order matters for PhoBERT-family models:
    1. light clean (lowercase, collapse whitespace)
    2. WORD SEGMENTATION with pyvi  -> "chất lượng" becomes "chất_lượng"
    3. stopword removal on the segmented tokens
This is the step the original notebook was missing (it imported `tokenize`
but never called it, so embeddings ran on raw whitespace-split text).
"""
from __future__ import annotations

import re

import pandas as pd
from pyvi.ViTokenizer import tokenize

_WHITESPACE = re.compile(r"\s+")


def _clean(text: str, lowercase: bool) -> str:
    text = text.strip()
    if lowercase:
        text = text.lower()
    return _WHITESPACE.sub(" ", text)


def _remove_stopwords(segmented: str, stopwords: set[str]) -> str:
    # pyvi joins compound words with underscores; split on whitespace.
    tokens = [t for t in segmented.split() if t not in stopwords]
    return " ".join(tokens)


def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return df with an added `clean` column ready for embedding.

    Drops empty / too-short rows and (optionally) exact duplicates.
    """
    lowercase = cfg.get("lowercase", True)
    min_chars = cfg.get("min_chars", 0)
    dedup = cfg.get("dedup", True)
    stopwords = set(cfg.get("extra_stopwords") or [])

    cleaned = df["text"].apply(lambda t: _clean(t, lowercase))
    segmented = cleaned.apply(tokenize)
    df = df.copy()
    df["clean"] = segmented.apply(lambda s: _remove_stopwords(s, stopwords))

    # Filter: drop empties and texts shorter than min_chars (measured on the
    # de-segmented form so underscores don't inflate the length).
    lengths = df["clean"].str.replace("_", " ", regex=False).str.len()
    df = df[(df["clean"].str.strip() != "") & (lengths >= min_chars)]

    if dedup:
        df = df.drop_duplicates(subset="clean")

    return df.reset_index(drop=True)
