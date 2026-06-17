"""Load raw feedback into a one-column DataFrame, validating the schema."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_feedback(data_cfg: dict) -> pd.DataFrame:
    """Read .xlsx/.csv and return a DataFrame with a single `text` column.

    Raises a clear error if the file or the configured text column is missing.
    """
    path = Path(data_cfg["path"])
    text_column = data_cfg["text_column"]

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=data_cfg.get("sheet", 0))
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data file type: {suffix} (use .xlsx or .csv)")

    if text_column not in df.columns:
        raise ValueError(
            f"Column {text_column!r} not found. Available columns: {list(df.columns)}"
        )

    out = df[[text_column]].rename(columns={text_column: "text"})
    out["text"] = out["text"].fillna("").astype(str)
    return out.reset_index(drop=True)
