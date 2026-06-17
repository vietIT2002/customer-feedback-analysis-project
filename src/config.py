"""Load and validate the pipeline configuration from a YAML file."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Typed view over config.yaml. Unknown sections are kept in `raw`."""

    data: dict[str, Any]
    preprocess: dict[str, Any]
    embedding: dict[str, Any]
    umap: dict[str, Any]
    hdbscan: dict[str, Any]
    bertopic: dict[str, Any]
    summarizer: dict[str, Any]
    output: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        required = [
            "data",
            "preprocess",
            "embedding",
            "umap",
            "hdbscan",
            "bertopic",
            "summarizer",
            "output",
        ]
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f"Config is missing required sections: {missing}")

        cfg = cls(
            data=raw["data"],
            preprocess=raw["preprocess"],
            embedding=raw["embedding"],
            umap=raw["umap"],
            hdbscan=raw["hdbscan"],
            bertopic=raw["bertopic"],
            summarizer=raw["summarizer"],
            output=raw["output"],
            raw=raw,
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if not self.data.get("text_column"):
            raise ValueError("data.text_column must be set")
        backend = self.summarizer.get("backend")
        valid = {"llama_cpp", "anthropic", "openai_compatible"}
        if self.summarizer.get("enabled") and backend not in valid:
            raise ValueError(
                f"summarizer.backend must be one of {sorted(valid)}, got {backend!r}"
            )

    @property
    def output_dir(self) -> Path:
        d = Path(self.output["dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d
