"""Generate a natural-language summary per topic.

Two interchangeable backends, selected by `summarizer.backend` in config:
  - llama_cpp : local GGUF model (offline, no data leaves the machine)
  - anthropic : Claude API (no GPU needed; requires ANTHROPIC_API_KEY)

Both share on-disk caching keyed by a hash of the prompt, so re-runs and
retrains don't pay for already-computed summaries.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


def _build_prompt(template: str, docs: list[str], keywords: list[str]) -> str:
    return template.replace("[DOCUMENTS]", "\n".join(docs)).replace(
        "[KEYWORDS]", ", ".join(keywords)
    )


class _CachingSummarizer(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        cache_dir = cfg.get("cache_dir")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, prompt: str) -> Path | None:
        if not self.cache_dir:
            return None
        h = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _respond_cached(self, prompt: str) -> str:
        path = self._cache_path(prompt)
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        response = self._generate(prompt)
        if path:
            path.write_text(
                json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False),
                encoding="utf-8",
            )
        return response

    @abstractmethod
    def _generate(self, prompt: str) -> str: ...

    def summarize_topics(self, topic_model) -> pd.DataFrame:
        """Return a DataFrame: Topic | Keywords | Summary, for non-noise topics."""
        template = self.cfg["prompt_template"]
        docs_per_topic = self.cfg.get("docs_per_topic", 20)
        max_keywords = self.cfg.get("max_keywords", 10)

        info = topic_model.get_topic_info()
        rep_docs = topic_model.get_representative_docs()

        rows = []
        for topic_id in info.Topic.tolist():
            if topic_id == -1:
                continue
            keywords = [w for w, _ in topic_model.get_topic(topic_id)][:max_keywords]
            docs = rep_docs.get(topic_id, [])[:docs_per_topic]
            prompt = _build_prompt(template, docs, keywords)
            try:
                summary = self._respond_cached(prompt).strip()
            except Exception as e:  # one bad topic shouldn't kill the run
                summary = f"[ERROR: {e}]"
            print(f"[summarize] topic {topic_id} done")
            rows.append(
                {
                    "Topic": topic_id,
                    "Keywords": " | ".join(keywords[:5]),
                    "Summary": summary,
                }
            )
        return pd.DataFrame(rows)


class LlamaCppSummarizer(_CachingSummarizer):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from llama_cpp import Llama

        lc = cfg["llama_cpp"]
        self.llm = Llama(
            model_path=lc["model_path"],
            n_ctx=lc.get("n_ctx", 4096),
            n_gpu_layers=lc.get("n_gpu_layers", 0),
            n_batch=lc.get("n_batch", 128),
            n_threads=lc.get("n_threads", 4),
            verbose=False,
        )

    def _generate(self, prompt: str) -> str:
        result = self.llm(
            prompt,
            max_tokens=self.cfg.get("max_tokens", 80),
            temperature=self.cfg.get("temperature", 0.7),
            top_p=self.cfg.get("top_p", 0.8),
        )
        choice = result["choices"][0]
        return choice.get("text") or choice.get("message", {}).get("content", "[NO OUTPUT]")


class AnthropicSummarizer(_CachingSummarizer):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import anthropic  # requires ANTHROPIC_API_KEY in the environment

        self.client = anthropic.Anthropic()
        self.model = cfg["anthropic"]["model"]

    def _generate(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.cfg.get("max_tokens", 80),
            temperature=self.cfg.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")


class OpenAICompatibleSummarizer(_CachingSummarizer):
    """Any OpenAI-compatible chat API: OpenRouter, Mistral, Groq, etc.

    Uses stdlib urllib (no extra dependency). Configure base_url, model and the
    name of the env var holding the API key in config.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        oc = cfg["openai_compatible"]
        self.base_url = oc["base_url"].rstrip("/")
        self.model = oc["model"]
        key_env = oc.get("api_key_env", "OPENROUTER_API_KEY")
        self.api_key = os.getenv(key_env)
        if not self.api_key:
            raise RuntimeError(
                f"Environment variable {key_env!r} is not set. "
                f"Set it to your API key before running."
            )

    def _generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.cfg.get("max_tokens", 80),
                "temperature": self.cfg.get("temperature", 0.7),
                "top_p": self.cfg.get("top_p", 0.8),
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:200]
            raise RuntimeError(f"API HTTP {e.code}: {detail}") from e
        return data["choices"][0]["message"]["content"]


def make_summarizer(cfg: dict) -> _CachingSummarizer:
    backend = cfg["backend"]
    if backend == "llama_cpp":
        return LlamaCppSummarizer(cfg)
    if backend == "anthropic":
        return AnthropicSummarizer(cfg)
    if backend == "openai_compatible":
        return OpenAICompatibleSummarizer(cfg)
    raise ValueError(f"Unknown summarizer backend: {backend!r}")
