# 🛍️ Customer Feedback Analysis — Vietnamese NLP Pipeline

> Automatically discover **what customers are talking about** from raw Vietnamese
> feedback, and turn each topic into a human-readable summary — built for a Gen Z
> fashion store drowning in daily messages.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![BERTopic](https://img.shields.io/badge/Topic%20Modeling-BERTopic-4B8BBE)
![PhoBERT](https://img.shields.io/badge/Embeddings-SimCSE--PhoBERT-FF6F00)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧩 Overview

A fashion store receives hundreds of customer messages a day. Reading and
classifying them by hand is slow and inconsistent. This project **automates the
whole loop**: it cleans Vietnamese text, groups feedback into topics with an
unsupervised pipeline, and uses an LLM to describe each topic in plain language —
so the team can act on *"what are people complaining about this week?"* in minutes.

- **Input:** an Excel/CSV file with a free-text feedback column
- **Output:** a table of discovered topics, their keywords, and an LLM summary
- **No labels required** — fully unsupervised topic discovery

## 🧭 Architecture

```
 Excel / CSV
     │
     ▼
 ┌──────────────┐   clean → pyvi word-segmentation → stopwords → dedup
 │ preprocess   │   (PhoBERT requires word-segmented input — the key fix)
 └──────┬───────┘
        ▼
 ┌──────────────┐   SimCSE-PhoBERT sentence embeddings (cached to .npy)
 │ embed        │   device auto-detect (CUDA → CPU fallback)
 └──────┬───────┘
        ▼
 ┌──────────────┐   UMAP (dim. reduction) → HDBSCAN (clustering)
 │ topic_model  │   → BERTopic (MMR + KeyBERTInspired keywords)
 └──────┬───────┘   → silhouette score, model saved to disk
        ▼
 ┌──────────────┐   per-topic natural-language summary
 │ summarize    │   pluggable backend: local GGUF  ◀▶  Claude API (cached)
 └──────┬───────┘
        ▼
 topic_representations.csv   (Topic | Keywords | Summary)
```

## ✨ Key Features

- **Correct Vietnamese preprocessing** — word segmentation with `pyvi` *before*
  embedding (PhoBERT-family models need it; skipping it quietly degrades quality).
- **Unsupervised topic discovery** — UMAP + HDBSCAN + BERTopic, no manual labels.
- **Pluggable LLM summarizer** — switch between a fully **offline** GGUF model and
  the **Claude API** with a single config line; responses are cached on disk.
- **Production-minded engineering** — config-driven, CLI entrypoint, embedding &
  LLM caching, device auto-detection, model save/load, graceful error handling.
- **Reproducible** — pinned dependencies; one command runs the full pipeline.

## 🧱 Tech Stack

| Stage | Tools |
|-------|-------|
| Data handling | `pandas`, `openpyxl` |
| Vietnamese preprocessing | `pyvi` (word segmentation), regex, stopwords |
| Embeddings | `sentence-transformers`, `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` |
| Dimensionality reduction | `UMAP` |
| Clustering | `HDBSCAN` |
| Topic modeling | `BERTopic` (MMR, KeyBERTInspired) |
| Summarization | `llama-cpp-python` (GGUF) **or** Claude API (`anthropic`) |
| Config / CLI | `PyYAML`, `argparse` |

## 🗂️ Project Structure

```
config.yaml              # all parameters in one place
src/
  config.py              # load & validate config
  ingest.py              # read .xlsx/.csv → validated text column
  preprocess.py          # clean → word-segment → stopwords → dedup
  embed.py               # embeddings, device auto-detect, .npy cache
  topic_model.py         # UMAP + HDBSCAN + BERTopic, save/load, silhouette
  summarize.py           # pluggable LLM summary (llama_cpp | anthropic), cached
  pipeline.py            # CLI entrypoint wiring it all together
notebooks/viet_request.py  # original exploratory notebook (kept for reference)
data/                    # sample input
```

## 🚀 Getting Started

```bash
git clone https://github.com/vietIT2002/customer-feedback-analysis-project.git
cd customer-feedback-analysis-project
pip install -r requirements.txt
```

Configure `config.yaml` (data path, text column, summarizer backend), then run:

```bash
# full pipeline
python -m src.pipeline --config config.yaml

# topics + keywords only (skip the LLM step)
python -m src.pipeline --config config.yaml --no-summarize
```

**Summarizer options**
- `backend: llama_cpp` (offline) — download a GGUF model and point `model_path` to it.
  Keep `n_gpu_layers: 0` on CPU; raise it when a GPU is available.
- `backend: anthropic` — `pip install anthropic`, set `ANTHROPIC_API_KEY`; no GPU needed.

## 📊 Output

Results are written to `output/topic_representations.csv`:

| Topic | Keywords | Summary |
|-------|----------|---------|
| 0 | giao_hàng \| chậm \| ship | Khách phàn nàn về tốc độ giao hàng chậm hơn cam kết. |
| 1 | chất_lượng \| vải \| đẹp | Phản hồi tích cực về chất lượng vải và kiểu dáng sản phẩm. |

*(Illustrative — actual topics depend on your dataset.)* The BERTopic model is also
saved to `output/bertopic_model/` for reuse, and a silhouette score is printed to
gauge cluster quality.

## 🎯 What This Project Demonstrates

- End-to-end **NLP pipeline design**: from messy raw text to actionable insight.
- Practical understanding of **embeddings, dimensionality reduction, and density-based
  clustering**, and how they combine in BERTopic.
- **Language-specific care** for Vietnamese (segmentation, stopwords) — and the ability
  to spot a subtle correctness bug (missing tokenization) and fix it.
- **Software engineering on ML code**: turning a notebook into a configurable,
  cached, testable, CLI-driven package with a swappable LLM backend.

## 📄 License

Released under the [MIT License](LICENSE).
