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
| Summarization | `llama-cpp-python` (GGUF), Claude API, or any OpenAI-compatible API |
| Dashboard | `Streamlit`, `Altair` |
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
  summarize.py           # pluggable LLM summary (llama_cpp | anthropic | openai_compatible)
  pipeline.py            # CLI entrypoint wiring it all together
app.py                   # Streamlit dashboard (upload → analyze → view)
docs/SYSTEM.md           # full system documentation (business + technical)
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
- `backend: openai_compatible` — OpenRouter / Mistral / Groq (free tiers). No extra
  package; just set the API key env var named in `config.yaml`. **Fastest path** —
  summarizes all topics via API in seconds, no GPU.

### 🖥️ Web dashboard (for non-technical users)

A Streamlit app lets store staff upload a feedback file and read the topics &
summaries — no command line needed:

```bash
streamlit run app.py
```

Upload an `.xlsx`/`.csv`, pick the text column, click **Phân tích**, and view the
topic table, a bar chart of topic sizes, AI summaries, and representative feedback
per topic (with CSV export). The API key is read from the environment, never the UI.

## 📊 Output

Results are written to `output/topic_representations.csv` (`Topic | Keywords | Summary`).
On the bundled sample (**146 feedback → 13 topics, silhouette ≈ 0.50**) the pipeline
surfaced coherent, real themes, e.g.:

| Topic | Keywords | What customers raised |
|-------|----------|-----------------------|
| 6 | giao_hàng · ship · đặt | Delivery time, order cancellation, address changes, checking goods on arrival. |
| 7 | zalopay · thanh_toán · ví | Payment methods (Momo, ZaloPay, ViettelPay) and pay-on-inspection. |
| 4 | size · đổi · sai | Size-exchange policy when the wrong/ill-fitting size is bought. |
| 9 | web · filter · font | Website bugs: broken fonts on mobile, wrong product colors, filter issues. |

The trained BERTopic model is saved to `output/bertopic_model/` for reuse, and a
silhouette score is printed to gauge cluster quality. *(Topics depend on your dataset.)*

## 🎯 What This Project Demonstrates

- End-to-end **NLP pipeline design**: from messy raw text to actionable insight.
- Practical understanding of **embeddings, dimensionality reduction, and density-based
  clustering**, and how they combine in BERTopic.
- **Language-specific care** for Vietnamese (segmentation, stopwords) — and the ability
  to spot a subtle correctness bug (missing tokenization) and fix it.
- **Software engineering on ML code**: turning a notebook into a configurable,
  cached, testable, CLI-driven package with a swappable LLM backend.

## 📚 Documentation

A full system document — **business context → technology → processing pipeline →
how to run → result interpretation → real-world deployment** — is in
[docs/SYSTEM.md](docs/SYSTEM.md).

## 📄 License

Released under the [MIT License](LICENSE).
