# System Documentation — Customer Feedback Analysis

> A complete walkthrough of the system: **business context → technology →
> processing pipeline → how to run → result interpretation → real-world
> deployment.** Written for both engineers and store operators.

---

## 1. Business context & goals

**Problem.** A Gen-Z fashion store receives hundreds of pieces of feedback every
day (chat, comments, forms). Reading and classifying them by hand is slow and
inconsistent, so the store reacts late to what customers actually care about
(sizing, fabric, delivery, payment, website bugs…).

**Goal.** Automatically:
1. **Group** feedback into topics *without any pre-labeling*.
2. **Summarize** each topic into a readable sentence for decision-makers.
3. Deliver results as a **table + chart** so the team can act fast.

**Who uses it & why.**

| Role | Uses it to |
|---|---|
| Owner / manager | See at a glance "what are customers complaining about most this week" |
| Customer support / ops | Spot recurring issues (size exchange, slow delivery) to fix |
| Marketing / product | Understand demand (styles, fabrics) to improve the catalog |

**Input:** an Excel/CSV file with one feedback-text column (default column `noidung`).
**Output:** a list of topics + keywords + summaries; a trained model for reuse.

---

## 2. System overview

The system is an **unsupervised NLP pipeline** combined with an **LLM** for the
summary step, with two ways to run it:

- **CLI** (`python -m src.pipeline`) — batch runs, good for automation.
- **Web dashboard** (`streamlit run app.py`) — for non-technical users.

All parameters live in `config.yaml`; the source code is split into modules under `src/`.

---

## 3. Architecture & technology

```
 Excel / CSV
     │
     ▼
 ┌──────────────┐  ingest.py      read & validate the data column
 ├──────────────┤
 │ preprocess   │  preprocess.py  clean → WORD-SEGMENT (pyvi) → stopwords → dedup
 ├──────────────┤
 │ embed        │  embed.py       SimCSE-PhoBERT → 768-dim vectors (.npy cache, auto CPU/GPU)
 ├──────────────┤
 │ topic_model  │  topic_model.py UMAP (reduce) → HDBSCAN (cluster) → BERTopic (label)
 ├──────────────┤
 │ summarize    │  summarize.py   summarize each topic with an LLM (3 backends, cached)
 └──────────────┘
     │
     ▼
 topic_representations.csv  +  dashboard
```

| Component | Technology | Why |
|---|---|---|
| Data handling | pandas, openpyxl | Standard for tables/Excel |
| Vietnamese segmentation | **pyvi** | PhoBERT needs word-segmented input (see 4.2) |
| Embeddings | sentence-transformers + **SimCSE-PhoBERT** | Strong Vietnamese semantic model |
| Dim. reduction | **UMAP** | Preserves cluster structure when compressing |
| Clustering | **HDBSCAN** | Finds the number of clusters; has a "noise" concept (-1) |
| Topic modeling | **BERTopic** | Combines embeddings + clusters + keyword extraction |
| Summarization | LLM (llama.cpp / Claude / OpenAI-compatible) | Turns keywords into readable sentences |
| UI | **Streamlit + Altair** | Fast dashboard, reuses the Python modules |

---

## 4. Processing pipeline in detail

### 4.1 Ingest (`ingest.py`)
Reads `.xlsx`/`.csv`, checks the text column exists, renames it to `text`, casts
to string. Raises a clear error if the file or column is missing.

### 4.2 Preprocess (`preprocess.py`)
Order **matters**:
1. Light cleaning (lowercase, collapse whitespace).
2. **Word segmentation with pyvi**: `chất lượng` → `chất_lượng`.
3. Remove stopwords on the *segmented* tokens.
4. Deduplicate + drop too-short rows.

> ⚠️ **Key point:** PhoBERT / SimCSE-PhoBERT are trained on **word-segmented**
> text. Skipping this step (as the original notebook did — it imported `tokenize`
> but never called it) corrupts the embeddings and noticeably degrades clustering.
> This was a real bug, found and fixed.

### 4.3 Embed (`embed.py`)
Uses `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` to produce a **768-dim** vector
per feedback. Auto-detects the device (`cuda`→`cpu`) and **caches to `.npy`** so it
isn't recomputed.

### 4.4 Topic model (`topic_model.py`)
- **UMAP**: compress 768 → 5 dims, preserving cluster structure (cosine metric).
- **HDBSCAN**: density-based clustering; points in no cluster become **noise (-1)**.
- **BERTopic**: combine embeddings + clusters, extract representative keywords
  (MMR + KeyBERTInspired).
- Saves the model to `output/bertopic_model/` and prints a **silhouette score**.

### 4.5 Summarize (`summarize.py`)
For each topic: take its keywords + a few representative docs → build a prompt →
call the LLM → a Vietnamese description. **Cached by prompt hash** (no re-calling
already-summarized topics).

Three pluggable backends (set `summarizer.backend` in config):

| Backend | When to use | Requires |
|---|---|---|
| `llama_cpp` | Offline, data never leaves the machine | A GGUF model + (ideally) a GPU |
| `anthropic` | High quality, no GPU | `ANTHROPIC_API_KEY` + `pip install anthropic` |
| `openai_compatible` | **Fast & free** (OpenRouter/Mistral/Groq) | An API key (no extra package) |

> Performance lesson: running a 7B LLM on a **weak CPU** is very slow (20–40 min).
> Switching to an **API** backend summarizes everything in **seconds**.

---

## 5. Configuration (`config.yaml`)

Every parameter is centralized. Main sections:

- `data`: file path, text column name.
- `preprocess`: stopwords, minimum length, dedup.
- `embedding`: model name, device, batch size, cache path.
- `umap` / `hdbscan`: reduction & clustering params (tune to data size).
- `bertopic`: language, keyword diversity.
- `summarizer`: backend, docs per topic, `max_tokens`, prompt, cache, per-backend config.
- `output`: directory, CSV name, model location.

---

## 6. How to run

### 6.1 Prerequisites
- **Python 3.10–3.12** (3.13+ may lack wheels for `numba`/`torch`; 3.12 recommended).
- ~3 GB disk for dependencies; the embedding model (~500 MB) downloads on first run.

### 6.2 Install
```bash
git clone https://github.com/vietIT2002/customer-feedback-analysis-project.git
cd customer-feedback-analysis-project

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 6.3 Configure
Edit `config.yaml`: set `data.path`, `data.text_column`, and pick a summarizer
backend (`summarizer.backend`).

### 6.4 Choose a summarizer backend

**Option A — OpenAI-compatible API (fastest, free tiers; recommended)**
```yaml
summarizer:
  backend: openai_compatible
```
```bash
# OpenRouter (free models) — or use the Mistral block in config.yaml
export OPENROUTER_API_KEY="sk-or-..."     # Windows PowerShell: $env:OPENROUTER_API_KEY="sk-or-..."
```

**Option B — Anthropic (Claude)**
```yaml
summarizer: { backend: anthropic }
```
```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Option C — Offline GGUF (no API, needs the model file)**
```yaml
summarizer: { backend: llama_cpp }
```
```bash
# download the model referenced by llama_cpp.model_path in config.yaml
wget https://huggingface.co/vilm/vinallama-7b-chat-GGUF/resolve/main/vinallama-7b-chat_q5_0.gguf
```
Keep `llama_cpp.n_gpu_layers: 0` on CPU; raise it (e.g. 35) when a GPU is present.

> 🔒 Never hardcode an API key in `config.yaml` or commit it. Use environment
> variables (the `.env` file is git-ignored).

### 6.5 Run the CLI
```bash
# full pipeline (with summaries)
python -m src.pipeline --config config.yaml

# topics + keywords only (skip the LLM step — fast, no key/model needed)
python -m src.pipeline --config config.yaml --no-summarize
```
Outputs: `output/topic_representations.csv`, the saved model, and caches.

### 6.6 Run the dashboard
```bash
# set your API key first (same as above) if you want AI summaries
streamlit run app.py
```
Open `http://localhost:8501`, upload an `.xlsx`/`.csv`, pick the text column, click
**Phân tích**, and view metrics, the chart, the topic table, summaries, and
representative feedback (with CSV export). The API key is read from the environment,
never entered in the UI.

### 6.7 Troubleshooting
- **`429` / `503` from the API** — free tiers are rate-limited or briefly busy; just
  rerun (completed topics are cached, so only failures retry).
- **LLM very slow** — you're running a local model on CPU; use an API backend instead.
- **Install fails on `numba`/`torch`** — your Python is too new; use Python 3.12.

---

## 7. Reading the results (business view)

| Concept | Business meaning |
|---|---|
| **Topic** | A group of feedback sharing one concern (e.g. "delivery", "size exchange") |
| **Keywords** | The defining words of a topic — a quick read of what it's about |
| **Summary** | An AI-generated sentence — for reporting / decisions |
| **Feedback count** | How "hot" a topic is — prioritize the large ones |
| **Noise (-1)** | Feedback too unique to group yet; lots of noise = sparse data |
| **Silhouette** | Cluster-quality score (higher = better separated); ~0.5 is good |

**Real example** (run on the 146-row sample → 13 topics, silhouette ≈ 0.50):
delivery/shipping time · payment (Momo/ZaloPay) · size exchange · fabric quality ·
website bugs (font/color) · shipping fees & membership · styles & outfit pairing…

---

## 8. Recommended business workflow

```
(1) Collect feedback  →  (2) Export Excel/CSV  →  (3) Run analysis (weekly)
        →  (4) Review topics & summaries on the dashboard
        →  (5) Prioritize the biggest topics  →  (6) Assign fixes (warehouse / support / web / product)
        →  (7) Next week: did that topic shrink?
```

Because summarization scales with the **number of topics (dozens), not the number
of feedback items**, cost and time stay bounded even as data grows.

---

## 9. Real-world deployment & scaling

**Recommended for a single store:** run as a **periodic batch** (cron/Airflow),
use an **API** backend for summaries (removes the GPU requirement), store results
in a **database**, and have the dashboard read from the DB so users see results
instantly.

Split the **hot path** from the **cold path**:
- **Hot (real-time, fast):** new feedback → embed + assign to an existing topic →
  milliseconds, *no LLM call*.
- **Cold (batch, can be slow):** periodically re-cluster + summarize *only new or
  changed topics*.

Upgrade directions: track **topic drift** (new topics appearing), add **sentiment**
per topic, and alert when a negative topic spikes.

---

## 10. Limitations

- Quality depends on data: slang, typos, and missing context reduce accuracy.
- Small datasets → more noise; tune `min_cluster_size` / `min_samples`.
- Offline LLMs need strong hardware; APIs require considering customer-data privacy.
- Topics are unsupervised — operators should still skim to confirm the labels.
