# Automated Customer Feedback Analysis for GenZ Fashion Store 👗🧠

This project leverages BERTopic and LLM-based summarization to automatically analyze Vietnamese customer feedback for a fashion store. The goal is to identify key topics, visualize them, and generate human-readable summaries to support decision-making.

---

## 📌 Problem Statement

GenZ Fashion Store receives a large number of customer feedback entries daily. Manually classifying and analyzing these messages is slow and inefficient.

**Objective**: Automate the topic modeling and summarization of feedback to improve product & service decisions, save time, and reduce manual labor.

---

## 🔍 Features

- Clean and preprocess Vietnamese feedback (stopword removal, normalization)
- Generate semantic embeddings using `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- Dimensionality reduction with UMAP
- Clustering with HDBSCAN
- Topic modeling with BERTopic
- Generate human-like summaries using a local LLaMA model (`vinallama-7b-chat`)
- Export results to CSV
- Optional: visualize document-topic distribution

---

## 🧱 Tech Stack

| Task                        | Tool / Library                          |
|-----------------------------|------------------------------------------|
| Data handling               | `pandas`, `openpyxl`                     |
| Text preprocessing          | `re`, `pyvi`, custom stopword filter     |
| Embedding                   | `sentence-transformers`, HuggingFace     |
| Dimensionality reduction    | `UMAP`                                   |
| Clustering                  | `HDBSCAN`                                |
| Topic modeling              | `BERTopic`                               |
| Summarization               | `llama-cpp-python` with `vinallama` GGUF |
| Visualization               | `plotly` (via BERTopic)                  |

---

## 🗂️ Project Structure

```
config.yaml              # all parameters (data, models, UMAP/HDBSCAN, summarizer)
src/
  config.py              # loads & validates config.yaml
  ingest.py              # read .xlsx/.csv -> validated `text` column
  preprocess.py          # clean -> word-segment (pyvi) -> stopwords -> dedup
  embed.py               # SentenceTransformer, device auto-detect, npy cache
  topic_model.py         # UMAP + HDBSCAN + BERTopic, save/load, silhouette
  summarize.py           # pluggable LLM summary: llama_cpp | anthropic (cached)
  pipeline.py            # CLI entrypoint that wires everything together
notebooks/viet_request.py  # original exploratory Colab notebook (kept for reference)
```

## 🚀 How to Run

### 1. Clone and install
```bash
git clone https://github.com/vietIT2002/customer-feedback-analysis-project.git
cd customer-feedback-analysis-project
pip install -r requirements.txt
```

### 2. Configure
Edit `config.yaml` — set `data.path`, `data.text_column`, and pick the
summarizer backend:

- `summarizer.backend: llama_cpp` (default, offline) — download the GGUF model:
  ```bash
  wget https://huggingface.co/vilm/vinallama-7b-chat-GGUF/resolve/main/vinallama-7b-chat_q5_0.gguf
  ```
  On CPU-only machines keep `llama_cpp.n_gpu_layers: 0`; raise it (e.g. 35) when a GPU is available.
- `summarizer.backend: anthropic` (no GPU needed) — `pip install anthropic` and
  set `ANTHROPIC_API_KEY` in your environment.

### 3. Run the pipeline
```bash
python -m src.pipeline --config config.yaml
# keywords only, skip the LLM step:
python -m src.pipeline --config config.yaml --no-summarize
```

Outputs land in `output/`: `topic_representations.csv` (Topic | Keywords | Summary),
the saved BERTopic model, the embedding cache, and the LLM response cache.

---

<details>
<summary>Legacy: original Colab notebook</summary>

The initial exploratory version lives at `notebooks/viet_request.py` (a Colab
export). The modular pipeline above supersedes it; the notebook is kept for
reference only.
</details>
