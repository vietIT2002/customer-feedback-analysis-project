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

## 🚀 How to Run

### 1. Clone this repo:
```bash
git clone https://github.com/yourusername/customer-feedback-analysis-project.git
cd customer-feedback-analysis-project
