---
title: candiRank
emoji: 🚀
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10.13
app_file: app.py
pinned: false
---

<div align="center">
  <img src="https://img.shields.io/badge/Status-Deployed-success?style=for-the-badge&logo=huggingface&logoColor=white" alt="Deployed on Hugging Face" />
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10" />
  <img src="https://img.shields.io/badge/Gradio-4.44.1-orange?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio" />
  
  <br />
  <br />

  <h1>🚀 candiRank: High-Precision Candidate Retrieval System</h1>
  
  <p><b>An ultra-fast, production-ready candidate ranking engine built for the Redrob India Runs Data and AI Challenge.</b></p>
  
  <h3>
    <a href="https://huggingface.co/spaces/NaveenGP2005/candiRank">🌐 Live Demo on Hugging Face Spaces</a>
  </h3>
</div>

---

## 🎯 Overview

**candiRank** is engineered specifically to identify **Senior AI Retrieval/Ranking Engineers**. The system ingests, filters, and processes 100,000 synthetic candidate profiles, outputting a high-confidence Top 100 Leaderboard in **under 4 minutes** on standard CPU hardware. 

We achieved this by building a highly optimized, multi-stage retrieval cascade that completely abandons slow "embed-everything" architectures in favor of a lightning-fast hybrid pipeline.

---

## 🏆 Key Architecture Highlights

Our architecture balances extreme speed with rigorous recall, relying on several core innovations:

### 1. Robust Honeypot & Fraud Filter `O(N)`
* **Chronological Validation:** Uses strict checks (e.g., date parsing, overlapping full-time employment vs. undergraduate studies) and heuristic YOE anomaly detection to catch synthetic anomalies.
* **Smart Soft-Penalties:** Rather than broadly penalizing non-traditional paths (like freelance work during college), only mathematically impossible timelines are hard-removed (dropping rejection rates from 11.5% to **3.1%**). Suspicious profiles receive an automated soft penalty.

### 2. FlashText Feature Engineering `O(N)`
* **Aho-Corasick Automaton:** Extracts over 20 specific technical features across 5 core verticals (Vector DBs, Search/Retrieval, Ranking/Evaluation, ML Frameworks, MLOps) instantly.
* **Contextual Extraction:** Extracts features **exclusively from job descriptions** (ignoring the standalone 'skills' list). This guarantees candidates only receive ranking points if they actually *shipped* the technology in production.

### 3. Lexical First-Pass (BM25)
* Instead of embedding 100,000 candidates (which takes hours on a CPU), an expanded BM25 lexical index handles the first-pass retrieval, fetching the **Top 1000 candidates in ~13 seconds**. 
* The query is highly engineered to include semantic synonyms (`marketplace ranking`, `matching engines`, `search relevance`) to ensure excellent candidates without standard buzzwords survive the initial cutoff.

### 4. Semantic Dense Retrieval (BGE-Small)
* The Top 1000 candidates are embedded dynamically using `BAAI/bge-small-en-v1.5` (~160 seconds).
* This provides deep semantic understanding to evaluate whether a candidate's actual responsibilities match the true intent of the Senior AI Engineer Job Description.

### 5. Reciprocal Rank Fusion & JD-Calibrated Heuristics
* **The "Feature Collapse" Discovery:** Initial iterations used a weak-supervised LightGBM model. However, sparse but high-value JD features (like production vector DB experience) collapsed below the decision tree split thresholds, causing the model to over-index on generic YOE.
* **The Fix:** We pivoted to a highly calibrated heuristic reranker anchored by a **Reciprocal Rank Fusion (RRF)** score (merging Lexical and Semantic ranks) with precise, JD-aligned multipliers:
  * *Explicit Vector DB production experience (+3.0)*
  * *Ranking / NDCG / LTR experience (+2.0)*
* **Audit Proven:** A raw text audit of our submitted Top 20 candidates confirmed **19/20 have explicit Information Retrieval / Search experience**, and **18/20 have explicitly built Ranking systems**.

### 6. Natural Language Generator (NLG)
* The final pipeline outputs deterministic, highly interpretable justifications for *why* a candidate was ranked, calling out both strengths (e.g., YOE, Product Company background) and specific technical/availability gaps.

---

## ⚡ Performance Summary

candiRank is heavily optimized for CPU bottlenecks. On a standard CPU instance (16GB RAM):

| Pipeline Stage | Runtime | Component Type |
|----------------|---------|----------------|
| **Data Ingestion** | `4.8s` | I/O Bound |
| **Honeypot Filter** | `13.0s` | Pure Python |
| **Feature Extraction** | `26.7s` | FlashText |
| **BM25 Retrieval** | `12.8s` | Lexical Index |
| **BGE Embeddings** | `164.4s` | PyTorch CPU (Top 1000) |
| **Rerank & NLG** | `1.1s` | Numpy Heuristics |
| **Total Runtime** | **`~214.6s`** | **Budget: `290.0s`** |

---

## 🛠 Local Setup & Execution

### 1. Precompute Models
Run this once to download the local `BAAI/bge-small-en-v1.5` model to the `artifacts/` folder:
```bash
python precompute.py
```

### 2. Generate the Submission
Run the complete pipeline against the full 100,000 candidate dataset:
```bash
python rank.py --candidates "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" --out submission.csv
```

### 3. Validate Output
Validate the generated CSV format against the competition rubric:
```bash
python "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" submission.csv
```

---

## 🎨 Interactive Gradio Demo

We've built a full Gradio web dashboard to explore the candidates dynamically, view the leaderboard, and inspect the specific scoring reasoning for each candidate.

You can view the **[Live Deployment on Hugging Face Spaces](https://huggingface.co/spaces/NaveenGP2005/candiRank)**, or run it locally:

```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

---

<div align="center">
  <p><i>Built with precision for the Redrob India Runs Data and AI Challenge.</i></p>
</div>
