---
title: candiRank
emoji: 🚀
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🚀 candiRank: High-Precision Candidate Retrieval System

**candiRank** is an ultra-fast, production-ready candidate ranking engine built for the Redrob India Runs Data and AI Challenge. Designed specifically to identify **Senior AI Retrieval/Ranking Engineers**, the system processes 100,000 synthetic candidate profiles and outputs a high-confidence Top 100 in **under 4 minutes** on standard CPU hardware.

---

## 🏆 Key Architecture Highlights

The system relies on a heavily optimized, multi-stage cascade to balance speed and recall.

### 1. Robust Honeypot & Fraud Filter (O(N))
* Uses strict chronological checks (e.g., date parsing, overlapping full-time employment vs. undergraduate studies) and heuristic YOE anomaly detection.
* **Smart Soft-Penalties:** Rather than broadly penalizing non-traditional paths (like freelance work during college), only mathematically impossible timelines are hard-removed (down from 11.5% to **3.1%**). Suspicious profiles (e.g. advanced skills never mentioned in career histories) receive a soft penalty.

### 2. FlashText Feature Engineering (O(N))
* A fast Aho-Corasick automaton extracts over 20 specific technical features across 5 core verticals (Vector DBs, Search/Retrieval, Ranking/Evaluation, ML Frameworks, MLOps).
* Extracts features **exclusively from job descriptions** (ignoring the 'skills' list), guaranteeing that candidates only receive points if they actually shipped the technology in production.

### 3. Lexical First-Pass (BM25)
* Instead of embedding 100,000 candidates (which takes hours on a CPU), an expanded BM25 index handles the first-pass retrieval, fetching the **Top 1000** candidates in **~13 seconds**. 
* The query is highly engineered to include semantic synonyms (`marketplace ranking`, `matching engines`, `search relevance`) to ensure candidates without standard buzzwords survive the cutoff.

### 4. Semantic Dense Retrieval (BGE-Small)
* The Top 1000 candidates are embedded on-the-fly using `BAAI/bge-small-en-v1.5` (~160 seconds).
* This provides deep semantic understanding to evaluate whether a candidate's actual responsibilities match the intent of the Senior AI Engineer JD.

### 5. Reciprocal Rank Fusion & JD-Calibrated Heuristics
* **The "Feature Collapse" Discovery:** Initial tests used a weak-supervised LightGBM LambdaMART model trained on a small sample set. However, sparse but high-value JD features (like production vector DBs) collapsed below the decision tree split thresholds, causing the model to over-index on generic YOE.
* **The Fix:** We pivoted to a highly calibrated heuristic reranker. It anchors around a **Reciprocal Rank Fusion (RRF)** score (merging Lexical and Semantic ranks) and applies precise, JD-aligned multipliers for:
  * Explicit Vector DB production experience (+3.0)
  * Ranking / NDCG / LTR experience (+2.0)
  * Pre-LLM era ML experience
* **Audit Proven:** This guarantees the top results aren't just generic ML engineers. A raw text audit of the Top 20 submitted candidates confirmed **19/20 have explicit Information Retrieval / Search experience**, and **18/20 have explicitly built Ranking systems**.

### 6. Natural Language Generator (NLG)
* The final pipeline outputs deterministic, highly interpretable justifications for *why* a candidate was ranked, calling out both strengths (e.g., YOE, Product Company background) and technical/availability gaps.

---

## ⚡ Performance Summary

On a standard CPU instance (16GB RAM):

| Pipeline Stage | Runtime | Component Type |
|----------------|---------|----------------|
| **Data Ingestion** | 4.8s | I/O Bound |
| **Honeypot Filter** | 13.0s | Pure Python |
| **Feature Extraction** | 26.7s | FlashText |
| **BM25 Retrieval** | 12.8s | Lexical Index |
| **BGE Embeddings** | 164.4s | PyTorch CPU (Top 1000) |
| **Rerank & NLG** | 1.1s | Numpy Heuristics |
| **Total Runtime** | **~214.6s** | **Budget: 290.0s** |

---

## 🛠 How to Run

### 1. Precompute Models & Embeddings
Run this once to download the local `BAAI/bge-small-en-v1.5` model to the `artifacts/` folder:
```bash
python precompute.py
```

### 2. Generate the Submission
Run the complete pipeline against the 100,000 dataset:
```bash
python rank.py --candidates "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" --out submission.csv
```

### 3. Validate
Validate the output CSV formatting:
```bash
python "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" submission.csv
```

---

## 🎨 Interactive Gradio Demo

Want to explore the candidates dynamically? A full Gradio web application is included to view the leaderboard, search individual profiles, and inspect the scoring mechanism.

```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

---
*Built for the Redrob India Runs Data and AI Challenge.*
