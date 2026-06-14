---
title: CandidateRank - Redrob AI Challenge
emoji: 🎯
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
short_description: Intelligent candidate ranking for Senior AI Engineer role
---

# 🎯 CandidateRank — Redrob Intelligent Candidate Discovery Pipeline

**Submission for the Redrob India Runs Data & AI Challenge**

## Overview

This system ranks 100,000 AI/ML candidates for a **Senior AI Engineer** role using an 8-stage CPU-bound pipeline that completes in **under 160 seconds** (well within the 5-minute constraint).

## Pipeline Stages

1. **Ingestion** — orjson streaming (100K candidates in ~4s)
2. **Honeypot Eradication** — Deterministic heuristics (temporal impossibility, impossible YOE)
3. **Feature Engineering** — FlashText keyword extraction + 23 behavioral signals
4. **BM25 Retrieval** — Lexical first-pass, top-600 shortlist
5. **Dense Embedding** — BAAI/bge-small-en-v1.5, cosine similarity
6. **RRF Fusion** — Reciprocal Rank Fusion (k=60)
7. **LightGBM Reranking** — LambdaMART with pseudo-labels
8. **Template NLG** — Deterministic, zero-hallucination reasoning

## Key Stats

- ⏱️ **Runtime:** 159.3s on CPU (54% of 5-min budget)
- 🚫 **Honeypots detected:** 11,511 hard-removed + 5,501 soft-penalized
- ✅ **Submission validated:** Pass
- 📊 **Output:** 100 ranked candidates with reasoning

## Usage

The demo shows precomputed rankings from the full 100K candidate dataset.
Use the **Rankings** tab to browse and filter. Use the **Candidate Deep Dive** to view details.
