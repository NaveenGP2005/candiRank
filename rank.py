"""
rank.py — Main ranking entry point.
Must complete in ≤5 minutes on CPU, 16GB RAM, no network.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
import argparse
import csv
import time
import sys
from pathlib import Path

import numpy as np

# ─── Timing budget tracker ────────────────────────────────────────────────────
BUDGET_SECONDS = 290  # leave 10s buffer from 5-min limit
t_start = time.time()


def elapsed():
    return time.time() - t_start


def check_budget(stage: str):
    e = elapsed()
    remaining = BUDGET_SECONDS - e
    print(f"[timer] After {stage}: {e:.1f}s elapsed, {remaining:.1f}s remaining")
    if remaining < 30:
        print(f"[WARN] Less than 30s remaining! Stage: {stage}")


# ─── JD text (same as in precompute.py) ──────────────────────────────────────
JD_TEXT = """
Senior AI Engineer role at Redrob AI. Requires production experience with 
embeddings-based retrieval systems (sentence-transformers, BGE, E5, OpenAI embeddings).
Production experience with vector databases: Pinecone, Qdrant, Weaviate, Milvus, FAISS, 
OpenSearch, Elasticsearch. Strong Python. Hands-on experience designing evaluation 
frameworks for ranking systems: NDCG, MRR, MAP, offline-to-online correlation, 
A/B test interpretation. 5-9 years experience. Applied ML at product companies.
Pre-LLM era machine learning production experience required. Not pure research, 
not consulting-only. LLM fine-tuning (LoRA, QLoRA, PEFT) is a plus. 
Learning-to-rank models (XGBoost) is a plus. Hybrid retrieval, dense retrieval, 
BM25, reranking, cross-encoder. Recommendation systems. India preferred: Pune, Noida, 
Delhi, Hyderabad, Mumbai. GitHub activity valued. Open source contributions valued.
"""

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "lgb_model.lgb"
ONNX_MODEL_DIR = ARTIFACTS_DIR / "bge_small_int8_onnx"


def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top-k-bm25", type=int, default=1000, help="BM25 first-pass top-K")
    parser.add_argument("--top-k-rrf", type=int, default=500, help="RRF shortlist size")
    parser.add_argument("--demo", action="store_true", help="Demo mode: only process first 500 candidates")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"REDROB CANDIDATE RANKER")
    print(f"Input: {args.candidates}")
    print(f"Output: {args.out}")
    print(f"{'='*60}\n")

    # ── Stage 1: Load ─────────────────────────────────────────────────────────
    print("[Stage 1] Loading candidates...")
    from pipeline.ingest import load_candidates, build_text
    candidates = load_candidates(args.candidates)
    if args.demo:
        candidates = candidates[:500]
        print(f"[demo] Truncated to {len(candidates)} candidates")
    check_budget("Stage 1: Load")

    # ── Stage 2: Honeypot filter ──────────────────────────────────────────────
    print("\n[Stage 2] Filtering honeypots...")
    from pipeline.honeypot_filter import filter_honeypots
    clean_candidates, flagged = filter_honeypots(candidates)
    check_budget("Stage 2: Honeypot")

    # ── Stage 3: Feature engineering ─────────────────────────────────────────
    print("\n[Stage 3] Extracting features...")
    from pipeline.feature_engineer import extract_all_features
    all_features = extract_all_features(clean_candidates)
    feat_by_id = {f["candidate_id"]: f for f in all_features}
    cand_by_id = {c["candidate_id"]: c for c in clean_candidates}
    check_budget("Stage 3: Features")

    # ── Stage 4: BM25 retrieval ───────────────────────────────────────────────
    print(f"\n[Stage 4] BM25 first-pass (top-{args.top_k_bm25})...")
    from pipeline.bm25_retriever import build_bm25_index, bm25_retrieve, JD_QUERY_TEXT
    corpus_texts = [build_text(c) for c in clean_candidates]
    bm25_index = build_bm25_index(corpus_texts)
    top_k_bm25 = min(args.top_k_bm25, len(clean_candidates))
    bm25_indices, bm25_raw_scores = bm25_retrieve(bm25_index, JD_QUERY_TEXT, top_k=top_k_bm25)
    # Map to candidate IDs
    bm25_candidates = [clean_candidates[i] for i in bm25_indices]
    bm25_texts = [corpus_texts[i] for i in bm25_indices]
    # Normalize BM25 scores to [0,1]
    bm25_max = bm25_raw_scores.max() if bm25_raw_scores.max() > 0 else 1.0
    bm25_norm_scores = bm25_raw_scores / bm25_max
    bm25_score_by_id = {
        clean_candidates[idx]["candidate_id"]: float(score)
        for idx, score in zip(bm25_indices, bm25_norm_scores)
    }
    check_budget("Stage 4: BM25")

    # ── Stage 5: Dense embedding ──────────────────────────────────────────
    print(f"\n[Stage 5] Dense embedding {len(bm25_candidates)} candidates...")
    from pipeline.dense_retriever import load_model, embed_texts, get_or_compute_jd_embedding

    dense_model = load_model()
    jd_emb = get_or_compute_jd_embedding(JD_TEXT, dense_model)
    cand_emb = embed_texts(bm25_texts, dense_model, batch_size=64)
    semantic_raw = (cand_emb @ jd_emb.T).flatten()
    dense_sorted_idx = np.argsort(semantic_raw)[::-1]
    dense_cand_ids = [bm25_candidates[i]["candidate_id"] for i in dense_sorted_idx]
    semantic_score_by_id = {
        bm25_candidates[i]["candidate_id"]: float(semantic_raw[i])
        for i in range(len(bm25_candidates))
    }
    check_budget("Stage 5: Dense embed")

    # ── Stage 6: Reciprocal Rank Fusion ──────────────────────────────────────
    print(f"\n[Stage 6] RRF fusion → top-{args.top_k_rrf}...")
    from pipeline.dense_retriever import reciprocal_rank_fusion
    bm25_ranking = [c["candidate_id"] for c in bm25_candidates]
    rrf_top_ids, rrf_raw_scores = reciprocal_rank_fusion(
        [bm25_ranking, dense_cand_ids],
        k=60,
        top_k=args.top_k_rrf,
    )
    rrf_score_by_id = {cid: score for cid, score in zip(rrf_top_ids, rrf_raw_scores)}
    # Normalize RRF scores
    rrf_max = max(rrf_raw_scores) if rrf_raw_scores else 1.0
    rrf_score_by_id_norm = {k: v / rrf_max for k, v in rrf_score_by_id.items()}

    # Get features for RRF shortlist
    rrf_features = [feat_by_id[cid] for cid in rrf_top_ids if cid in feat_by_id]
    check_budget("Stage 6: RRF")

    # ── Stage 7: Reranking ──────────────────────────────────────────
    print(f"\n[Stage 7] Reranking {len(rrf_features)} candidates...")
    from pipeline.reranker import score_candidates

    lgb_model = None
    if MODEL_PATH.exists():
        try:
            from pipeline.reranker import load_reranker
            lgb_model = load_reranker()
        except Exception as e:
            print(f"[reranker] Could not load model: {e}. Using fallback.")

    top_100_raw = score_candidates(
        model=lgb_model,
        feature_dicts=rrf_features,
        bm25_scores=bm25_score_by_id,
        semantic_scores=semantic_score_by_id,
        rrf_scores=rrf_score_by_id_norm,
        top_k=150,  # Get 150, we'll filter to 100 after penalty
    )

    # Post-processing: apply consulting penalty and normalize scores
    # This is a safety net since LightGBM is trained on only 41 samples
    CONSULTING_FIRMS = {
        "tcs", "tata consultancy", "infosys", "wipro", "accenture",
        "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
        "hexaware", "mindtree", "ltimindtree", "birlasoft",
    }
    def is_consulting_current(cid):
        c = cand_by_id.get(cid, {})
        current = (c.get("career_history") or [{}])[0].get("company", "").lower()
        return any(f in current for f in CONSULTING_FIRMS)

    penalized = []
    for cid, score in top_100_raw:
        feat = feat_by_id.get(cid, {})
        s = score
        # Heavy penalty for consulting-only candidates
        if feat.get("consulting_only_flag"):
            s *= 0.4
        # Moderate penalty if current employer is consulting
        elif is_consulting_current(cid):
            s *= 0.7
        # Penalty for very long notice (>90 days)
        notice = feat.get("notice_period_days", 60)
        if notice > 90:
            s *= max(0.7, 1.0 - (notice - 90) / 180)
        penalized.append((cid, s))

    # Sort again after penalties and take top 100
    penalized.sort(key=lambda x: x[1], reverse=True)
    top_100_penalized = penalized[:100]

    # Normalize scores to [0, 100]
    if top_100_penalized:
        max_s = top_100_penalized[0][1]
        min_s = top_100_penalized[-1][1]
        score_range = max_s - min_s if max_s != min_s else 1.0
        top_100 = [
            (cid, round(100.0 * (s - min_s) / score_range, 4))
            for cid, s in top_100_penalized
        ]
    else:
        top_100 = top_100_penalized

    check_budget("Stage 7: Rerank")

    # ── Stage 8: Generate reasoning ───────────────────────────────────────────
    print("\n[Stage 8] Generating reasoning...")
    from pipeline.reasoning import generate_reasoning

    rows = []
    for rank, (cid, score) in enumerate(top_100, start=1):
        candidate = cand_by_id.get(cid, {"candidate_id": cid, "profile": {}, "career_history": [], "redrob_signals": {}})
        feat = feat_by_id.get(cid, {"candidate_id": cid})
        reasoning = generate_reasoning(candidate, feat, rank)
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning,
        })
    check_budget("Stage 8: Reasoning")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    print(f"\n[Output] Writing {len(rows)} rows to {args.out}...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    total = elapsed()
    print(f"\n{'='*60}")
    print(f"✓ Done! {len(rows)} candidates ranked in {total:.1f}s")
    print(f"  Output: {args.out}")
    print(f"  Honeypots filtered: {len(flagged)}")
    print(f"  Budget used: {total:.1f}s / {BUDGET_SECONDS}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
