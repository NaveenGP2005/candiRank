"""
precompute.py — Run ONCE before ranking.
1. Saves BAAI/bge-small-en-v1.5 locally (tries ONNX backend, falls back to PyTorch)
2. Pre-embeds the JD
3. Trains LightGBM on sample candidates with pseudo-labels

Usage:
    python precompute.py [--sample path/to/sample_candidates.json]
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_DIR = ARTIFACTS_DIR / "bge_small_local"
JD_EMBEDDING_PATH = ARTIFACTS_DIR / "jd_embedding.npy"
MODEL_PATH = ARTIFACTS_DIR / "lgb_model.lgb"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

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


def get_st_model(path=None):
    """Load sentence-transformer model from path or HuggingFace."""
    from sentence_transformers import SentenceTransformer
    src = str(path) if path else MODEL_NAME
    return SentenceTransformer(src)


def save_model():
    """Download and save the model locally for offline use during ranking."""
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print(f"[precompute] Model already at {MODEL_DIR}, skipping download.")
        return

    print(f"[precompute] Downloading {MODEL_NAME} from HuggingFace...")
    start = time.time()
    model = get_st_model()
    model.save(str(MODEL_DIR))
    print(f"[precompute] Model saved in {time.time()-start:.1f}s → {MODEL_DIR}")


def precompute_jd_embedding():
    """Pre-embed the JD and save to disk."""
    if JD_EMBEDDING_PATH.exists():
        print(f"[precompute] JD embedding already exists, skipping.")
        return

    print("[precompute] Computing JD embedding...")
    model = get_st_model(MODEL_DIR)
    emb = model.encode([JD_TEXT], normalize_embeddings=True)
    np.save(str(JD_EMBEDDING_PATH), emb)
    print(f"[precompute] JD embedding saved: shape={emb.shape}")


def train_lgb_model(sample_path: str):
    """Train LightGBM on sample candidates with pseudo-labels."""
    if MODEL_PATH.exists():
        print(f"[precompute] LGB model already exists, skipping training.")
        return

    print(f"[precompute] Training LightGBM on {sample_path}...")
    with open(sample_path) as f:
        samples = json.load(f)

    from pipeline.honeypot_filter import filter_honeypots
    from pipeline.feature_engineer import extract_all_features
    from pipeline.reranker import generate_pseudo_label, build_feature_matrix, train_reranker

    clean, _ = filter_honeypots(samples)
    features = extract_all_features(clean)

    y_pseudo = np.array([generate_pseudo_label(f) for f in features])
    print(f"[precompute] Pseudo-label distribution: {np.bincount(y_pseudo)}")

    empty_scores = {f["candidate_id"]: 0.0 for f in features}
    X = build_feature_matrix(features, empty_scores, empty_scores, empty_scores)

    train_reranker(X, y_pseudo, save=True)
    print("[precompute] LightGBM model trained and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        default=r"dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PRECOMPUTE PHASE — run once before ranking")
    print("=" * 60)

    print("\n[1/3] Saving model locally...")
    save_model()

    print("\n[2/3] Pre-computing JD embedding...")
    precompute_jd_embedding()

    print("\n[3/3] Training LightGBM model...")
    train_lgb_model(args.sample)

    print("\n✓ Precompute complete. Run rank.py to generate submission.")
