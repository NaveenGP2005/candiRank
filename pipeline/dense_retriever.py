"""
Stage 5 + 6: Dense semantic retrieval (sentence-transformers) + RRF fusion.
Pre-computation: run precompute.py to save the model locally first.
"""
import time
import numpy as np
from pathlib import Path
from collections import defaultdict


ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "bge_small_local"
JD_EMBEDDING_PATH = ARTIFACTS_DIR / "jd_embedding.npy"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ─── Dense retrieval ─────────────────────────────────────────────────────────

def load_model() -> "SentenceTransformer":
    """
    Load sentence-transformer model from local saved dir.
    Falls back to downloading from HuggingFace if not found locally.
    """
    from sentence_transformers import SentenceTransformer
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        model = SentenceTransformer(str(MODEL_DIR))
        print(f"[dense] Loaded local model from {MODEL_DIR}")
    else:
        print(f"[dense] Local model not found, loading {MODEL_NAME} from HuggingFace")
        model = SentenceTransformer(MODEL_NAME)
    return model


def embed_texts(texts: list[str], model, batch_size: int = 64) -> np.ndarray:
    """Embed texts using sentence-transformers, L2-normalized."""
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def get_or_compute_jd_embedding(jd_text: str, model) -> np.ndarray:
    """Load pre-computed JD embedding or compute and save it."""
    if JD_EMBEDDING_PATH.exists():
        emb = np.load(str(JD_EMBEDDING_PATH))
        print(f"[dense] Loaded pre-computed JD embedding {emb.shape}")
        return emb
    print("[dense] Computing JD embedding...")
    emb = embed_texts([jd_text], model)
    np.save(str(JD_EMBEDDING_PATH), emb)
    return emb


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def reciprocal_rank_fusion(
    rankings: list[list],
    k: int = 60,
    top_k: int = 500,
) -> list:
    """
    Merge multiple ranked lists via RRF.
    Each element of `rankings` is an ordered list of candidate indices (best first).
    Returns a merged list of candidate indices (best first), limited to top_k.
    """
    rrf_scores: dict = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    rrf_score_array = [rrf_scores[d] for d in merged[:top_k]]
    print(f"[rrf] Merged {len(rankings)} lists → top-{top_k} candidates")
    return merged[:top_k], rrf_score_array


if __name__ == "__main__":
    print("Run precompute.py first to export the ONNX model.")
    print(f"Expected ONNX model path: {ONNX_MODEL_DIR}")
