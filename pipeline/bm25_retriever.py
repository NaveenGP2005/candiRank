"""
Stage 4: BM25 lexical first-pass retrieval.
Reduces 100K candidates to top-5000 for dense embedding.
"""
import time
import numpy as np
from rank_bm25 import BM25Okapi


# ─── JD query — manually crafted to capture INTENT not just keywords ─────────
# The JD explicitly says: reason about what the JD means, not what it says.
# We include both explicit terms AND semantic equivalents.

JD_QUERY_TEXT = """
senior AI engineer machine learning production embeddings retrieval ranking
vector database hybrid search dense retrieval semantic search
sentence transformers BGE E5 embedding model FAISS Pinecone Qdrant Weaviate 
Milvus OpenSearch Elasticsearch HNSW approximate nearest neighbor
NDCG MAP MRR evaluation framework ranking metrics offline evaluation
A/B testing learning to rank LambdaMART XGBoost gradient boosting
recommendation system search relevance retrieval quality
Python production deployment scaling inference optimization
product company not consulting not services
pre-LLM machine learning experience NLP information retrieval
fine-tuning LoRA PEFT QLoRA instruction tuning
5 years 6 years 7 years 8 years experience applied ML AI
Pune Noida Delhi Hyderabad Mumbai India
open source GitHub contributions technical blog papers
LLM integration RAG retrieval augmented generation
cross-encoder reranker bi-encoder
""".strip()


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def build_bm25_index(corpus_texts: list[str]) -> BM25Okapi:
    """Build BM25 index over candidate texts."""
    start = time.time()
    tokenized = [tokenize(t) for t in corpus_texts]
    index = BM25Okapi(tokenized, k1=1.5, b=0.75)
    print(f"[bm25] Index built in {time.time()-start:.1f}s")
    return index


def bm25_retrieve(
    index: BM25Okapi,
    query_text: str = JD_QUERY_TEXT,
    top_k: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (top_k_indices, top_k_scores) sorted best-first.
    Indices are relative to the corpus passed to build_bm25_index.
    """
    start = time.time()
    query_tokens = tokenize(query_text)
    scores = np.array(index.get_scores(query_tokens))
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]
    print(f"[bm25] Retrieved top-{top_k} in {time.time()-start:.1f}s | "
          f"max_score={top_scores[0]:.3f} min_score={top_scores[-1]:.3f}")
    return top_indices, top_scores


if __name__ == "__main__":
    import json
    from pipeline.ingest import build_text

    with open(r"dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json") as f:
        samples = json.load(f)

    corpus_texts = [build_text(c) for c in samples]
    index = build_bm25_index(corpus_texts)
    indices, scores = bm25_retrieve(index, top_k=10)
    print("\nTop-10 BM25 results:")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        cid = samples[idx]["candidate_id"]
        title = samples[idx]["profile"].get("current_title", "?")
        print(f"  {rank}. {cid} [{title}] score={score:.3f}")
