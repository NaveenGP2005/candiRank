"""
Stage 7: LightGBM LambdaMART reranker.
Uses pseudo-labels (weak supervision) since no ground-truth labels exist.
"""
import numpy as np
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "lgb_model.lgb"

FEATURE_COLS = [
    "bm25_score", "semantic_score", "rrf_score",
    "honeypot_soft_flag",
    "yoe_total", "yoe_applied_ml", "yoe_pre_llm_ml",
    "seniority_tier", "avg_tenure_months", "product_company_ratio",
    "consulting_only_flag",
    "has_vector_db_prod", "has_embedding_prod", "has_ranking_eval",
    "has_ltr", "has_llm_finetuning", "has_rag", "has_recsys",
    "framework_tourist_flag",
    "location_score", "willing_to_relocate",
    "recruiter_response_rate", "open_to_work", "notice_period_days",
    "github_activity_score", "profile_completeness",
    "interview_completion_rate", "offer_acceptance_rate",
    "recency_score", "last_active_days", "avg_assessment",
    "availability_score",
]


def generate_pseudo_label(feat: dict) -> int:
    """
    Rule-based relevance score (0-4) for weak supervision.
    Mirrors the 5-tier grading of the JD.
    """
    score = 0.0

    # Core technical skills in career (not just skills list)
    score += 2.0 * feat.get("has_vector_db_prod", 0)
    score += 2.0 * feat.get("has_embedding_prod", 0)
    score += 1.5 * feat.get("has_ranking_eval", 0)
    score += 1.0 * feat.get("has_rag", 0)
    score += 0.5 * feat.get("has_ltr", 0)
    score += 0.5 * feat.get("has_recsys", 0)
    score += 0.5 * feat.get("has_llm_finetuning", 0)

    # YOE in applied ML (sweet spot: 4-8 years)
    yoe_ml = feat.get("yoe_applied_ml", 0)
    if 4 <= yoe_ml <= 8:
        score += 2.0
    elif 2 <= yoe_ml < 4:
        score += 1.0
    elif yoe_ml > 8:
        score += 1.5  # senior but ok

    # Pre-LLM ML experience (JD explicitly requires this)
    if feat.get("yoe_pre_llm_ml", 0) > 2:
        score += 1.5
    elif feat.get("yoe_pre_llm_ml", 0) > 1:
        score += 0.5

    # Product company experience
    score += 1.0 * feat.get("product_company_ratio", 0)

    # Disqualifiers (heavy penalties)
    score -= 3.0 * feat.get("consulting_only_flag", 0)
    score -= 2.0 * feat.get("framework_tourist_flag", 0)

    # Location
    score += feat.get("location_score", 0.3) * 0.5

    # Availability multiplier (not additive — multiplicative)
    availability = feat.get("availability_score", 0.5)
    score *= (0.3 + 0.7 * availability)

    # GitHub bonus (JD explicitly values external validation)
    gh = feat.get("github_activity_score", 0)
    if gh > 50:
        score += 0.5

    return int(np.clip(round(score), 0, 4))


def build_feature_matrix(
    feature_dicts: list[dict],
    bm25_scores: dict,
    semantic_scores: dict,
    rrf_scores: dict,
) -> np.ndarray:
    """
    Build feature matrix X from list of feature dicts + retrieval scores.
    bm25_scores / semantic_scores / rrf_scores: {candidate_id: score}
    Returns X (n_candidates × n_features).
    """
    rows = []
    for feat in feature_dicts:
        cid = feat["candidate_id"]
        row = []
        for col in FEATURE_COLS:
            if col == "bm25_score":
                row.append(bm25_scores.get(cid, 0.0))
            elif col == "semantic_score":
                row.append(semantic_scores.get(cid, 0.0))
            elif col == "rrf_score":
                row.append(rrf_scores.get(cid, 0.0))
            else:
                row.append(float(feat.get(col, 0.0)))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def train_reranker(X: np.ndarray, y_pseudo: np.ndarray, save: bool = True):
    """Train LightGBM LambdaMART on pseudo-labels."""
    import lightgbm as lgb

    # For LambdaMART, treat all candidates as one group
    group = [len(X)]

    train_data = lgb.Dataset(
        X, label=y_pseudo,
        group=group,
        feature_name=FEATURE_COLS,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 50],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "n_estimators": 300,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }

    model = lgb.train(params, train_data, num_boost_round=300)

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        model.save_model(str(MODEL_PATH))
        print(f"[reranker] Model saved to {MODEL_PATH}")

    return model


def load_reranker():
    """Load pre-trained LightGBM model."""
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(MODEL_PATH))
    print(f"[reranker] Loaded model from {MODEL_PATH}")
    return model


def score_candidates(
    model,
    feature_dicts: list[dict],
    bm25_scores: dict,
    semantic_scores: dict,
    rrf_scores: dict,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    Score candidates and return top-k as [(candidate_id, score), ...] best first.
    Falls back to weighted formula if no trained model.
    """
    # Force weighted linear combination to preserve JD-specific weights
    # (LightGBM trained on 50 samples was ignoring sparse features like Vector DBs)
    raw_scores = weighted_score_fallback(feature_dicts, bm25_scores, semantic_scores, rrf_scores)

    # Sort and return top-k
    sorted_idx = np.argsort(raw_scores)[::-1][:top_k]
    results = [
        (feature_dicts[i]["candidate_id"], float(raw_scores[i]))
        for i in sorted_idx
    ]
    return results


def weighted_score_fallback(
    feature_dicts: list[dict],
    bm25_scores: dict,
    semantic_scores: dict,
    rrf_scores: dict,
) -> np.ndarray:
    """
    Fallback scoring when LightGBM model not available.
    Pure weighted combination of features.
    """
    scores = []
    for feat in feature_dicts:
        cid = feat["candidate_id"]
        s = 0.0
        # Retrieval scores
        s += 0.25 * rrf_scores.get(cid, 0.0) * 100  # normalize
        s += 0.15 * semantic_scores.get(cid, 0.0) * 10

        # Skill presence in career
        s += 3.0 * feat.get("has_vector_db_prod", 0)
        s += 3.0 * feat.get("has_embedding_prod", 0)
        s += 2.0 * feat.get("has_ranking_eval", 0)
        s += 1.5 * feat.get("has_rag", 0)

        # YOE
        yoe_ml = feat.get("yoe_applied_ml", 0)
        s += min(yoe_ml / 5, 2.0)
        s += min(feat.get("yoe_pre_llm_ml", 0) / 2, 1.5)

        # Product company
        s += feat.get("product_company_ratio", 0) * 1.0

        # Penalties
        s -= 4.0 * feat.get("consulting_only_flag", 0)
        s -= 3.0 * feat.get("framework_tourist_flag", 0)

        # Availability multiplier
        avail = feat.get("availability_score", 0.5)
        s *= (0.3 + 0.7 * avail)

        scores.append(s)
    return np.array(scores, dtype=np.float32)


if __name__ == "__main__":
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Model path: {MODEL_PATH}")
