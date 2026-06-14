"""
Stage 3: Feature engineering.
Uses FlashText (Aho-Corasick) for fast keyword extraction + rule-based features.
Produces a feature dict per candidate for LightGBM reranking.
"""
import re
from datetime import datetime, date
from typing import Optional

# ─── Tech synonym dictionary ──────────────────────────────────────────────────
# Maps surface forms → canonical form for standardization

TECH_SYNONYMS: dict[str, list[str]] = {
    # Vector DBs
    "pinecone": ["Pinecone"],
    "qdrant": ["Qdrant"],
    "weaviate": ["Weaviate"],
    "milvus": ["Milvus"],
    "faiss": ["FAISS", "Facebook AI Similarity Search"],
    "opensearch": ["OpenSearch", "Open Search"],
    "elasticsearch": ["ElasticSearch", "Elastic Search", "ES"],
    "chroma": ["Chroma", "ChromaDB"],
    "pgvector": ["pgvector", "pg_vector"],
    # Embedding models
    "sentence_transformers": ["sentence-transformers", "sentence transformers", "SentenceTransformers"],
    "bge": ["BGE", "BAAI/bge"],
    "openai_embeddings": ["OpenAI embeddings", "text-embedding-ada", "text-embedding-3"],
    # Ranking / IR
    "bm25": ["BM25", "BM-25", "Okapi BM25"],
    "ndcg": ["NDCG", "Normalized Discounted Cumulative Gain"],
    "map_metric": ["MAP", "Mean Average Precision"],
    "mrr": ["MRR", "Mean Reciprocal Rank"],
    "learning_to_rank": ["LTR", "Learning to Rank", "LambdaMART", "RankNet", "LambdaRank"],
    # ML frameworks
    "pytorch": ["PyTorch", "torch"],
    "tensorflow": ["TensorFlow", "TF", "Keras"],
    "scikit_learn": ["scikit-learn", "sklearn", "sci-kit learn"],
    "xgboost": ["XGBoost", "xgb"],
    "lightgbm": ["LightGBM", "lgbm"],
    # LLM / GenAI
    "rag": ["RAG", "Retrieval Augmented Generation", "Retrieval-Augmented Generation"],
    "lora": ["LoRA", "Low-Rank Adaptation"],
    "qlora": ["QLoRA"],
    "peft": ["PEFT", "Parameter Efficient Fine-tuning"],
    "langchain": ["LangChain", "LangGraph"],
    # Infra
    "kubernetes": ["Kubernetes", "k8s"],
    "docker": ["Docker"],
    "airflow": ["Apache Airflow", "Airflow"],
    "spark": ["Apache Spark", "PySpark", "Spark"],
    "kafka": ["Apache Kafka", "Kafka"],
    # Cloud
    "aws": ["AWS", "Amazon Web Services"],
    "gcp": ["GCP", "Google Cloud", "Google Cloud Platform"],
    "azure": ["Azure", "Microsoft Azure"],
    # Databases
    "snowflake": ["Snowflake"],
    "bigquery": ["BigQuery"],
    "postgres": ["PostgreSQL", "Postgres"],
    # Hybrid search
    "hybrid_search": ["hybrid search", "hybrid retrieval"],
    "dense_retrieval": ["dense retrieval", "dense passage retrieval", "DPR"],
    "reranking": ["reranking", "re-ranking", "cross-encoder reranking"],
    # Misc AI
    "onnx": ["ONNX", "Open Neural Network Exchange"],
    "huggingface": ["HuggingFace", "Hugging Face", "transformers"],
    "mlflow": ["MLflow", "ML Flow"],
    "recommendation_system": ["recommendation system", "recommender system", "RecSys"],
}

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "ltimindtree", "birlasoft", "niit",
    "zensar", "persistent", "cyient", "coforge",
}

PRODUCT_COMPANIES_KEYWORDS = {
    # If company name contains any of these, it's likely a product company
    "labs", "ai", "tech", "io", "inc", "corp", "systems",
}

LLM_ERA_START = date(2022, 1, 1)


def _build_flashtext_processor():
    """Build and return a FlashText KeywordProcessor with all tech synonyms."""
    try:
        from flashtext import KeywordProcessor
        kp = KeywordProcessor(case_sensitive=False)
        for canonical, variants in TECH_SYNONYMS.items():
            kp.add_keyword(canonical, canonical)
            for v in variants:
                kp.add_keyword(v, canonical)
        return kp
    except ImportError:
        print("[WARNING] flashtext not installed, falling back to simple matching")
        return None


_KP = None  # lazy-init


def get_keyword_processor():
    global _KP
    if _KP is None:
        _KP = _build_flashtext_processor()
    return _KP


def _extract_keywords(text: str) -> set[str]:
    kp = get_keyword_processor()
    if kp:
        return set(kp.extract_keywords(text))
    # Fallback: simple case-insensitive search
    found = set()
    tl = text.lower()
    for canonical, variants in TECH_SYNONYMS.items():
        if any(v.lower() in tl for v in variants) or canonical in tl:
            found.add(canonical)
    return found


def _parse_date(d: Optional[str], fallback=None):
    if d is None:
        return fallback
    try:
        parts = d.split("-")
        return date(int(parts[0]), int(parts[1]) if len(parts) > 1 else 1, 1)
    except Exception:
        return fallback


def _is_consulting(company_name: str) -> bool:
    cl = company_name.lower()
    return any(f in cl for f in CONSULTING_FIRMS)


def _is_ml_ai_role(title: str, description: str = "") -> bool:
    keywords = {"machine learning", "ml", "ai ", "data science", "nlp",
                "deep learning", "neural", "ranking", "retrieval",
                "recommendation", "search", "embeddings", "llm"}
    text = (title + " " + description).lower()
    return any(kw in text for kw in keywords)


# ─── main feature extractor ──────────────────────────────────────────────────

def extract_features(candidate: dict) -> dict:
    """Extract all features for one candidate. Returns a flat dict."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})
    today = date.today()

    # ── build full text ───────────────────────────────────────────────────────
    summary = profile.get("summary", "")
    headline = profile.get("headline", "")
    career_text = " ".join(
        (j.get("title", "") + " " + j.get("description", "")) for j in career
    )
    all_text = f"{summary} {headline} {career_text}"
    skill_names_text = " ".join(s.get("name", "") for s in skills)
    full_text = all_text + " " + skill_names_text

    # ── keyword extraction from CAREER text (must appear in actual work) ─────
    career_keywords = _extract_keywords(career_text)

    # ── years of experience ───────────────────────────────────────────────────
    yoe_total = profile.get("years_of_experience", 0) or 0

    # YOE in applied ML/AI roles at any company
    yoe_applied_ml = 0.0
    yoe_pre_llm_ml = 0.0
    avg_tenure_list = []
    consulting_months = 0
    product_months = 0

    for job in career:
        jstart = _parse_date(job.get("start_date"))
        jend = _parse_date(job.get("end_date"), today) if not job.get("is_current") else today
        if not jstart or not jend:
            continue
        dur_months = max(0, (jend - jstart).days / 30)
        avg_tenure_list.append(dur_months)

        company = job.get("company", "")
        title = job.get("title", "")
        desc = job.get("description", "")

        if _is_consulting(company):
            consulting_months += dur_months
        else:
            product_months += dur_months

        if _is_ml_ai_role(title, desc):
            yoe_applied_ml += dur_months / 12
            # Pre-LLM portion
            pre_llm_end = min(jend, LLM_ERA_START)
            if pre_llm_end > jstart:
                yoe_pre_llm_ml += (pre_llm_end - jstart).days / 365

    avg_tenure_months = (sum(avg_tenure_list) / len(avg_tenure_list)) if avg_tenure_list else 0
    total_months = consulting_months + product_months
    product_company_ratio = (product_months / total_months) if total_months > 0 else 0.0
    consulting_only_flag = int(consulting_months > 0 and product_months == 0)

    # ── seniority tier ────────────────────────────────────────────────────────
    current_title = profile.get("current_title", "").lower()
    if any(w in current_title for w in ("principal", "staff", "director", "vp", "head", "chief")):
        seniority_tier = 4
    elif any(w in current_title for w in ("senior", "lead", "architect")):
        seniority_tier = 3
    elif any(w in current_title for w in ("engineer", "scientist", "analyst", "developer")):
        seniority_tier = 2
    else:
        seniority_tier = 1

    # ── specific skill presence in CAREER (not just skills list) ─────────────
    has_vector_db_prod = int(bool(career_keywords & {
        "pinecone", "qdrant", "weaviate", "milvus", "faiss",
        "opensearch", "elasticsearch", "chroma", "pgvector"
    }))
    has_embedding_prod = int(bool(career_keywords & {
        "sentence_transformers", "bge", "openai_embeddings",
        "dense_retrieval", "rag"
    }))
    has_ranking_eval = int(bool(career_keywords & {
        "ndcg", "map_metric", "mrr", "learning_to_rank"
    }))
    has_ltr = int("learning_to_rank" in career_keywords or "xgboost" in career_keywords)
    has_llm_finetuning = int(bool(career_keywords & {"lora", "qlora", "peft"}))
    has_rag = int("rag" in career_keywords or "hybrid_search" in career_keywords)
    has_recsys = int("recommendation_system" in career_keywords)

    # ── framework tourist flag ────────────────────────────────────────────────
    # Claims recent LLM work but no pre-LLM ML depth
    langchain_in_career = "langchain" in career_keywords
    framework_tourist_flag = int(
        langchain_in_career
        and yoe_pre_llm_ml < 1.0
        and yoe_applied_ml < 2.0
    )

    # ── location scoring ──────────────────────────────────────────────────────
    location = (profile.get("location", "") + " " + profile.get("country", "")).lower()
    INDIA_PREF = {"pune", "noida", "delhi", "gurgaon", "gurugram", "hyderabad",
                  "bangalore", "bengaluru", "mumbai", "chennai"}
    if any(city in location for city in ("pune", "noida", "delhi", "ncr", "gurgaon")):
        location_score = 1.0
    elif any(city in location for city in INDIA_PREF):
        location_score = 0.8
    elif "india" in location:
        location_score = 0.6
    else:
        location_score = 0.3

    # ── behavioral signals ────────────────────────────────────────────────────
    recruiter_response_rate = signals.get("recruiter_response_rate", 0.5) or 0.5
    open_to_work = int(signals.get("open_to_work_flag", False) or False)
    notice_period_days = signals.get("notice_period_days", 60) or 60
    github_activity_score = signals.get("github_activity_score", -1)
    github_score_clean = github_activity_score if github_activity_score >= 0 else 0
    profile_completeness = signals.get("profile_completeness_score", 50) or 50
    interview_completion_rate = signals.get("interview_completion_rate", 0.5) or 0.5
    offer_acceptance_rate = signals.get("offer_acceptance_rate", -1)
    offer_rate_clean = offer_acceptance_rate if offer_acceptance_rate >= 0 else 0.5
    willing_to_relocate = int(signals.get("willing_to_relocate", False) or False)

    # Last active recency (0=inactive, 1=very recent)
    last_active_str = signals.get("last_active_date", "")
    last_active_days = 999
    if last_active_str:
        la = _parse_date(last_active_str)
        if la:
            last_active_days = (today - la).days
    recency_score = max(0.0, 1.0 - last_active_days / 180)

    # Avg assessment score
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}
    avg_assessment = (sum(assessment_scores.values()) / len(assessment_scores)) if assessment_scores else 50.0

    # ── availability composite ────────────────────────────────────────────────
    notice_penalty = max(0.0, 1.0 - max(0, notice_period_days - 30) / 90)
    availability_score = (
        0.30 * recruiter_response_rate
        + 0.20 * open_to_work
        + 0.20 * recency_score
        + 0.15 * notice_penalty
        + 0.15 * interview_completion_rate
    )

    return {
        "candidate_id": candidate["candidate_id"],
        # Honeypot soft flag (suspicious but not hard-filtered)
        "honeypot_soft_flag": int(candidate.get("_honeypot_soft_flag", False)),
        # YOE
        "yoe_total": yoe_total,
        "yoe_applied_ml": yoe_applied_ml,
        "yoe_pre_llm_ml": yoe_pre_llm_ml,
        # Career quality
        "seniority_tier": seniority_tier,
        "avg_tenure_months": avg_tenure_months,
        "product_company_ratio": product_company_ratio,
        "consulting_only_flag": consulting_only_flag,
        # Skill presence in career text
        "has_vector_db_prod": has_vector_db_prod,
        "has_embedding_prod": has_embedding_prod,
        "has_ranking_eval": has_ranking_eval,
        "has_ltr": has_ltr,
        "has_llm_finetuning": has_llm_finetuning,
        "has_rag": has_rag,
        "has_recsys": has_recsys,
        "framework_tourist_flag": framework_tourist_flag,
        # Location
        "location_score": location_score,
        "willing_to_relocate": willing_to_relocate,
        # Behavioral
        "recruiter_response_rate": recruiter_response_rate,
        "open_to_work": open_to_work,
        "notice_period_days": notice_period_days,
        "github_activity_score": github_score_clean,
        "profile_completeness": profile_completeness,
        "interview_completion_rate": interview_completion_rate,
        "offer_acceptance_rate": offer_rate_clean,
        "recency_score": recency_score,
        "last_active_days": last_active_days,
        "avg_assessment": avg_assessment,
        "availability_score": availability_score,
    }


def extract_all_features(candidates: list[dict]) -> list[dict]:
    """Extract features for all candidates. Returns list of feature dicts."""
    import time
    start = time.time()
    features = [extract_features(c) for c in candidates]
    print(f"[features] Extracted {len(features):,} feature rows in {time.time()-start:.1f}s")
    return features


if __name__ == "__main__":
    import json
    with open(r"dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json") as f:
        samples = json.load(f)
    feats = extract_all_features(samples[:5])
    for f in feats:
        print(f"\n{f['candidate_id']}:")
        print(f"  yoe_total={f['yoe_total']}, yoe_applied_ml={f['yoe_applied_ml']:.1f}, "
              f"has_vector_db={f['has_vector_db_prod']}, availability={f['availability_score']:.2f}")
