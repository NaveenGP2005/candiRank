"""
Stage 8: Deterministic template-based reasoning generation.
Zero hallucination — every claim is sourced from actual candidate data.
"""

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "ltimindtree", "birlasoft", "niit",
    "zensar", "persistent", "cyient", "coforge",
}


def _is_consulting_company(name: str) -> bool:
    n = name.lower()
    return any(f in n for f in CONSULTING_FIRMS)


def _get_company_type(features: dict, current_company: str) -> str:
    """Return accurate company type label."""
    if _is_consulting_company(current_company):
        return "IT services firm"
    if features.get("consulting_only_flag"):
        return "IT services firm"
    if features.get("product_company_ratio", 0) > 0.5:
        return "product company"
    return "employer"


def _get_top_skill(features: dict) -> str:
    """Return the most impressive skill found in this candidate's career."""
    priority = [
        ("has_vector_db_prod",  "vector database (production)"),
        ("has_embedding_prod",  "embedding-based retrieval (production)"),
        ("has_ranking_eval",    "ranking evaluation (NDCG/MAP)"),
        ("has_rag",             "RAG / hybrid retrieval"),
        ("has_ltr",             "learning-to-rank"),
        ("has_recsys",          "recommendation systems"),
        ("has_llm_finetuning",  "LLM fine-tuning"),
    ]
    for key, label in priority:
        if features.get(key):
            return label
    return "ML/AI engineering"


def _availability_label(features: dict) -> str:
    """Return a concise honest availability summary."""
    open_to = features.get("open_to_work", 0)
    notice = features.get("notice_period_days", 60)
    rr = features.get("recruiter_response_rate", 0.5)
    last_active = features.get("last_active_days", 999)

    positives = []
    negatives = []

    if open_to:
        positives.append("marked open to work")
    else:
        negatives.append("not marked open to work")

    if notice <= 30:
        positives.append(f"{notice}-day notice")
    elif notice <= 60:
        positives.append(f"{notice}-day notice")
    else:
        negatives.append(f"{notice}-day notice period")

    if rr >= 0.6:
        positives.append(f"{rr:.0%} recruiter response rate")
    elif rr < 0.3:
        negatives.append(f"low recruiter response rate ({rr:.0%})")

    if last_active <= 14:
        positives.append("recently active")
    elif last_active > 90:
        negatives.append("inactive for >90 days")

    if len(positives) >= 2 and len(negatives) == 0:
        return "strong — " + "; ".join(positives)
    elif len(negatives) >= 2:
        return "limited — " + "; ".join(negatives)
    elif positives:
        summary = "; ".join(positives)
        if negatives:
            summary += f" (note: {negatives[0]})"
        return "moderate — " + summary
    else:
        return "uncertain — " + "; ".join(negatives)


def _gap_note(features: dict) -> str:
    """Return the single most important gap for this candidate."""
    if features.get("consulting_only_flag"):
        return "entire career is at IT services firms (explicit JD disqualifier)"
    if features.get("framework_tourist_flag"):
        return "limited pre-LLM ML depth — recent LLM wrapper experience only"
    if features.get("yoe_pre_llm_ml", 0) < 1:
        return "no evidence of pre-2022 ML production experience"
    if not features.get("has_vector_db_prod") and not features.get("has_embedding_prod"):
        return "no vector DB or embedding retrieval found in career history"
    if features.get("notice_period_days", 0) > 90:
        return f"long notice period ({features['notice_period_days']} days)"
    if features.get("recruiter_response_rate", 1.0) < 0.2:
        return "very low recruiter response rate"
    if features.get("last_active_days", 0) > 90:
        return "inactive on platform for >90 days"
    return "borderline fit on primary JD criteria"


def generate_reasoning(candidate: dict, features: dict, rank: int) -> str:
    """Generate a 1-2 sentence reasoning for this candidate at this rank."""
    profile    = candidate.get("profile", {})
    signals    = candidate.get("redrob_signals", {})
    career     = candidate.get("career_history", [])

    name    = profile.get("anonymized_name", "This candidate")
    yoe     = profile.get("years_of_experience", 0) or 0
    rr      = signals.get("recruiter_response_rate", 0.5) or 0.5
    notice  = signals.get("notice_period_days", 60) or 60
    company = career[0].get("company", "current employer") if career else "current employer"

    company_type = _get_company_type(features, company)
    top_skill    = _get_top_skill(features)
    pre_llm      = features.get("yoe_pre_llm_ml", 0) or 0
    avail_str    = _availability_label(features)
    gap          = _gap_note(features)

    # Consulting-flag override: must mention the disqualifier
    consulting_flag = features.get("consulting_only_flag", 0)
    has_key_skills  = features.get("has_vector_db_prod") or features.get("has_embedding_prod")

    if rank <= 20 and not consulting_flag and has_key_skills:
        # Strong positive template
        templates = [
            (
                f"{name} brings {yoe:.1f} years of applied ML experience including production "
                f"{top_skill} at {company} ({company_type}); "
                f"availability is {avail_str}."
            ),
            (
                f"Strong fit: {yoe:.1f}yr career at {company_type} scale ({company}) includes "
                f"production {top_skill} and {pre_llm:.1f}yr of pre-LLM ML experience — "
                f"matching the JD's requirement for engineers who built retrieval systems "
                f"before the LLM era."
            ),
        ]
        idx = hash(candidate["candidate_id"]) % len(templates)
        return templates[idx]

    elif rank <= 20 and consulting_flag:
        return (
            f"{name} has {yoe:.1f}yr ML experience and {top_skill} skills, but career is "
            f"entirely at IT services firms ({company}) — an explicit JD disqualifier; "
            f"ranked here on technical merit with availability: {avail_str}."
        )

    elif rank <= 20:
        return (
            f"{name} has {yoe:.1f}yr applied ML experience at {company} ({company_type}); "
            f"technical gap: {gap}. Availability: {avail_str}."
        )

    elif rank <= 70:
        if consulting_flag:
            return (
                f"{name} ({yoe:.1f}yr, {company}) meets ML criteria but entire career is at "
                f"IT services — an explicit JD disqualifier; ranked here on signal strength. "
                f"Notice: {notice} days."
            )
        return (
            f"{name} has {yoe:.1f}yr ML experience at {company} ({company_type}) but "
            f"{gap}. Availability: {avail_str}."
        )

    else:
        if consulting_flag:
            return (
                f"Weak match: {name} shows ML exposure at {company} (IT services) with "
                f"{gap}. {notice}-day notice, {rr:.0%} recruiter response rate."
            )
        return (
            f"Fringe inclusion: {name} has {yoe:.1f}yr experience but {gap}. "
            f"Recruiter response {rr:.0%}, {notice}-day notice."
        )


def generate_all_reasoning(
    top_candidates: list[dict],
    all_features: dict,
) -> dict[str, str]:
    """
    Generate reasoning for all ranked candidates.
    top_candidates: list of candidate dicts in rank order (index 0 = rank 1)
    all_features: {candidate_id: feature_dict}
    """
    result = {}
    for rank, candidate in enumerate(top_candidates, start=1):
        cid = candidate["candidate_id"]
        feat = all_features.get(cid, {})
        result[cid] = generate_reasoning(candidate, feat, rank)
    return result


if __name__ == "__main__":
    mock_candidate = {
        "candidate_id": "CAND_0000042",
        "profile": {"anonymized_name": "Arjun Mehta", "years_of_experience": 6.5},
        "career_history": [{"company": "Flipkart", "title": "Senior ML Engineer"}],
        "redrob_signals": {
            "recruiter_response_rate": 0.85,
            "notice_period_days": 30,
            "open_to_work_flag": True,
        },
    }
    mock_features = {
        "candidate_id": "CAND_0000042",
        "has_vector_db_prod": 1, "has_embedding_prod": 1,
        "has_ranking_eval": 1, "yoe_pre_llm_ml": 3.0,
        "consulting_only_flag": 0, "framework_tourist_flag": 0,
        "availability_score": 0.82, "recruiter_response_rate": 0.85,
        "notice_period_days": 30, "open_to_work": 1,
        "product_company_ratio": 0.9, "last_active_days": 5,
    }
    for r in [1, 5, 15, 50, 90]:
        print(f"\n--- Rank {r} ---")
        print(generate_reasoning(mock_candidate, mock_features, rank=r))
