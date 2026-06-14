"""
Stage 2: Deterministic honeypot / fraud detection.
Runs BEFORE any expensive embedding — cheap rule-based filters.
Disqualification: >10% honeypots in top-100 = full submission invalidated.
"""
from datetime import datetime, date
from typing import Optional


# ─── helpers ────────────────────────────────────────────────────────────────

SENIOR_TITLE_KEYWORDS = {
    "senior", "lead", "principal", "staff", "director", "head",
    "manager", "architect", "chief", "vp", "president",
}

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "ltimindtree", "lti", "birlasoft",
    "niit technologies", "zensar", "persistent systems",
}


def _parse_date(d: Optional[str], fallback: date = None) -> Optional[date]:
    if d is None:
        return fallback
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(d[:len(fmt.replace("%Y", "0000").replace("%m", "00").replace("%d", "00"))], fmt).date()
        except Exception:
            pass
    # try splitting by -
    try:
        parts = d.split("-")
        return date(int(parts[0]), int(parts[1]) if len(parts) > 1 else 1, int(parts[2]) if len(parts) > 2 else 1)
    except Exception:
        return fallback


def _is_senior_title(title: str) -> bool:
    tl = title.lower()
    return any(kw in tl for kw in SENIOR_TITLE_KEYWORDS)


def _months_overlap(s1: date, e1: date, s2: date, e2: date) -> float:
    """Return overlap in months between two date ranges."""
    latest_start = max(s1, s2)
    earliest_end = min(e1, e2)
    if earliest_end <= latest_start:
        return 0.0
    return (earliest_end - latest_start).days / 30.0


# ─── detection functions ─────────────────────────────────────────────────────

def check_temporal_impossibility(candidate: dict) -> tuple[bool, str]:
    """
    Flag if a senior role overlaps significantly with undergraduate study,
    or multiple simultaneous full-time jobs exist.
    """
    today = date.today()
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])

    # Build education windows
    edu_windows = []
    for edu in education:
        s = edu.get("start_year")
        e = edu.get("end_year")
        degree = edu.get("degree", "").lower()
        if s and e and ("b." in degree or "bachelor" in degree or "b.e" in degree or "b.tech" in degree):
            edu_windows.append((date(s, 6, 1), date(e, 6, 1)))

    # Build career windows
    job_windows = []
    for job in career:
        s = _parse_date(job.get("start_date"), None)
        e = _parse_date(job.get("end_date"), today) if not job.get("is_current") else today
        if s and e:
            job_windows.append((s, e, job.get("title", ""), job.get("duration_months", 0)))

    # Check: senior role overlapping with undergrad by >12 months
    for js, je, jtitle, jdur in job_windows:
        for es, ee in edu_windows:
            overlap = _months_overlap(js, je, es, ee)
            if overlap > 12 and _is_senior_title(jtitle):
                return True, f"Senior role '{jtitle}' overlaps undergrad by {overlap:.0f} months"

    # Check: two simultaneous full-time jobs for >6 months
    for i in range(len(job_windows)):
        for j in range(i + 1, len(job_windows)):
            s1, e1, t1, _ = job_windows[i]
            s2, e2, t2, _ = job_windows[j]
            overlap = _months_overlap(s1, e1, s2, e2)
            if overlap > 6:
                return True, f"Simultaneous jobs: '{t1}' and '{t2}' for {overlap:.0f} months"

    return False, ""


def check_skill_hallucination(candidate: dict) -> tuple[bool, str]:
    """
    Flag if candidate claims advanced multi-year skill but it never appears
    in any career description or job title.
    """
    all_career_text = " ".join(
        (job.get("title", "") + " " + job.get("description", "")).lower()
        for job in candidate.get("career_history", [])
    )

    hallucinated = []
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        duration = skill.get("duration_months", 0)
        proficiency = skill.get("proficiency", "")
        if duration < 24 or proficiency not in ("advanced", "expert"):
            continue
        # Check if skill name (or any word in it) appears in career text
        skill_words = [w.lower() for w in name.split() if len(w) > 3]
        if not any(w in all_career_text for w in skill_words):
            hallucinated.append(name)

    if len(hallucinated) >= 3:
        return True, f"Hallucinated skills: {hallucinated[:5]}"
    return False, ""


def check_domain_incoherence(candidate: dict) -> tuple[bool, str]:
    """
    Flag radical domain jumps with no narrative explanation.
    e.g. graphic design → mechanical CAD → SaaS support → claims 3yr React.
    """
    profile = candidate.get("profile", {})
    summary = (profile.get("summary", "") + " " + profile.get("headline", "")).lower()

    NON_TECH_DOMAINS = {"design", "marketing", "sales", "hr", "finance", "accounting",
                        "legal", "operations", "customer support", "graphic", "mechanical"}
    TECH_SKILLS = {"python", "react", "tensorflow", "pytorch", "kubernetes",
                   "aws", "gcp", "azure", "docker", "ml", "ai", "llm"}

    career = candidate.get("career_history", [])
    titles_lower = [j.get("title", "").lower() for j in career]
    has_non_tech = any(any(nd in t for nd in NON_TECH_DOMAINS) for t in titles_lower)

    skills = candidate.get("skills", [])
    advanced_tech_skills = [
        s["name"] for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months", 0) > 24
        and any(ts in s["name"].lower() for ts in TECH_SKILLS)
    ]

    if has_non_tech and len(advanced_tech_skills) >= 3 and "engineer" not in summary:
        return True, f"Non-tech career + advanced tech skills claimed: {advanced_tech_skills[:3]}"

    return False, ""


def check_impossible_experience(candidate: dict) -> tuple[bool, str]:
    """
    Flag if total claimed experience far exceeds possible given graduation year.
    """
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)

    education = candidate.get("education", [])
    grad_years = [e.get("end_year") for e in education if e.get("end_year")]
    if not grad_years:
        return False, ""

    earliest_grad = min(grad_years)
    current_year = datetime.now().year
    max_possible_yoe = current_year - earliest_grad

    if yoe > max_possible_yoe + 2:  # 2yr buffer for early starts
        return True, f"Claims {yoe:.1f} YOE but graduated {earliest_grad} (max ~{max_possible_yoe})"
    return False, ""


# ─── main filter ─────────────────────────────────────────────────────────────

def filter_honeypots(candidates: list[dict], verbose: bool = False) -> tuple[list[dict], list[dict]]:
    """
    Returns (clean_candidates, flagged_honeypots).
    Flagged candidates get a honeypot_reason key for debugging.
    """
    clean = []
    flagged = []

    for c in candidates:
        reasons = []

        is_temp, reason = check_temporal_impossibility(c)
        if is_temp:
            reasons.append(f"TEMPORAL: {reason}")

        is_hall, reason = check_skill_hallucination(c)
        if is_hall:
            reasons.append(f"SKILL_HALLUCINATION: {reason}")

        is_incoherent, reason = check_domain_incoherence(c)
        if is_incoherent:
            reasons.append(f"DOMAIN_INCOHERENCE: {reason}")

        is_impossible, reason = check_impossible_experience(c)
        if is_impossible:
            reasons.append(f"IMPOSSIBLE_EXP: {reason}")

        if reasons:
            c["_honeypot_reasons"] = reasons
            flagged.append(c)
            if verbose:
                print(f"  [HONEYPOT] {c['candidate_id']}: {reasons[0]}")
        else:
            clean.append(c)

    print(f"[honeypot] {len(flagged)} flagged, {len(clean)} clean from {len(candidates)} total")
    return clean, flagged


if __name__ == "__main__":
    import json, sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"
    with open(path) as f:
        samples = json.load(f)
    clean, flagged = filter_honeypots(samples, verbose=True)
    print(f"\nFlagged candidates:")
    for c in flagged:
        print(f"  {c['candidate_id']}: {c['_honeypot_reasons']}")
