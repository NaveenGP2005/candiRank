"""
Stage 1: High-performance ingestion using orjson + gzip streaming.
Loads all 100K candidates into memory efficiently.
"""
import gzip
import time
from pathlib import Path

import orjson


def load_candidates(path: str) -> list[dict]:
    """Load candidates.jsonl or candidates.jsonl.gz using orjson."""
    p = Path(path)
    start = time.time()

    candidates = []
    if p.suffix == ".gz":
        with gzip.open(p, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(orjson.loads(line))
    else:
        with open(p, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(orjson.loads(line))

    elapsed = time.time() - start
    print(f"[ingest] Loaded {len(candidates):,} candidates in {elapsed:.1f}s")
    return candidates


def build_text(candidate: dict) -> str:
    """
    Concatenate all meaningful text fields for BM25 / embedding.
    Order: summary, career descriptions, skills, certifications.
    """
    parts = []

    profile = candidate.get("profile", {})
    if profile.get("summary"):
        parts.append(profile["summary"])
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("current_title"):
        parts.append(profile["current_title"])

    for job in candidate.get("career_history", []):
        if job.get("title"):
            parts.append(job["title"])
        if job.get("description"):
            parts.append(job["description"])

    skill_names = [s["name"] for s in candidate.get("skills", []) if s.get("name")]
    if skill_names:
        parts.append(" ".join(skill_names))

    for cert in candidate.get("certifications", []):
        if cert.get("name"):
            parts.append(cert["name"])

    return " ".join(parts)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
    cands = load_candidates(path)
    print(f"First candidate: {cands[0]['candidate_id']}")
    print(f"Text preview: {build_text(cands[0])[:200]}")
