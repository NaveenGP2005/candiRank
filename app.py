"""
app.py — Gradio sandbox for Redrob Candidate Ranking Pipeline.
Submission requirement: hosted interactive demo.

Displays precomputed top-100 results from submission.csv with
interactive filtering, pipeline methodology explanation, and
candidate detail view.
"""
import gradio as gr
import csv
import json
import os
from pathlib import Path

# ─── Load precomputed results ─────────────────────────────────────────────────

SUBMISSION_CSV = Path("submission.csv")
SAMPLE_JSON = Path("dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json")


def load_results():
    """Load submission.csv into a list of dicts."""
    if not SUBMISSION_CSV.exists():
        return []
    with open(SUBMISSION_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_sample_index():
    """Build a candidate_id → candidate dict from sample JSON."""
    if not SAMPLE_JSON.exists():
        return {}
    try:
        with open(SAMPLE_JSON) as f:
            samples = json.load(f)
        return {c["candidate_id"]: c for c in samples}
    except Exception:
        return {}


RESULTS = load_results()
SAMPLE_INDEX = load_sample_index()

# ─── Helper functions ─────────────────────────────────────────────────────────

def format_results_table(rows, top_n=100):
    """Format results for Gradio Dataframe."""
    data = []
    for r in rows[:top_n]:
        score = float(r["score"])
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        data.append([
            int(r["rank"]),
            r["candidate_id"],
            f"{score:.2f}",
            f"{bar} {score:.1f}",
            r["reasoning"][:120] + "..." if len(r["reasoning"]) > 120 else r["reasoning"],
        ])
    return data


def get_candidate_detail(candidate_id: str):
    """Return detailed view for a specific candidate."""
    # Find in results
    result_row = next((r for r in RESULTS if r["candidate_id"] == candidate_id), None)
    if not result_row:
        return f"❌ Candidate `{candidate_id}` not found in top-100 results."

    lines = [
        f"## 🏆 Rank #{result_row['rank']} — {candidate_id}",
        f"**Score:** {float(result_row['score']):.2f} / 100",
        f"",
        f"### 📝 Reasoning",
        f"> {result_row['reasoning']}",
        f"",
    ]

    # Enrich with sample data if available
    if candidate_id in SAMPLE_INDEX:
        c = SAMPLE_INDEX[candidate_id]
        profile = c.get("profile", {})
        signals = c.get("redrob_signals", {})
        career = c.get("career_history", [])
        skills = c.get("skills", [])

        lines += [
            f"### 👤 Profile",
            f"- **Name:** {profile.get('anonymized_name', 'N/A')}",
            f"- **Title:** {profile.get('current_title', 'N/A')}",
            f"- **Location:** {profile.get('location', 'N/A')}",
            f"- **YOE:** {profile.get('years_of_experience', 'N/A')} years",
            f"",
            f"### 💼 Career (last 3 roles)",
        ]
        for job in career[:3]:
            lines.append(f"- **{job.get('title', '?')}** @ {job.get('company', '?')} ({job.get('start_date', '?')} – {job.get('end_date', 'present')})")

        lines += [
            f"",
            f"### 🛠 Skills",
            ", ".join(s.get("name", "") for s in skills[:15]),
            f"",
            f"### 📊 Behavioral Signals",
            f"- Recruiter response rate: {signals.get('recruiter_response_rate', 'N/A')}",
            f"- Open to work: {'✅' if signals.get('open_to_work_flag') else '❌'}",
            f"- Notice period: {signals.get('notice_period_days', 'N/A')} days",
            f"- GitHub activity score: {signals.get('github_activity_score', 'N/A')}",
            f"- Profile completeness: {signals.get('profile_completeness_score', 'N/A')}%",
            f"- Last active: {signals.get('last_active_date', 'N/A')}",
        ]
    else:
        lines.append("*Full profile data not available in demo (sample only includes 50 candidates)*")

    return "\n".join(lines)


def filter_and_display(top_n, min_score, search_text):
    """Filter results and return table data."""
    filtered = [
        r for r in RESULTS
        if float(r["score"]) >= min_score
        and (search_text.lower() in r["candidate_id"].lower()
             or search_text.lower() in r["reasoning"].lower()
             or not search_text)
    ]
    return format_results_table(filtered, top_n=int(top_n))


# ─── Pipeline methodology text ────────────────────────────────────────────────

METHODOLOGY = """
## 🏗️ Pipeline Architecture

This system implements an **8-stage CPU-bound ranking pipeline** that processes 100,000 candidates in under **160 seconds** (well within the 5-minute constraint).

---

### Stage 1 — High-Speed Ingestion
- **Library:** `orjson` + streaming `gzip`
- **Speed:** 100K candidates in ~4 seconds
- **Format:** JSONL (100K lines × ~3KB each)

### Stage 2 — Honeypot Eradication
Deterministic heuristics to remove fraudulent profiles **before** any scoring:
- **TEMPORAL impossibility:** Senior role overlapping heavily with undergraduate enrollment
- **IMPOSSIBLE YOE:** Claims more experience than time since graduation allows
- **Soft-flagged** (penalized, not removed): Skill hallucination, domain incoherence

> ⚠️ **Critical:** The hackathon automatically disqualifies submissions with >10% honeypots in top-100. Our filter is applied FIRST, before any retrieval.

### Stage 3 — Feature Engineering
- **FlashText (Aho-Corasick):** 60+ tech keyword synonyms standardized in one pass
- **23 behavioral signals** extracted: YOE, tenure, product vs consulting ratio, availability
- **Career-text-only claims:** Skills only count if mentioned in *career history*, not just skills list

### Stage 4 — BM25 Lexical Retrieval
- **Library:** `rank-bm25` (BM25Okapi, k1=1.5, b=0.75)
- **Top-600** candidates selected from ~88K clean candidates
- JD query crafted to capture **intent** (not just surface keywords)

### Stage 5 — Dense Semantic Embedding
- **Model:** `BAAI/bge-small-en-v1.5` (33M params, 384-dim)
- **Speed:** 600 candidates embedded in ~110 seconds on CPU
- Cosine similarity against pre-computed JD embedding

### Stage 6 — Reciprocal Rank Fusion
- Merges BM25 and dense rankings with **RRF (k=60)**
- No hyperparameter tuning required — robust by design
- Top-500 shortlist forwarded to reranker

### Stage 7 — LightGBM LambdaMART Reranking
- **Weak supervision:** Pseudo-labels generated from rule-based heuristics
- **32 features:** Retrieval scores + career features + behavioral signals
- **Post-processing penalties:** Consulting-only (×0.4), current-consulting (×0.7)
- Scores normalized to [0, 100]

### Stage 8 — Deterministic Template NLG
- **Zero hallucination:** Every sentence cites actual candidate data
- Rank-aware templates (top-20 vs mid vs fringe)
- Honest about gaps (e.g., "consulting-only career — explicit JD disqualifier")

---

### 🔑 Key Design Decisions
| Decision | Rationale |
|---|---|
| CPU-only, no GPU | Constraint compliance |
| BGE-small (33M params) | Fast CPU inference, strong retrieval quality |
| Honeypot = hard filter before retrieval | Prevents fraudulent profiles from polluting any rank |
| Consulting penalty post-LGBMrank | LightGBM trained on 46 samples — rule-based safety net |
| Tie-break by candidate_id ASC | Validator requirement |
"""

# ─── Gradio UI ────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(
        title="Redrob CandidateRank — AI Candidate Discovery Pipeline",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css="""
        .rank-badge { font-weight: bold; color: #7c3aed; }
        .score-bar { font-family: monospace; font-size: 12px; }
        footer { display: none !important; }
        .header-title { font-size: 2rem; font-weight: 800; }
        """
    ) as app:

        gr.Markdown("""
        # 🎯 Redrob CandidateRank
        ### Intelligent Candidate Discovery & Ranking Pipeline
        *Redrob India Runs Data & AI Challenge — Submission Demo*

        > **Task:** Rank 100,000 candidates for a **Senior AI Engineer** role using a CPU-bound pipeline completing in under 5 minutes with zero network access.
        """)

        with gr.Tabs():

            # ── Tab 1: Rankings ───────────────────────────────────────────────
            with gr.Tab("📊 Top-100 Rankings"):
                gr.Markdown(f"**{len(RESULTS)} candidates ranked** from 100,000 processed. Scores are normalized 0–100.")

                with gr.Row():
                    top_n_slider = gr.Slider(10, 100, value=25, step=5, label="Show top N")
                    min_score_slider = gr.Slider(0, 100, value=0, step=5, label="Min score filter")
                    search_box = gr.Textbox(placeholder="Search by ID or keyword...", label="Search")

                results_table = gr.Dataframe(
                    headers=["Rank", "Candidate ID", "Score", "Score Bar", "Reasoning (preview)"],
                    value=format_results_table(RESULTS, top_n=25),
                    interactive=False,
                    wrap=True,
                )

                filter_btn = gr.Button("🔍 Apply Filter", variant="primary")
                filter_btn.click(
                    fn=filter_and_display,
                    inputs=[top_n_slider, min_score_slider, search_box],
                    outputs=results_table,
                )

                gr.Markdown("---")
                gr.Markdown("### 🔎 Candidate Deep Dive")
                gr.Markdown("*Enter a candidate ID from the table above to see full reasoning and profile (sample candidates only)*")
                with gr.Row():
                    cand_id_input = gr.Textbox(placeholder="e.g. CAND_0000005", label="Candidate ID")
                    detail_btn = gr.Button("View Detail", variant="secondary")
                detail_output = gr.Markdown()
                detail_btn.click(fn=get_candidate_detail, inputs=cand_id_input, outputs=detail_output)
                cand_id_input.submit(fn=get_candidate_detail, inputs=cand_id_input, outputs=detail_output)

            # ── Tab 2: Methodology ────────────────────────────────────────────
            with gr.Tab("🏗️ Pipeline Methodology"):
                gr.Markdown(METHODOLOGY)

            # ── Tab 3: Stats ──────────────────────────────────────────────────
            with gr.Tab("📈 Pipeline Stats"):
                stats_md = f"""
## Runtime Statistics (Full 100K Run)

| Stage | Time | Notes |
|---|---|---|
| Stage 1: Ingestion | ~4s | orjson streaming |
| Stage 2: Honeypot Filter | ~11s | 11,511 hard-removed |
| Stage 3: Feature Engineering | ~22s | 88,489 candidates |
| Stage 4: BM25 Retrieval | ~13s | Top-600 shortlist |
| Stage 5: Dense Embedding | ~110s | BAAI/bge-small-en-v1.5 |
| Stage 6: RRF Fusion | <1s | BM25 + Dense merged |
| Stage 7: Reranking | <1s | LightGBM LambdaMART |
| Stage 8: Reasoning | <1s | Template NLG |
| **Total** | **~159s** | **Budget: 290s** |

## Honeypot Detection Results
- **Hard-removed (impossible):** 11,511 (11.5%)
  - Temporal impossibility (senior role during undergrad)
  - Impossible YOE (claims more YOE than years since graduation)
- **Soft-flagged (penalized, kept):** 5,501 (5.5%)
  - Skill hallucination (claims skills not found in career text)
  - Domain incoherence (non-tech career + advanced tech skills)

## Score Distribution (Top-100)
- **Rank 1 score:** 100.0
- **Rank 10 score:** ~75–85
- **Rank 50 score:** ~30–50  
- **Rank 100 score:** 0.0 (normalized floor)

## Key Differentiators
- **Pre-LLM ML depth** is weighted heavily (JD requirement)
- **Product company ratio** vs consulting: explicit penalty
- **Vector DB + embedding retrieval** in career text: strongest signal
- **Availability composite:** notice period + open_to_work + response rate
                """
                gr.Markdown(stats_md)

        gr.Markdown("""
        ---
        *Built for the Redrob India Runs Data & AI Challenge | Pipeline: BM25 → Dense Embeddings → RRF → LightGBM LambdaMART*
        """)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_port=7860)
