"""
streamlit_app.py — local Streamlit frontend for the research synthesis pipeline.

Place in backend/ (sibling of core/ and mcp_server/).
Run from inside backend/:  streamlit run streamlit_app.py

Full local architecture: local SentenceTransformer + CrossEncoder,
spaCy keyphrase extraction, MCP subprocess fallback — the exact Kaggle
stack, no serverless APIs, no RAM constraints.
"""

import os
import time
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="synth — research synthesis engine",
    page_icon="◈",
    layout="wide",
)

# ── dark theme styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }
    .stApp { background-color: #08090D; }
    h1, h2, h3 { color: #EDEAE1 !important; }
    .stMarkdown { color: #C8C6BE; }
    .accent { color: #E8A33D; }
    div[data-testid="stFileUploader"] { border: 1.5px dashed #22262F; border-radius: 8px; padding: 8px; }
    .stButton>button {
        background: #E8A33D; color: #241804; font-weight: 700;
        border: none; border-radius: 7px; padding: 0.6em 2em;
        font-family: 'IBM Plex Mono', monospace;
    }
    .stButton>button:hover { opacity: 0.9; background: #E8A33D; color: #241804; }
    .stage-done { color: #4FD1C5; }
    .stage-active { color: #E8A33D; }
</style>
""", unsafe_allow_html=True)


def _safe_unlink(path, retries=5, delay=0.5):
    """Windows-safe temp deletion: AV scanners / lazy file mappings can hold
    a transient lock (WinError 32). Retry briefly; if still locked, leave it —
    it's in the OS temp dir and harmless."""
    for _ in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay)


# ── pipeline loading (cached — survives Streamlit reruns) ────────────────────
# st.cache_resource is CRITICAL here: Streamlit reruns this entire script on
# every widget interaction. Without caching, the embedder + reranker + corpus
# would reload on every button click.

@st.cache_resource(show_spinner="Loading pipeline (models + corpus — first run only)…")
def load_pipeline():
    from core.config import big, fast, safe_parse_json
    from core.store import manifest, collection
    from core.retrieval import rebuild_bm25
    from core.orchestrator import extract_domain_anchor, orchestrate_final
    from core.librarian import librarian_v2
    from core.agents import mcp_fallback_node
    from core.synthesis import ingest_papers_quick, synthesize
    from core.ps_parser import parse_ps_pdf
    return {
        "big": big, "fast": fast, "safe_parse_json": safe_parse_json,
        "manifest": manifest, "collection": collection,
        "rebuild_bm25": rebuild_bm25,
        "extract_domain_anchor": extract_domain_anchor,
        "orchestrate_final": orchestrate_final,
        "librarian_v2": librarian_v2,
        "mcp_fallback_node": mcp_fallback_node,
        "ingest_papers_quick": ingest_papers_quick,
        "synthesize": synthesize,
        "parse_ps_pdf": parse_ps_pdf,
    }


def structure_raw_ps(raw_text: str, P) -> dict:
    """LLM structuring for pasted-text PS (mirrors parse_ps_pdf's second half)."""
    prompt = f"""You are extracting a structured problem statement from a competition document.

RAW TEXT:
{raw_text[:6000]}

Extract ONLY the following sections. IGNORE company descriptions, team rules,
eligibility, registration, prizes, timelines, FAQs, and legal text.

Return ONLY JSON:
{{
  "problem_statement": "the core technical problem description (2-3 paragraphs)",
  "tasks": [
    {{"number": 1, "name": "task name", "description": "what to build/do", "weight": 25}}
  ],
  "deliverables": ["deliverable 1", "deliverable 2"],
  "evaluation_criteria": [
    {{"criterion": "name", "weight": 25, "description": "what is judged"}}
  ],
  "technical_requirements": "specific models, tools, constraints mentioned",
  "dataset_description": "any dataset details if mentioned"
}}"""
    out = P["fast"].invoke(prompt)
    parsed = P["safe_parse_json"](out.content)
    if not parsed:
        return {"clean_text": raw_text, "weights": {}, "deliverables": [], "tasks": []}

    clean_parts = []
    if parsed.get("problem_statement"):
        clean_parts.append(parsed["problem_statement"])
    if parsed.get("technical_requirements"):
        clean_parts.append(parsed["technical_requirements"])
    for t in parsed.get("tasks", []):
        clean_parts.append(f"Task {t.get('number','?')}: {t.get('name','')}. "
                           f"{t.get('description','')}")
    if parsed.get("dataset_description"):
        clean_parts.append(parsed["dataset_description"])

    weights = {}
    for ec in parsed.get("evaluation_criteria", []):
        if ec.get("criterion") and ec.get("weight"):
            weights[ec["criterion"]] = ec["weight"]
    if not weights:
        for t in parsed.get("tasks", []):
            if t.get("name") and t.get("weight"):
                weights[t["name"]] = t["weight"]

    return {
        "clean_text": "\n\n".join(clean_parts),
        "weights": weights,
        "deliverables": parsed.get("deliverables", []),
        "tasks": parsed.get("tasks", []),
    }


# ── header ───────────────────────────────────────────────────────────────────
st.markdown("# ◈ synth — <span class='accent'>research synthesis engine</span>",
            unsafe_allow_html=True)
st.markdown(
    "Paste a problem statement or upload a PDF. Synth retrieves literature from "
    "arXiv + Semantic Scholar, generates grounded hypotheses, and verifies each "
    "against its sources."
)
st.divider()

# ── layout: input left, output right ────────────────────────────────────────
col_in, col_out = st.columns([2, 3], gap="large")

with col_in:
    st.markdown("### intake")
    ps_text = st.text_area(
        "Problem statement (paste text)",
        height=220,
        placeholder="e.g. Design a lightweight, on-device recommendation system "
                    "for low-bandwidth rural users…",
    )
    ps_file = st.file_uploader("Or upload a PDF", type=["pdf"])

    run = st.button("synthesize →", use_container_width=True)
    st.caption("Runs the full local pipeline — typically 2-4 minutes.")

with col_out:
    output_area = st.container()

# ── pipeline execution ───────────────────────────────────────────────────────
if run:
    if not ps_text.strip() and ps_file is None:
        st.error("Provide a problem statement or upload a PDF.")
        st.stop()

    P = load_pipeline()

    with output_area:
        # ---- Stage 0: parse PS ----
        with st.status("Stage 1 — Parsing problem statement…", expanded=False) as s1:
            try:
                if ps_file is not None:
                    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
                    tmp = Path(tmp_path)
                    try:
                        with os.fdopen(fd, "wb") as f:
                            f.write(ps_file.getvalue())
                        ps_data = P["parse_ps_pdf"](tmp)
                    finally:
                        _safe_unlink(tmp)
                else:
                    ps_data = structure_raw_ps(ps_text.strip(), P)
            except Exception as e:
                s1.update(label=f"❌ PS parsing failed: {e}", state="error")
                st.error(f"PS parsing failed: {e}")
                st.stop()

            PS = ps_data["clean_text"]
            ps_weights = ps_data.get("weights") or None
            ps_deliverables = ps_data.get("deliverables") or None
            s1.update(label=f"✅ Stage 1 — PS parsed ({len(PS)} chars)", state="complete")

        # ---- Stage 1: orchestrator ----
        with st.status("Stage 2 — Generating hypotheses…", expanded=True) as s2:
            try:
                domain_anchor = P["extract_domain_anchor"](PS)
                hypotheses = P["orchestrate_final"](
                    PS, P["big"], top_n=5,
                    weights=ps_weights, deliverables=ps_deliverables)
            except Exception as e:
                s2.update(label=f"❌ Hypothesis generation failed: {e}", state="error")
                st.error(f"Hypothesis generation failed: {e}")
                st.stop()
            for i, h in enumerate(hypotheses, 1):
                st.markdown(f"**H{i}** `{h['section_type']}` — {h['hypothesis'][:90]}…")
            s2.update(label=f"✅ Stage 2 — {len(hypotheses)} hypotheses generated",
                      state="complete", expanded=False)

        # ---- Stage 2: librarian ----
        with st.status("Stage 3 — Searching literature (arXiv + Semantic Scholar)…",
                       expanded=False) as s3:
            try:
                papers = P["librarian_v2"](
                    hypotheses, ps=PS, depth=1, seeds_per_query=3,
                    chase_top_n=5, max_total=60, domain_anchor=domain_anchor)
            except Exception as e:
                st.warning(f"Librarian error ({e}) — continuing with 0 new papers; "
                           "synthesis will use the existing corpus.")
                papers = []
            s3.update(label=f"✅ Stage 3 — {len(papers)} papers retrieved",
                      state="complete")

        # ---- Stage 3: MCP fallback ----
        empty_hyps = [
            h for h in hypotheses
            if not any(p.get("source_hypothesis", "")[:80] == h["hypothesis"][:80]
                       for p in papers)
        ]
        if empty_hyps:
            with st.status(f"Stage 4 — MCP fallback ({len(empty_hyps)} thin hypotheses)…",
                           expanded=True) as s4:
                for h in empty_hyps:
                    try:
                        extra = P["mcp_fallback_node"](h, PS)
                    except Exception as e:
                        st.markdown(f"⚠️ Fallback error for `{h['hypothesis'][:50]}…`: {e}")
                        extra = []
                    papers.extend(extra)
                    if extra:
                        st.markdown(f"✅ {len(extra)} papers for "
                                    f"`{h['hypothesis'][:60]}…`")
                    else:
                        st.markdown(f"⚠️ Nothing found for `{h['hypothesis'][:60]}…`")
                s4.update(label="✅ Stage 4 — MCP fallback complete",
                          state="complete", expanded=False)

        # ---- Stage 4: ingestion ----
        with st.status("Stage 5 — Ingesting papers into vector store…",
                       expanded=False) as s5:
            try:
                added = P["ingest_papers_quick"](papers, max_papers=13)
                if added > 0:
                    P["rebuild_bm25"]()
            except Exception as e:
                st.warning(f"Ingestion partially failed ({e}) — "
                           "continuing with existing corpus.")
                added = 0
            s5.update(label=f"✅ Stage 5 — {added} new papers ingested",
                      state="complete")

        # ---- Stage 5: synthesis ----
        with st.status("Stage 6 — Synthesizing + verifying each hypothesis…",
                       expanded=False) as s6:
            try:
                report, sections = P["synthesize"](PS, hypotheses, papers)
                s6.update(label="✅ Stage 6 — Synthesis complete", state="complete")
            except Exception as e:
                s6.update(label=f"❌ Synthesis failed: {e}", state="error")
                st.error(f"Synthesis failed: {e}")
                st.stop()

        # ---- results ----
        st.divider()
        verified = sum(1 for s in sections if s["verified"])
        sources = len({c.get("title", "") for s in sections
                       for c in s.get("citations", [])})

        flagged = sum(len(s.get("unverified_claims", [])) for s in sections)
        st.markdown(f"## {len(sections)} hypotheses · "
                    f"{sources} sources · {verified}/{len(sections)} verified"
                    + (f" · {flagged} claims flagged" if flagged else ""))

        for i, s in enumerate(sections, 1):
            badge = "✅ verified" if s["verified"] else "⚠️ unverified"
            with st.expander(
                f"H{i} · [{s.get('section_type','general')}] · {badge} — "
                f"{s['hypothesis'][:80]}…",
                expanded=(i == 1),
            ):
                st.markdown(f"**{s['hypothesis']}**")
                st.markdown(s["answer"])
                if s.get("citations"):
                    st.markdown("**Sources:**")
                    for c in s["citations"][:6]:
                        st.markdown(f"- {c.get('title', 'Unknown')[:80]}")
                if s.get("unverified_claims"):
                    st.markdown("**⚠️ Unverified claims — review before trusting:**")
                    for uc in s["unverified_claims"]:
                        st.markdown(
                            f"- \"{uc['text'][:140]}\" — *{uc['reason'][:80]}*")

        st.download_button(
            "download report (.md) ↓",
            data=report,
            file_name="synthesis-report.md",
            mime="text/markdown",
        )
