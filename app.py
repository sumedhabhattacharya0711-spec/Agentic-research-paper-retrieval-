"""
app.py — Gradio Space entry point for the research synthesis pipeline.
Replaces api.py + Dockerfile for Hugging Face Spaces deployment.

Deploy by pushing this file + requirements.txt + backend/core/ to a
Gradio SDK Space. No Dockerfile needed — HF manages the runtime.

Streaming approach:
  - gr.Progress tracks real pipeline stages live (replaces SSE stage events)
  - Generator function yields markdown chunks as each stage completes
    (replaces SSE log + done events)
  - Final output is the full formatted report with all hypothesis cards
"""

import os
import tempfile
from pathlib import Path

import gradio as gr

# ── pipeline imports ────────────────────────────────────────────────────────
# core/ sits alongside app.py in the Space repo root
from core.config import big, fast, safe_parse_json
from core.store import manifest, collection
from core.retrieval import rebuild_bm25
from core.orchestrator import extract_domain_anchor, orchestrate_final
from core.librarian import librarian_v2
from core.agents import mcp_fallback_node
from core.synthesis import ingest_papers_quick, synthesize
from core.ps_parser import parse_ps_pdf


# ── PS structuring for raw-text input ───────────────────────────────────────
# Mirrors structure_raw_ps() from api.py — same prompt, same output shape.
def structure_raw_ps(raw_text: str) -> dict:
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
    out = fast.invoke(prompt)
    parsed = safe_parse_json(out.content)
    if not parsed:
        return {"clean_text": raw_text, "weights": {}, "deliverables": [], "tasks": []}

    clean_parts = []
    if parsed.get("problem_statement"):
        clean_parts.append(parsed["problem_statement"])
    if parsed.get("technical_requirements"):
        clean_parts.append(parsed["technical_requirements"])
    for t in parsed.get("tasks", []):
        clean_parts.append(
            f"Task {t.get('number','?')}: {t.get('name','')}. "
            f"{t.get('description','')}")
    if parsed.get("dataset_description"):
        clean_parts.append(parsed["dataset_description"])

    weights = {}
    for ec in parsed.get("evaluation_criteria", []):
        name, w = ec.get("criterion", ""), ec.get("weight", 0)
        if name and w:
            weights[name] = w
    if not weights:
        for t in parsed.get("tasks", []):
            name, w = t.get("name", ""), t.get("weight", 0)
            if w:
                weights[name] = w

    return {
        "clean_text": "\n\n".join(clean_parts),
        "weights": weights,
        "deliverables": parsed.get("deliverables", []),
        "tasks": parsed.get("tasks", []),
    }


# ── report formatter ─────────────────────────────────────────────────────────
def format_sections_as_markdown(sections: list) -> str:
    """
    Render sections[] into rich markdown for the Gradio output panel.
    Each hypothesis gets a card-style block with status badge, answer,
    and citation list.
    """
    lines = []
    verified_count = sum(1 for s in sections if s["verified"])
    total = len(sections)

    lines.append(f"### {total} hypotheses · {verified_count}/{total} verified\n")
    lines.append("---\n")

    for i, s in enumerate(sections, 1):
        status = "✅ verified" if s["verified"] else "⚠️ unverified"
        section_type = s.get("section_type", "general")
        lines.append(f"## H{i} · `{section_type}` · {status}\n")
        lines.append(f"**{s['hypothesis']}**\n")
        lines.append(s["answer"] + "\n")

        if s.get("citations"):
            lines.append("**Sources:**")
            for c in s["citations"][:5]:
                title = c.get("title", "Unknown")[:70]
                lines.append(f"- {title}")
            lines.append("")

        lines.append("---\n")

    gaps = [s for s in sections if s.get("grade") in ("irrelevant", "insufficient")]
    if gaps:
        lines.append("### ⚠️ Coverage Gaps\n")
        for s in gaps:
            lines.append(f"- {s['hypothesis'][:80]}")

    return "\n".join(lines)


# ── main pipeline generator ──────────────────────────────────────────────────
def run_pipeline(ps_text: str, ps_file, progress=gr.Progress(track_tqdm=True)):
    """
    Generator function — each yield pushes a markdown string to the
    Gradio output Markdown component live, before the pipeline finishes.
    gr.Progress updates the progress bar + label at each real stage.

    ps_file: a file-like object from gr.File (has a .name path attribute)
             or None if the user pasted text instead.
    """

    if not ps_text.strip() and ps_file is None:
        yield "❌ Provide a problem statement or upload a PDF."
        return

    output = ""

    def emit(text: str):
        nonlocal output
        output += text + "\n"
        return output

    # ── Stage 0: parse PS ───────────────────────────────────────────────────
    progress(0.0, desc="parsing problem statement…")
    yield emit("### 🔍 Stage 1 — Parsing problem statement…")

    try:
        if ps_file is not None:
            # gr.File with type="filepath" gives a plain string path, not an object
            ps_data = parse_ps_pdf(Path(ps_file))
        else:
            ps_data = structure_raw_ps(ps_text.strip())
    except Exception as e:
        yield emit(f"\n❌ PS parsing failed: {e}")
        return

    PS = ps_data["clean_text"]
    ps_weights = ps_data.get("weights") or None
    ps_deliverables = ps_data.get("deliverables") or None

    yield emit(f"✅ PS parsed — {len(PS)} chars extracted\n")

    # ── Stage 1: orchestrator ───────────────────────────────────────────────
    progress(0.1, desc="generating hypotheses…")
    yield emit("### 🧠 Stage 2 — Generating hypotheses (YAKE + spaCy + arXiv bootstrap)…")

    try:
        domain_anchor = extract_domain_anchor(PS)
        hypotheses = orchestrate_final(
            PS, big,
            top_n=5,
            weights=ps_weights,
            deliverables=ps_deliverables,
        )
    except Exception as e:
        yield emit(f"\n❌ Orchestrator failed: {e}")
        return

    yield emit(f"✅ {len(hypotheses)} hypotheses generated\n")
    for i, h in enumerate(hypotheses, 1):
        yield emit(f"  **H{i}** `{h['section_type']}` — {h['hypothesis'][:80]}…")
    yield emit("")

    # ── Stage 2: librarian ──────────────────────────────────────────────────
    progress(0.25, desc="searching arXiv + Semantic Scholar…")
    yield emit("### 📚 Stage 3 — Searching literature (arXiv + Semantic Scholar)…")

    try:
        papers = librarian_v2(
            hypotheses, ps=PS, depth=1,
            seeds_per_query=3, chase_top_n=5,
            max_total=60, domain_anchor=domain_anchor,
        )
    except Exception as e:
        yield emit(f"\n❌ Librarian failed: {e}")
        return

    yield emit(f"✅ {len(papers)} papers retrieved\n")

    # ── Stage 3: MCP fallback ───────────────────────────────────────────────
    empty_hyps = [
        h for h in hypotheses
        if not any(
            p.get("source_hypothesis", "")[:80] == h["hypothesis"][:80]
            for p in papers
        )
    ]

    if empty_hyps:
        progress(0.45, desc=f"MCP fallback for {len(empty_hyps)} thin hypotheses…")
        yield emit(f"### 🔄 Stage 4 — MCP fallback search ({len(empty_hyps)} hypotheses had no papers)…")
        for h in empty_hyps:
            try:
                extra = mcp_fallback_node(h, PS)
                papers.extend(extra)
                if extra:
                    yield emit(f"  ✅ {len(extra)} papers found for `{h['hypothesis'][:60]}…`")
                else:
                    yield emit(f"  ⚠️ Nothing found for `{h['hypothesis'][:60]}…`")
            except Exception as e:
                yield emit(f"  ❌ MCP fallback error: {e}")
        yield emit("")
    else:
        progress(0.45, desc="all hypotheses have papers, skipping MCP fallback")

    # ── Stage 4: ingestion ──────────────────────────────────────────────────
    progress(0.55, desc="ingesting papers into vector store…")
    yield emit("### 💾 Stage 5 — Ingesting papers into vector store…")

    try:
        added = ingest_papers_quick(papers, max_papers=13)
        if added > 0:
            rebuild_bm25()
        yield emit(f"✅ {added} new papers ingested\n")
    except Exception as e:
        yield emit(f"\n⚠️ Ingestion partially failed: {e} — continuing with existing corpus\n")

    # ── Stage 5: synthesis ──────────────────────────────────────────────────
    progress(0.70, desc="synthesizing hypotheses (this takes a few minutes)…")
    yield emit("### ✍️ Stage 6 — Synthesizing + verifying each hypothesis…")
    yield emit("*(each hypothesis runs a full retrieve → grade → verify loop — please wait)*\n")

    try:
        report, sections = synthesize(PS, hypotheses, papers)
    except Exception as e:
        yield emit(f"\n❌ Synthesis failed: {e}")
        return

    progress(1.0, desc="done ✅")

    # ── Final output ────────────────────────────────────────────────────────
    verified = sum(1 for s in sections if s["verified"])
    sources = len(set(
        c.get("title", "") for s in sections for c in s.get("citations", [])
    ))

    yield emit(f"\n---\n## ✅ Report complete — {len(sections)} hypotheses · {sources} sources · {verified}/{len(sections)} verified\n")
    yield emit(format_sections_as_markdown(sections))
    yield emit("\n---\n*Full markdown report available below ↓*")
    yield emit("\n```markdown\n" + report + "\n```")


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Research Synthesis Engine",
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    ),
    css="""
    .gradio-container { max-width: 960px !important; margin: 0 auto; }
    #title { text-align: center; margin-bottom: 8px; }
    #subtitle { text-align: center; color: #888; margin-bottom: 24px; font-size: 14px; }
    #run-btn { background: #E8A33D !important; color: #1a1a1a !important; font-weight: 700; }
    #run-btn:hover { opacity: 0.9; }
    .output-panel { font-family: 'IBM Plex Mono', monospace; font-size: 13px; }
    """,
) as demo:

    gr.Markdown("# ◈ synth — research synthesis engine", elem_id="title")
    gr.Markdown(
        "Paste a problem statement or upload a PDF. "
        "Synth retrieves literature from arXiv + Semantic Scholar, "
        "generates grounded hypotheses, and verifies each against its sources.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            ps_text = gr.Textbox(
                label="Problem Statement (paste text)",
                placeholder="e.g. Design a lightweight, on-device recommendation system for low-bandwidth rural users...",
                lines=8,
            )
            ps_file = gr.File(
                label="Or upload a PDF",
                file_types=[".pdf"],
                type="filepath",
            )

            with gr.Row():
                clear_btn = gr.Button("clear", variant="secondary")
                run_btn   = gr.Button("synthesize →", variant="primary", elem_id="run-btn")

            gr.Markdown(
                "⏱️ *Typically 3–6 minutes. Progress updates live below.*",
                visible=True,
            )

        with gr.Column(scale=2):
            output = gr.Markdown(
                value="*Output will appear here as each stage completes…*",
                elem_classes=["output-panel"],
                height=680,
            )

    clear_btn.click(
        fn=lambda: ("", None, "*Output will appear here as each stage completes…*"),
        outputs=[ps_text, ps_file, output],
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[ps_text, ps_file],
        outputs=[output],
        show_progress=True,
    )

    gr.Markdown(
        "---\n*groq · chromadb · bge-base · bge-reranker · arxiv · semantic scholar*",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)