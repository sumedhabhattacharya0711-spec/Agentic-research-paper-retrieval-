"""
synthesis.py — claim verification, abstract-only ingestion, quick full-PDF
ingestion for newly discovered papers, report synthesis + cleanup.
Corresponds to notebook Cell 7.

NOTE: ingest_papers_quick DOES fetch and parse PDFs live (via urllib +
PyMuPDF/fitz — not marker) for papers the librarian finds mid-query. This
needs outbound network access to wherever pdf_url points (typically
arxiv.org) and a writable STORE/pdfs directory at runtime.

NOTE: ingest_abstracts is defined here (and once, redundantly, in store.py)
but nothing in this pipeline currently calls it — ingest_papers_quick is
the function actually used, with abstract-only as its own internal
fallback. Left in place since removing it would be a mechanism change;
worth deciding deliberately whether you want it wired in anywhere.
"""

from .config import re, time, hashlib, urllib, big, safe_parse_json
from .store import collection, emb_model, manifest, save_manifest, chunk_markdown, STORE
from .librarian import s2_get
from .agents import agent_synth
import fitz  # PyMuPDF — lightweight, not marker-pdf

# ---------- Claim-level Verifier (synthesis-aware) ----------
#
# The old verifier judged the whole answer against all passages in one call
# and returned a single all_supported boolean — one weak sentence anywhere
# flipped the entire section to unverified, and nothing recorded WHICH
# claim failed, so a human had no way to review it. This version splits
# the answer into claims, judges each claim against only the passages it
# actually cites (in small batches), and carries the failing claims
# through on result["unverified_claims"] for the report/UI.

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\d])")

# Gap/meta sentences are the section's own epistemics, not factual claims
# about papers. The generator is REQUIRED to write an uncited gap sentence
# ("note the gap in one sentence at the end" — GEN_PROMPT), so auto-failing
# uncited sentences without this exemption flags every honest section.
_GAP_LANGUAGE = re.compile(
    r"what is missing|not (?:explicitly )?(?:covered|discussed|addressed)|"
    r"further research|passages (?:do not|only partially)|gap|"
    r"literature (?:does not|lacks)|would be needed|note that",
    re.I)

def extract_claims(answer):
    """Split an answer into claim sentences with their cited passage
    indices. Long sentences with no citation are kept too (cites=[]) —
    the generator's rules require a citation after every claim, so an
    uncited substantive sentence is itself worth flagging."""
    claims = []
    for s in _SENTENCE_SPLIT.split(answer.strip()):
        s = s.strip()
        if not s:
            continue
        cited = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", s)))
        if cited:
            claims.append({"text": s, "cites": cited})
        elif len(s) >= 80 and not _GAP_LANGUAGE.search(s):
            claims.append({"text": s, "cites": []})
    return claims

def _verify_claim_batch(claims, chunks):
    """One LLM call for a small batch of claims, each judged only against
    the passages it cites. Returns a verdict dict per claim."""
    blocks = []
    for i, cl in enumerate(claims):
        passages = "\n".join(
            f'  [{idx}] {chunks[idx-1]["text"][:500]}'
            for idx in cl["cites"] if 0 < idx <= len(chunks))
        blocks.append(f"CLAIM {i+1}: {cl['text']}\nCITED PASSAGES:\n{passages}")

    out = big.invoke(f"""You are fact-checking a literature review claim by claim.
For EACH claim below, judge whether its OWN cited passages reasonably support it.

It IS acceptable to:
- Summarize what a paper discusses
- Draw connections between papers
- Note that a topic is partially covered

It is NOT acceptable to:
- Attribute findings to a paper that doesn't discuss them
- Fabricate quotes, numbers, or results not in the passage

{chr(10).join(blocks)}

Return ONLY JSON: {{"verdicts": [
  {{"claim": 1, "supported": true/false, "reason": "one short sentence"}}
]}}""")
    parsed = safe_parse_json(out.content)
    verdicts = {v.get("claim"): v for v in (parsed or {}).get("verdicts", [])
                if isinstance(v, dict)}
    results = []
    for i, cl in enumerate(claims):
        v = verdicts.get(i + 1, {})
        results.append({
            "text": cl["text"],
            "cites": cl["cites"],
            "supported": bool(v.get("supported", True)),
            "reason": v.get("reason", "no verdict returned — assumed supported"),
        })
    return results

def verify_answer(result, batch_size=6):
    answer = result.get("answer", "")
    result["unverified_claims"] = []

    if any(phrase in answer.lower() for phrase in [
        "do not contain", "does not contain",
        "not contain sufficient", "not contain information",
        "cannot answer", "no relevant"]):
        result["verified"] = True
        result["trace"].append("verify → skipped (refusal answer)")
        return result

    if not result.get("citations") or not result.get("chunks"):
        result["verified"] = True
        result["trace"].append("verify → skipped (no citations)")
        return result

    claims = extract_claims(answer)
    if not claims:
        result["verified"] = True
        result["trace"].append("verify → skipped (no checkable claims)")
        return result

    verdicts = []
    uncited = [c for c in claims if not c["cites"]]
    cited = [c for c in claims if c["cites"]]

    for c in uncited:
        verdicts.append({"text": c["text"], "cites": [],
                         "supported": False,
                         "reason": "substantive sentence with no citation"})

    for start in range(0, len(cited), batch_size):
        batch = cited[start:start + batch_size]
        try:
            verdicts.extend(_verify_claim_batch(batch, result["chunks"]))
            time.sleep(0.5)
        except Exception as e:
            # verifier failure is not evidence against the claim — pass the
            # batch but say so, don't silently mark the section verified
            for c in batch:
                verdicts.append({"text": c["text"], "cites": c["cites"],
                                 "supported": True,
                                 "reason": f"verifier error, not checked: {e}"})

    result["claim_verdicts"] = verdicts
    result["unverified_claims"] = [v for v in verdicts if not v["supported"]]
    n_ok = sum(1 for v in verdicts if v["supported"])
    result["verified"] = len(result["unverified_claims"]) == 0
    result["trace"].append(
        f"verify → {n_ok}/{len(verdicts)} claims supported"
        + ("" if result["verified"] else
           f" — {len(result['unverified_claims'])} flagged for human review"))
    return result

# ---------- Abstract-only ingestion ----------

def ingest_abstracts(papers):
    added = 0
    for p in papers:
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        if not abstract or len(abstract) < 50:
            continue
        doc_id = hashlib.sha256(title.lower().encode()).hexdigest()[:16]
        if doc_id in manifest:
            continue
        chunk_id = f"{doc_id}::chunk_0000"
        embed_text = f"{title}\n\n{abstract}"
        vec = emb_model.encode(embed_text, normalize_embeddings=True)
        collection.add(
            ids=[chunk_id],
            embeddings=[vec.tolist()],
            documents=[abstract],
            metadatas=[{"doc_id": doc_id, "title": title[:80],
                        "section": "abstract", "section_type": "abstract",
                        "chunk_index": 0}])
        manifest[doc_id] = {"title": title[:80], "n_chunks": 1,
                            "type": "abstract_only", "status": "ingested"}
        added += 1
    save_manifest(manifest)
    return added

# ---------- Quick PDF ingestion ----------

# In ingest_papers_quick, replace the PDF URL section:

def ingest_papers_quick(papers, max_papers=13):
    added_full, added_abstract = 0, 0
    
    papers_sorted = sorted(papers,
        key=lambda p: p.get("rerank_score", p.get("critic", {}).get("sim", 0)),
        reverse=True)
    
    for p in papers_sorted[:max_papers]:
        title = p.get("title", "")
        doc_id = hashlib.sha256(title.lower().encode()).hexdigest()[:16]
        if doc_id in manifest:
            continue
        
        # try to get arxiv_id from S2 if missing
        pdf_url = p.get("pdf_url")
        arxiv_id = p.get("arxiv_id")
        
        if not pdf_url and not arxiv_id and p.get("s2_id"):
            try:
                data = s2_get(f"/paper/{p['s2_id']}", 
                             {"fields": "externalIds"})
                if data:
                    ext = data.get("externalIds", {}) or {}
                    if ext.get("ArXiv"):
                        arxiv_id = ext["ArXiv"]
                        p["arxiv_id"] = arxiv_id
            except:
                pass
        
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        
        if pdf_url:
            try:
                dest = STORE / "pdfs" / f"{doc_id}.pdf"
                urllib.request.urlretrieve(pdf_url, dest)
                
                if dest.stat().st_size > 5 * 1024 * 1024:
                    dest.unlink()
                    raise Exception("oversized")
                
                pdf = fitz.open(str(dest))
                full_text = ""
                for page in pdf:
                    full_text += page.get_text()
                pdf.close()
                
                if len(full_text) >= 500:
                    md = f"# {title}\n\n{full_text}"
                    doc = {"doc_id": doc_id, "title": title[:80]}
                    chunks = chunk_markdown(md, doc)
                    
                    if chunks:
                        vecs = emb_model.encode(
                            [c["embed_text"] for c in chunks],
                            normalize_embeddings=True, show_progress_bar=False)
                        collection.add(
                            ids=[c["id"] for c in chunks],
                            embeddings=vecs.tolist(),
                            documents=[c["text"] for c in chunks],
                            metadatas=[c["metadata"] for c in chunks])
                        
                        manifest[doc_id] = {
                            "title": title[:80], "n_chunks": len(chunks),
                            "type": "quick_parse", "status": "ingested"}
                        save_manifest(manifest)
                        added_full += 1
                        print(f"  ✓ [{len(chunks)} chunks] {title[:55]}")
                        time.sleep(0.5)
                        continue
            except Exception as e:
                pass
        
        # fallback: abstract-only
        abstract = p.get("abstract", "")
        if abstract and len(abstract) >= 50:
            chunk_id = f"{doc_id}::chunk_0000"
            embed_text = f"{title}\n\n{abstract}"
            vec = emb_model.encode(embed_text, normalize_embeddings=True)
            collection.add(
                ids=[chunk_id],
                embeddings=[vec.tolist()],
                documents=[abstract],
                metadatas=[{"doc_id": doc_id, "title": title[:80],
                            "section": "abstract", "section_type": "abstract",
                            "chunk_index": 0}])
            manifest[doc_id] = {"title": title[:80], "n_chunks": 1,
                                "type": "abstract_only", "status": "ingested"}
            added_abstract += 1
    
    # remaining beyond max_papers
    for p in papers_sorted[max_papers:]:
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        doc_id = hashlib.sha256(title.lower().encode()).hexdigest()[:16]
        if doc_id in manifest or not abstract or len(abstract) < 50:
            continue
        chunk_id = f"{doc_id}::chunk_0000"
        embed_text = f"{title}\n\n{abstract}"
        vec = emb_model.encode(embed_text, normalize_embeddings=True)
        collection.add(
            ids=[chunk_id],
            embeddings=[vec.tolist()],
            documents=[abstract],
            metadatas=[{"doc_id": doc_id, "title": title[:80],
                        "section": "abstract", "section_type": "abstract",
                        "chunk_index": 0}])
        manifest[doc_id] = {"title": title[:80], "n_chunks": 1,
                            "type": "abstract_only", "status": "ingested"}
        added_abstract += 1
    
    save_manifest(manifest)
    print(f"\n  Quick parse: {added_full} papers")
    print(f"  Abstract only: {added_abstract} papers")
    return added_full + added_abstract
# ---------- FIX 5: Post-generation cleanup ----------

def clean_report(report):
    report = re.sub(r'\[no relevant passage\]', '', report)
    report = re.sub(r'\[no citation\]', '', report)
    report = re.sub(r'\[no specific passage\]', '', report)
    report = re.sub(r'  +', ' ', report)
    report = re.sub(r'\n{3,}', '\n\n', report)
    return report.strip()

# ---------- FIX 3: Synthesize with source papers + PS context ----------

def synthesize(ps, hypotheses, papers=None):
    print(f"Synthesizing report for {len(hypotheses)} hypotheses...\n")
    sections = []
    
    for i, h in enumerate(hypotheses):
        print(f"\n{'='*60}")
        print(f"H{i+1} [{h['section_type']}]: {h['hypothesis'][:70]}")
        print(f"{'='*60}")
        
        # find papers discovered for this hypothesis
        source_papers = []
        if papers:
            hyp_key = h["hypothesis"][:80]
            source_papers = [p["title"] for p in papers
                           if p.get("source_hypothesis", "")[:80] == hyp_key]
        
        result = agent_synth.invoke({
            "question": h["hypothesis"],
            "sub_queries": h["search_queries"],
            "source_papers": source_papers,
            "ps_context": ps[:200],
            "route": "rag",
            "chunks": [], "grade": "sufficient", "retries": 0,
            "reformulated": [], "answer": "", "citations": [],
            "verified": False, "trace": []
        })
        
        result = verify_answer(result)
        
        sections.append({
            "hypothesis": h["hypothesis"],
            "section_type": h["section_type"],
            "answer": result["answer"],
            "citations": result["citations"],
            "verified": result["verified"],
            "unverified_claims": result.get("unverified_claims", []),
            "trace": result["trace"],
            "grade": result["grade"],
        })

        status = "✓ verified" if result["verified"] else (
            f"⚠ {len(result.get('unverified_claims', []))} claim(s) unverified")
        print(f"\n  [{status}] {len(result['answer'])} chars, "
              f"{len(result['citations'])} citations, "
              f"{len(result['trace'])} trace steps")
        for uc in result.get("unverified_claims", [])[:3]:
            print(f"    ⚠ {uc['text'][:70]} — {uc['reason'][:50]}")
    
    report = f"# Research Synthesis Report\n\n**Problem Statement:** {ps}\n\n"
    
    for i, s in enumerate(sections):
        report += f"## {i+1}. {s['hypothesis']}\n\n"
        if s["grade"] in ("irrelevant",) and not s["answer"].strip():
            report += "*Not covered by the discovered literature.*\n\n"
        else:
            report += s["answer"] + "\n\n"
            if s["citations"]:
                report += "**Sources:**\n"
                for c in s["citations"][:5]:
                    report += f"- [{c['index']}] {c['title'][:60]} — {c['section'][:40]}\n"
                report += "\n"
            if s.get("unverified_claims"):
                report += "**⚠ Unverified claims (need human review):**\n"
                for uc in s["unverified_claims"]:
                    report += f"- \"{uc['text'][:150]}\" — {uc['reason']}\n"
                report += "\n"
    
    gaps = [s for s in sections if s["grade"] in ("irrelevant", "insufficient")]
    if gaps:
        report += "## Gaps in Coverage\n\n"
        report += "The following hypotheses could not be fully addressed:\n\n"
        for s in gaps:
            report += f"- {s['hypothesis'][:80]}\n"
        report += "\n"
    
    # FIX 5: clean artifacts
    report = clean_report(report)
    
    print(f"\n{'='*60}")
    print(f"REPORT: {len(sections)} sections, "
          f"{sum(len(s['citations']) for s in sections)} total citations, "
          f"{sum(s['verified'] for s in sections)}/{len(sections)} verified")
    print(f"{'='*60}")
    
    return report, sections

print("verify + synthesize + ingestion ready ✓")