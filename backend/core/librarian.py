"""
librarian.py — arXiv/Semantic Scholar search, tiered relevance critic,
cross-encoder paper reranking, citation-graph chasing.
Corresponds to notebook Cell 5.
"""

from .config import json, time, requests, arxiv, np, safe_parse_json, S2_API_KEY, S2_FIELDS, fast
from .store import STORE, Q_PREFIX, emb_model
from .retrieval import reranker

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_CACHE = STORE / "cache" / "s2"  # use STORE from Cell 2, not WORK_STORE
S2_CACHE.mkdir(parents=True, exist_ok=True)
_last_call = 0
def rate_limit(delay=0.2):
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_call = time.time()

def s2_get(endpoint, params=None):
    cache_key = endpoint.replace("/", "_") + "_" + json.dumps(params or {}, sort_keys=True)
    cache_key = "".join(c for c in cache_key if c.isalnum() or c in "_-")[:150]
    cache_path = S2_CACHE / f"{cache_key}.json"
    if cache_path.exists():
        raw = cache_path.read_text()
        if raw.strip() == "null":
            return None
        return json.loads(raw)
    rate_limit()
    try:
        headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
        r = requests.get(f"{S2_BASE}{endpoint}", params=params,
                        headers=headers, timeout=15)
        if r.status_code == 429:
            print("    S2 rate limited, waiting 3s...")
            time.sleep(3)
            r = requests.get(f"{S2_BASE}{endpoint}", params=params,
                            headers=headers, timeout=15)
        if r.status_code != 200:
            cache_path.write_text("null")
            return None
        data = r.json()
        cache_path.write_text(json.dumps(data))
        return data
    except Exception as e:
        print(f"    S2 error: {e}")
        cache_path.write_text("null")
        return None

def s2_search(query, limit=5):
    data = s2_get("/paper/search", {"query": query, "limit": limit, "fields": S2_FIELDS})
    return data.get("data") or [] if data else []

def s2_references(paper_id, limit=20):
    data = s2_get(f"/paper/{paper_id}/references", {"fields": S2_FIELDS, "limit": limit})
    return [r["citedPaper"] for r in (data.get("data") or [] if data else [])
            if r.get("citedPaper", {}).get("title")]

def s2_citations(paper_id, limit=20):
    data = s2_get(f"/paper/{paper_id}/citations", {"fields": S2_FIELDS, "limit": limit})
    return [r["citingPaper"] for r in (data.get("data") or [] if data else [])
            if r.get("citingPaper", {}).get("title")]

def arxiv_search(query, max_results=3):
    results = []
    try:
        for r in arxiv.Client().results(
            arxiv.Search(query=query, max_results=max_results,
                         sort_by=arxiv.SortCriterion.Relevance)):
            results.append({
                "title": r.title,
                "abstract": r.summary[:500],
                "year": r.published.year,
                "pdf_url": r.pdf_url,
                "arxiv_id": r.entry_id.split("/")[-1],
                "source": "arxiv",
            })
        time.sleep(1.0)
    except Exception as e:
        print(f"    arXiv error: {e}")
    return results

def normalize_s2(paper, source_tag="s2_search"):
    if not paper or not paper.get("title"):
        return None
    ext = paper.get("externalIds", {}) or {}
    return {
        "title": paper["title"],
        "abstract": (paper.get("abstract") or "")[:500],
        "year": paper.get("year"),
        "pdf_url": f"https://arxiv.org/pdf/{ext['ArXiv']}" if ext.get("ArXiv") else None,
        "arxiv_id": ext.get("ArXiv"),
        "s2_id": paper.get("paperId"),
        "source": source_tag,
    }

# ---------- Tiered Critic ----------

CRITIC_PROMPT = """Is this paper relevant to the team's project?

Project: {ps}

Paper:
TITLE: {title}
ABSTRACT: {abstract}

A paper is relevant if it covers ANY technique, method, model, dataset, or
system component that could be USED IN or INFORM the team's work.
A paper about a COMPONENT of the project is relevant even if it doesn't
mention the project's exact goal.
Only reject papers about completely unrelated domains.

Return ONLY JSON: {{"relevant": true/false, "reason": "one sentence"}}"""

def critic_filter_tiered(papers, ps, sim_high=0.65, sim_low=0.25, verbose=True):
    ps_vec = emb_model.encode(Q_PREFIX + ps[:500], normalize_embeddings=True)
    
    kept, dropped, llm_judged = [], [], 0
    
    for p in papers:
        title = p.get("title", "")
        abstract = p.get("abstract", "")[:400]
        paper_text = f"{title}. {abstract}"
        
        paper_vec = emb_model.encode(paper_text, normalize_embeddings=True)
        sim = float(np.dot(ps_vec, paper_vec))
        
        if sim > sim_high:
            p["critic"] = {"keep": True, "sim": round(sim, 3), "method": "embed_high"}
            kept.append(p)
            if verbose:
                print(f"  ✓ [SIM {sim:.2f}] {title[:60]}")
        elif sim < sim_low:
            p["critic"] = {"keep": False, "sim": round(sim, 3), "method": "embed_low"}
            dropped.append(p)
            if verbose:
                print(f"  ✗ [SIM {sim:.2f}] {title[:50]}")
        else:
            llm_judged += 1
            try:
                time.sleep(0.5)
                out = fast.invoke(CRITIC_PROMPT.format(
                    ps=ps[:300], title=title, abstract=abstract))
                result = safe_parse_json(out.content)
                if result:
                    is_relevant = result.get("relevant", True)
                    reason = result.get("reason", "")
                else:
                    is_relevant = True
                    reason = "parse fallback"
                
                p["critic"] = {"keep": is_relevant, "sim": round(sim, 3),
                              "method": "LLM", "reason": reason}
                if is_relevant:
                    kept.append(p)
                    if verbose:
                        print(f"  ✓ [LLM {sim:.2f}] {title[:60]}")
                else:
                    dropped.append(p)
                    if verbose:
                        print(f"  ✗ [LLM {sim:.2f}] {title[:50]} | {reason[:40]}")
            except Exception as e:
                p["critic"] = {"keep": True, "sim": round(sim, 3),
                              "method": "error_fallback"}
                kept.append(p)
    
    print(f"\n  Critic: {len(kept)} kept, {len(dropped)} dropped "
          f"({len(dropped)/(len(papers) or 1)*100:.0f}% filtered)")
    print(f"  LLM calls used: {llm_judged}/{len(papers)} papers")
    return kept

# ---------- FIX 2: Cross-encoder paper reranking ----------

def rerank_papers(papers, ps, top_n=20):
    """Precision filter: cross-encoder reranks papers against PS."""
    if len(papers) <= top_n:
        return papers
    
    pairs = [(ps[:300], f"{p.get('title','')}. {p.get('abstract','')[:300]}")
             for p in papers]
    scores = reranker.predict(pairs)
    
    for p, score in zip(papers, scores):
        p["rerank_score"] = float(score)
    
    ranked = sorted(papers, key=lambda p: -p["rerank_score"])
    
    print(f"\n  Paper reranking: {len(papers)} → {top_n}")
    for p in ranked[:5]:
        print(f"    {p['rerank_score']:+.2f}  {p['title'][:55]}")
    if len(ranked) > top_n:
        print(f"    ... dropped {len(ranked) - top_n} below threshold")
    
    return ranked[:top_n]

# ---------- Librarian v2 ----------

def librarian_v2(hypotheses, ps, depth=1, seeds_per_query=3,
                 chase_top_n=5, max_total=60, domain_anchor=""):

    seen_titles = set()
    all_papers = []

    def dedup_add(paper, tag):
        if not paper or not paper.get("title"):
            return False
        key = paper["title"].lower().strip()[:80]
        if key in seen_titles or len(all_papers) >= max_total:
            return False
        seen_titles.add(key)
        paper["source_tag"] = tag
        all_papers.append(paper)
        return True

    # FIX 1: domain-grounded queries
    print("=== Phase 1: Keyword Search ===\n")
    phase1 = []
    for h in hypotheses:
        print(f"[{h['section_type']}] {h['hypothesis'][:65]}")
        for query in h["search_queries"]:
            # append domain anchor to ground queries in the right domain
            grounded = f"{query} {domain_anchor}" if domain_anchor else query
            
            for p in arxiv_search(grounded, max_results=seeds_per_query):
                p["source_hypothesis"] = h["hypothesis"][:80]
                if dedup_add(p, f"arxiv|{query[:30]}"):
                    phase1.append(p)
                    print(f"  + [{p['year']}] {p['title'][:65]}")
            for sp in s2_search(grounded, limit=seeds_per_query):
                np_paper = normalize_s2(sp, f"s2|{query[:30]}")
                if np_paper:
                    np_paper["source_hypothesis"] = h["hypothesis"][:80]
                    if dedup_add(np_paper, f"s2|{query[:30]}"):
                        phase1.append(np_paper)
                        print(f"  + [{np_paper.get('year','')}] {np_paper['title'][:65]}")

    print(f"\n--- Phase 1: {len(phase1)} papers found ---")

    # FIX 2: critic then rerank
    print(f"\n=== Phase 2: Critic Filtering Phase 1 ===\n")
    survivors = critic_filter_tiered(phase1, ps)
    survivors = rerank_papers(survivors, ps, top_n=20)

    if depth == 0:
        print(f"\n{'='*60}")
        print(f"FINAL (depth=0): {len(survivors)} papers")
        print(f"{'='*60}")
        return survivors

    print(f"\n=== Phase 3: Citation Chase (depth={depth}) ===\n")

    chase_candidates = sorted(
        [p for p in survivors if p.get("critic", {}).get("sim", 0) > 0],
        key=lambda p: -p.get("critic", {}).get("sim", 0))

    chase_ready = []
    for p in chase_candidates[:chase_top_n]:
        if p.get("s2_id"):
            chase_ready.append(p)
        elif p.get("arxiv_id"):
            data = s2_get(f"/paper/ArXiv:{p['arxiv_id']}", {"fields": "paperId"})
            if data and data.get("paperId"):
                p["s2_id"] = data["paperId"]
                chase_ready.append(p)

    phase3 = []
    for p in chase_ready:
        if len(all_papers) >= max_total:
            break
        print(f"  Chasing: {p['title'][:55]}")
        for ref in s2_references(p["s2_id"], limit=10):
            nr = normalize_s2(ref, f"ref_of|{p['title'][:20]}")
            if nr:
                nr["source_hypothesis"] = p.get("source_hypothesis", "")
                if dedup_add(nr, nr.get("source", "")):
                    phase3.append(nr)
                    print(f"    ← [{nr.get('year','')}] {nr['title'][:50]}")
        for cit in s2_citations(p["s2_id"], limit=10):
            nc = normalize_s2(cit, f"citer_of|{p['title'][:20]}")
            if nc:
                nc["source_hypothesis"] = p.get("source_hypothesis", "")
                if dedup_add(nc, nc.get("source", "")):
                    phase3.append(nc)
                    print(f"    → [{nc.get('year','')}] {nc['title'][:50]}")

    print(f"\n--- Phase 3: {len(phase3)} papers from citation chase ---")

    if phase3:
        print(f"\n=== Phase 4: Critic Filtering Citation Chase ===\n")
        phase3_survivors = critic_filter_tiered(phase3, ps)
        phase3_survivors = rerank_papers(phase3_survivors, ps, top_n=10)
    else:
        phase3_survivors = []

    final = survivors + phase3_survivors
    print(f"\n{'='*60}")
    print(f"FINAL: {len(survivors)} from search + {len(phase3_survivors)} "
          f"from citations = {len(final)} papers")
    print(f"{'='*60}")
    return final

print("librarian + tiered critic + paper reranking ready ✓")
