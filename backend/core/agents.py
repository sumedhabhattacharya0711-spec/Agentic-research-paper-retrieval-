"""
agents.py — LangGraph state machine: router, planner, hypothesis-aware
retriever, grader, reformulator, PS-grounded generator. Compiles both the
synthesis graph (agent_synth) and the interactive Q&A graph (agent_interactive).
Corresponds to notebook Cell 6.
"""

from .config import re, json, TypedDict, List, Dict, Literal, StateGraph, END, fast, big
from .store import collection
from .retrieval import doc_lookup, hybrid_retrieve, reranker
from .mcp_fallback import run_mcp_fallback

class RAGState(TypedDict):
    question: str
    sub_queries: List[str]
    source_papers: List[str]
    ps_context: str
    route: Literal["rag", "direct"]
    chunks: List[Dict]
    grade: Literal["sufficient", "insufficient", "irrelevant"]
    retries: int
    reformulated: List[str]
    answer: str
    citations: List[Dict]
    verified: bool
    trace: List[str]

# ---------- ROUTER (interactive only) ----------

ROUTE_PROMPT = """Classify this input:
- "rag": needs information from research papers to answer
- "direct": greeting, meta question, or general chat
Input: {question}
Return ONLY JSON: {{"route": "rag"}} or {{"route": "direct"}}"""

def route_node(state: RAGState) -> RAGState:
    out = fast.invoke(ROUTE_PROMPT.format(question=state["question"]))
    m = re.search(r"\{.*\}", out.content, re.DOTALL)
    state["route"] = json.loads(m.group())["route"] if m else "rag"
    state["trace"].append(f"route → {state['route']}")
    return state

# ---------- PLANNER (interactive only) ----------

PLAN_PROMPT = """Decompose this question into 1-3 standalone search queries.
Each query should retrieve relevant passages from research papers.
If the question is already specific enough, return it unchanged.
Question: {question}
Return ONLY JSON: {{"queries": ["...", ...]}}"""

def plan_node(state: RAGState) -> RAGState:
    out = fast.invoke(PLAN_PROMPT.format(question=state["question"]))
    m = re.search(r"\{.*\}", out.content, re.DOTALL)
    queries = json.loads(m.group())["queries"][:3] if m else [state["question"]]
    state["sub_queries"] = queries
    state["trace"].append(f"plan → {queries}")
    return state

# ---------- FIX 3: HYPOTHESIS-AWARE RETRIEVER ----------

def retrieve_node(state: RAGState) -> RAGState:
    all_chunks = {}
    k_each = 15 + (5 * state["retries"])
    
    source_titles = set()
    if state.get("source_papers"):
        source_titles = {t[:50].lower() for t in state["source_papers"]}
    
    for q in state["sub_queries"]:
        cands = hybrid_retrieve(q, top_k=k_each)
        scores = reranker.predict([(q, doc_lookup[c]) for c in cands])
        for cid, score in zip(cands, scores):
            if cid not in all_chunks or score > all_chunks[cid]["rerank_score"]:
                meta = collection.get(ids=[cid], include=["metadatas"])["metadatas"][0]
                
                # boost chunks from papers found for THIS hypothesis
                boost = 0.0
                if source_titles and meta.get("title", "")[:50].lower() in source_titles:
                    boost = 0.5
                
                all_chunks[cid] = {
                    "id": cid, "text": doc_lookup[cid],
                    "rerank_score": float(score) + boost,
                    "title": meta.get("title", ""),
                    "section": meta.get("section", ""),
                }
    
    # enforce source diversity — max 3 chunks per paper
    ranked = sorted(all_chunks.values(), key=lambda x: -x["rerank_score"])
    diverse = []
    title_counts = {}
    for c in ranked:
        t = c["title"][:50]
        title_counts[t] = title_counts.get(t, 0) + 1
        if title_counts[t] <= 3:
            diverse.append(c)
        if len(diverse) >= 12:
            break
    
    state["chunks"] = diverse
    n_papers = len(set(c["title"][:50] for c in diverse))
    state["trace"].append(f"retrieve → {len(diverse)} chunks from "
                         f"{n_papers} papers (k_each={k_each})")
    return state

# ---------- GRADER (synthesis-aware) ----------

GRADE_PROMPT = """Topic: {question}

Retrieved passages:
{chunks}

Judge the SET of passages for a LITERATURE REVIEW section about this topic.
The passages do NOT need to directly answer the topic as a question.
They ARE useful if they discuss related methods, techniques, findings,
or concepts that inform the topic.

- "sufficient": passages contain relevant research to write about
- "insufficient": on-topic but too thin to write a meaningful section
- "irrelevant": mostly about unrelated domains
Return ONLY JSON: {{"grade": "...", "missing": "what is missing, if anything"}}"""

def grade_node(state: RAGState) -> RAGState:
    if not state["chunks"]:
        state["grade"] = "irrelevant"
        state["trace"].append("grade → irrelevant (no chunks)")
        return state
    
    scores = [c["rerank_score"] for c in state["chunks"]]
    top_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    if top_score > 2.0 and avg_score > 0.5:
        state["grade"] = "sufficient"
        state["trace"].append(f"grade → sufficient (top={top_score:.1f}, avg={avg_score:.1f})")
    
    elif top_score < -1.0:
        state["grade"] = "irrelevant"
        state["trace"].append(f"grade → irrelevant (top={top_score:.1f})")
    
    else:
        q_words = set(re.findall(r"[a-z]+", state["question"].lower()))
        q_words -= {"the","a","an","of","in","for","and","or","to","with",
                    "by","on","is","are","how","what","can","does"}
        
        chunk_text = " ".join(c["text"].lower() for c in state["chunks"])
        overlap = sum(1 for w in q_words if w in chunk_text) / max(len(q_words), 1)
        
        if top_score > 0.5 and overlap > 0.5:
            state["grade"] = "sufficient"
            state["trace"].append(f"grade → sufficient (top={top_score:.1f}, overlap={overlap:.0%})")
        elif top_score > -0.5 and overlap > 0.3:
            state["grade"] = "insufficient"
            state["trace"].append(f"grade → insufficient (top={top_score:.1f}, overlap={overlap:.0%})")
        else:
            state["grade"] = "irrelevant"
            state["trace"].append(f"grade → irrelevant (top={top_score:.1f}, overlap={overlap:.0%})")
    
    return state
# ---------- REFORMULATOR ----------

REFORM_PROMPT = """The search failed to find sufficient passages.
Original question: {question}
Previous queries tried: {history}
What was missing: {missing}
Write ONE new search query using different vocabulary.
Return ONLY JSON: {{"query": "..."}}"""

def reformulate_node(state: RAGState) -> RAGState:
    missing = state["trace"][-1].split("missing: ")[-1].rstrip(")")
    out = fast.invoke(REFORM_PROMPT.format(
        question=state["question"],
        history=state["reformulated"],
        missing=missing))
    m = re.search(r"\{.*\}", out.content, re.DOTALL)
    new_q = json.loads(m.group())["query"] if m else state["question"]
    state["sub_queries"] = [new_q]
    state["reformulated"].append(new_q)
    state["retries"] += 1
    state["trace"].append(f"reformulate → '{new_q}' (retry {state['retries']})")
    return state

# ---------- FIX 4: PS-GROUNDED GENERATOR ----------

GEN_PROMPT = """You are writing a section of a research literature review
for a technical competition team.

Their problem statement: {ps_context}

Using ONLY the passages below, write a section with two parts:

FINDINGS (brief — 2-3 sentences max):
What the literature establishes about this topic. Compress aggressively;
no restating the same point twice.

RECOMMENDATIONS (the bulk of the section — 6-8 concrete items):
Specific, actionable recommendations for the team. Each one:
- Names the exact technique/model/method to use, from the passages
- States WHY, citing the supporting passage like [2] or [1][3]
- Where possible, includes concrete parameters, tradeoffs, or numbers
  from the passages (e.g. "4-bit quantization reduces memory 4x [3]")
- Must be DISTINCT from the other recommendations — if two passages
  support the same action, merge them into one recommendation with
  both citations, do not write it twice

Rules:
- Every claim cites its passage [n].
- Recommendations must be things the team can DO, not observations.
- If passages only partially cover the topic, recommend based on what
  IS covered and note the gap in one sentence at the end.
- If passages are completely unrelated, say so explicitly.
- Do not use outside knowledge.

Passages:
{chunks}

Topic: {question}"""

def generate_node(state: RAGState) -> RAGState:
    if state["grade"] in ("irrelevant",) and state["retries"] >= 2:
        state["answer"] = ("The indexed papers do not contain sufficient information "
                          "to address this topic.")
        state["citations"] = []
        state["trace"].append("generate → calibrated refusal")
        return state
    chunks_text = "\n\n".join(
        f'[{i+1}] ({c["title"][:40]} — {c["section"][:30]})\n{c["text"][:800]}'
        for i, c in enumerate(state["chunks"]))
    
    ps_context = state.get("ps_context", "")[:200]
    
    out = big.invoke(GEN_PROMPT.format(
        question=state["question"],
        chunks=chunks_text,
        ps_context=ps_context))
    state["answer"] = out.content
    state["citations"] = [{"index": i+1, "title": c["title"], "section": c["section"]}
                          for i, c in enumerate(state["chunks"])]
    state["trace"].append(f"generate → {len(state['answer'])} chars")
    return state

# ---------- DIRECT ANSWER ----------

def direct_node(state: RAGState) -> RAGState:
    out = fast.invoke(state["question"])
    state["answer"] = out.content
    state["trace"].append("direct → answered without retrieval")
    return state

# ---------- SYNTHESIS GRAPH ----------

g_synth = StateGraph(RAGState)
g_synth.add_node("retrieve", retrieve_node)
g_synth.add_node("grade", grade_node)
g_synth.add_node("reformulate", reformulate_node)
g_synth.add_node("generate", generate_node)

g_synth.set_entry_point("retrieve")
g_synth.add_edge("retrieve", "grade")
g_synth.add_conditional_edges("grade", lambda s:
    "generate" if s["grade"] == "sufficient" or s["retries"] >= 2
    else "reformulate",
    {"generate": "generate", "reformulate": "reformulate"})
g_synth.add_edge("reformulate", "retrieve")
g_synth.add_edge("generate", END)

agent_synth = g_synth.compile()

# ---------- INTERACTIVE GRAPH ----------

g_interactive = StateGraph(RAGState)
g_interactive.add_node("route", route_node)
g_interactive.add_node("plan", plan_node)
g_interactive.add_node("retrieve", retrieve_node)
g_interactive.add_node("grade", grade_node)
g_interactive.add_node("reformulate", reformulate_node)
g_interactive.add_node("generate", generate_node)
g_interactive.add_node("direct", direct_node)

g_interactive.set_entry_point("route")
g_interactive.add_conditional_edges("route", lambda s: s["route"],
    {"rag": "plan", "direct": "direct"})
g_interactive.add_edge("plan", "retrieve")
g_interactive.add_edge("retrieve", "grade")
g_interactive.add_conditional_edges("grade", lambda s:
    "generate" if s["grade"] == "sufficient" or s["retries"] >= 3
    else "reformulate",
    {"generate": "generate", "reformulate": "reformulate"})
g_interactive.add_edge("reformulate", "retrieve")
g_interactive.add_edge("generate", END)
g_interactive.add_edge("direct", END)

agent_interactive = g_interactive.compile()

def ask(question):
    result = agent_interactive.invoke({
        "question": question, "sub_queries": [], "source_papers": [],
        "ps_context": "", "route": "rag",
        "chunks": [], "grade": "sufficient", "retries": 0,
        "reformulated": [], "answer": "", "citations": [],
        "verified": False, "trace": []
    })
    print(f"\nQ: {question}")
    print(f"\nTRACE:")
    for t in result["trace"]:
        print(f"  {t}")
    print(f"\nANSWER:\n{result['answer'][:500]}")
    print(f"\nCITATIONS:")
    for c in result["citations"][:5]:
        print(f"  [{c['index']}] {c['title'][:50]} — {c['section'][:30]}")
    print("\n" + "="*70)
    return result

# ---------- MCP FALLBACK (not part of either compiled graph) ----------
#
# g_synth/g_interactive query the already-ingested vector store — they have
# nothing to do with live arXiv/S2 search. This fallback belongs at the
# librarian_v2 call site instead, in whatever orchestrates your pipeline:
#
#   papers = librarian_v2(hypotheses, ps=PS, ...)
#   for h in hypotheses:
#       hyp_key = h["hypothesis"][:80]
#       found = [p for p in papers if p.get("source_hypothesis", "")[:80] == hyp_key]
#       if len(found) == 0:
#           papers.extend(mcp_fallback_node(h, PS))

def mcp_fallback_node(hypothesis: Dict, ps: str) -> List[Dict]:
    """
    Per-hypothesis fallback: if librarian_v2 found nothing for `hypothesis`,
    call this to spawn the MCP research-tools server and let a ReAct agent
    search arXiv/S2 more adaptively. Returns a list of paper dicts in the
    same shape librarian_v2 produces, tagged with
    source="mcp_fallback" — safe to extend directly onto librarian_v2's
    `papers` list.
    """
    papers = run_mcp_fallback(hypothesis["hypothesis"], ps)
    if papers:
        print(f"  MCP fallback: {len(papers)} papers found for "
              f"'{hypothesis['hypothesis'][:60]}'")
    return papers


print("agentic loop ready ✓")