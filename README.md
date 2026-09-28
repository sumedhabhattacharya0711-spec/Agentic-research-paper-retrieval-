# Agentic RAG for Automated Research Synthesis

A research synthesis system that takes a competition problem statement as input and produces a multi-section review with per-claim verified citations. The pipeline generates research hypotheses with approriate techniques, discovers and ingests relevant papers at runtime, retrieves evidence through a hybrid search stack, and synthesizes findings into recommendations.

Built  under **IITG.AI**.
---

## 1. Overview

### Problem

Standard RAG fails on competition problem statements because:
- It doesn't know **what** to research — generic queries produce generic output
- It can only retrieve from a **fixed** corpus — novel topics produce empty sections
- It **hallucinates** citations — no mechanism to verify whether a cited passage actually supports the claim

### Solution

| Problem | Approach |
|---|---|
| What to research | An orchestrator extracts typed terms from the PS, classifies them against arXiv, and generates technique-level hypotheses via combinatorial patterns + one LLM polish call |
| Fixed corpus | A librarian discovers papers at runtime via arXiv + Semantic Scholar APIs; an agentic fallback searches live when the corpus proves insufficient mid-synthesis |
| Hallucinated citations | A claim-level verifier checks each sentence against its cited passage and surfaces a human-review queue of flagged claims |

### Cost

~6 large-model calls + ~25 small-model calls per full report (5 sections). Every other decision — grading, query reformulation, citation validation — is deterministic, each migrated from an LLM call after side-by-side comparison showed no quality loss.

---

## 2. Architecture

```
PS (text / PDF)
    │
    ▼
ORCHESTRATOR ─────────────── 5 hypotheses with search queries
    │                        (1 LLM call, rest deterministic)
    ▼
LIBRARIAN + CRITIC ───────── ~20 relevant papers discovered
    │                        (arXiv + S2 search, tiered filtering)
    ▼
DELTA INGESTION ──────────── top-13 full-text, rest abstract-only
    │                        (into the same vector store + BM25 index)
    ▼
AGENTIC SYNTHESIS LOOP ───── per hypothesis: retrieve → grade → generate
    │                        fallback: live search if corpus lacks coverage
    ▼
VERIFICATION + REPORT ────── per-claim entailment check
                             flagged claims queue + gaps appendix
```

The synthesis loop is a LangGraph state machine with conditional routing: sufficient chunks → generate; insufficient → reformulate query; retries exhausted → live-search fallback that ingests new papers and re-retrieves.

---

## 3. Corpus & Storage

### Pre-built Corpus

761 papers (9,653 chunks) built offline using marker-pdf for section-structured parsing. Stored in ChromaDB (persistent vector index) with a JSON manifest for dedup. The corpus provides baseline recall since established techniques are always retrievable regardless of the PS topic.

### Embeddings — BGE

`BAAI/bge-base-en-v1.5` (bi-encoder, 768-d vectors). Chosen for MTEB ranking at its size class, asymmetric query-document encoding , and CPU-inference viability.

The same embedder serves three roles beyond chunk retrieval:
- **Term classification** — comparing PS embedding against arXiv result titles to determine domain relevance (replaces all hardcoded domain-anchor heuristics)
- **Breadth filtering** — a term with many arXiv hits but few domain-relevant ones is too generic for this PS (catches "CNN" on an image-editing PS where its top results span medicine, driving, speech)
- **Hypothesis validation** — scoring and checking whether a hypothesis's search results are actually about this PS's domain

### Runtime Ingestion

Papers discovered by the librarian are ingested into the same ChromaDB collection before synthesis begins. Top-13 by relevance get full-text chunking (PyMuPDF); the rest get abstract-only single chunks. BM25 index is rebuilt once after the batch. The corpus is self-expanding — every run it expands for future runs on related topics.

---

## 4. Retrieval

### Hybrid BM25 + Dense

Neither retriever alone is enough: BM25 wins on exact technical terms ("QLoRA", "SDXL", "Mask R-CNN") where one character matters; dense wins on paraphrase ("reduce memory" ↔ "lower VRAM"). Research papers need both.

Fusion via **Reciprocal Rank Fusion (RRF)** — merges ranked lists without score normalization (BM25 and cosine scores are incomparable). A chunk ranked highly by both retrievers outscores one ranked highly by only one.

### Cross-Encoder Reranking

`BAAI/bge-reranker-v2-m3` — a cross-encoder that reads (query, passage) jointly through full attention, unlike the bi-encoder which embeds them independently. This catches cases where a chunk shares keywords with the query but answers a different question.

The retrieval funnel:
1. Hybrid retrieve ~20 candidates (cheap, high recall)
2. Cross-encoder rescore each pair (expensive, high precision)
3. +0.5 boost for papers discovered for this specific hypothesis
4. Max 2 chunks per paper (prevents survey-paper dominance)
5. Top-12 → generator context (~6-7 distinct papers)

### Reranker as Grader (Zero-LLM)

The cross-encoder's relevance scores double as the sufficiency grader for the agentic loop: high score → generate; low score →fallback. This eliminated 5–15 large-model grader calls per run as the cross-encoder is a better relevance judge than the LLM for this specific decision, verified via side-by-side comparison.

### Ablation Results

| Configuration | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| Dense only | 0.65 | 0.58 | 0.61 |
| BM25 only | 0.58 | 0.52 | 0.55 |
| Hybrid (no rerank) | 0.78 | 0.71 | 0.74 |
| Hybrid + rerank | 0.89 | 0.82 | 0.85 |
| **Full pipeline** | **0.91** | **0.85** | **0.88** |

The cross-encoder reranker provides the single largest jump (Recall@5: 0.78 → 0.89).
