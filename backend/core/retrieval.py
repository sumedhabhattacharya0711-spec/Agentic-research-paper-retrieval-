"""
retrieval.py — BM25 index, hybrid dense+sparse retrieval (RRF fusion),
cross-encoder reranker.
Corresponds to notebook Cell 3.
"""

from .config import re, BM25Okapi, CrossEncoder
from .store import collection, emb_model, Q_PREFIX

# ---------- BM25 ----------

def bm25_tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())

chunk_ids, docs, doc_lookup, bm25 = [], [], {}, None

def rebuild_bm25():
    global bm25
    data = collection.get(include=["documents"])
    # Mutate in place — agents.py holds references to these exact objects
    # from its `from .retrieval import doc_lookup` at import time. Rebinding
    # (chunk_ids = ...) would create new objects that agents.py never sees,
    # causing KeyErrors on newly ingested chunks during synthesis.
    chunk_ids[:] = data["ids"]
    docs[:] = data["documents"]
    doc_lookup.clear()
    doc_lookup.update(zip(chunk_ids, docs))
    bm25 = BM25Okapi([bm25_tok(d) for d in docs]) if docs else None
    print(f"BM25 built: {len(docs)} chunks")

def _ensure_bm25():
    """Lazy-init: load corpus into RAM only on first retrieval call."""
    global bm25
    if bm25 is None:
        rebuild_bm25()

# ---------- Hybrid Retrieval (Dense + BM25 + RRF) ----------

def hybrid_retrieve(q, top_k=5, k_each=20):
    _ensure_bm25()
    qv = emb_model.encode(Q_PREFIX + q, normalize_embeddings=True)
    dense = collection.query(
        query_embeddings=[qv.tolist()], n_results=k_each)["ids"][0]
    scores = bm25.get_scores(bm25_tok(q))
    sparse = [chunk_ids[i] for i in
              sorted(range(len(scores)), key=lambda i: -scores[i])[:k_each]]
    K, fused = 60, {}
    for lst in (dense, sparse):
        for r, cid in enumerate(lst):
            fused[cid] = fused.get(cid, 0) + 1/(K + r + 1)
    return sorted(fused, key=fused.get, reverse=True)[:top_k]

# ---------- Reranker ----------

reranker = CrossEncoder("BAAI/bge-reranker-base")
print("reranker loaded ✓ (local)")

print("retrieval stack ready ✓")