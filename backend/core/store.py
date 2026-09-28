"""
store.py — manifest, PDF parser (offline corpus-build only), chunker,
embedder, ChromaDB collection.
Corresponds to notebook Cell 2.

NOTE: parse_with_marker / ingest_pdf / ingest_folder are only used to build
the corpus OFFLINE (run them yourself whenever you add papers — e.g. back in
the Kaggle notebook, or a separate offline script). Nothing in the live
request path calls them; the marker-pdf import is deferred inside
parse_with_marker() itself, so it will simply never execute — and never
needs to be installed — in the production image.
"""

import os
from .config import re, json, hashlib, datetime, Path, chromadb, SentenceTransformer, AutoTokenizer

# ---------- Store ----------
STORE = Path(os.environ.get("DATA_DIR", "/app/store"))
STORE.mkdir(exist_ok=True)
(STORE / "pdfs").mkdir(exist_ok=True)
(STORE / "markdown").mkdir(exist_ok=True)
(STORE / "cache" / "s2").mkdir(parents=True, exist_ok=True)

# ---------- Manifest ----------
MANIFEST_PATH = STORE / "manifest.json"

def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}

def save_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

manifest = load_manifest()

def print_manifest_summary():
    m = load_manifest()
    ingested = sum(1 for v in m.values() if v.get("status") == "ingested")
    quarantined = sum(1 for v in m.values() if v.get("status") == "quarantined")
    total_chunks = sum(v.get("n_chunks", 0) for v in m.values())
    print(f"Manifest: {ingested} ingested, {quarantined} quarantined, "
          f"{total_chunks} total chunks")

# ---------- Parser (marker-pdf) ----------

def parse_with_marker(pdf_path):
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    if not hasattr(parse_with_marker, "_converter"):
        models = create_model_dict()
        parse_with_marker._converter = PdfConverter(artifact_dict=models)
    rendered = parse_with_marker._converter(str(pdf_path))
    return rendered.markdown

def parse_cached(pdf_path):
    md_path = STORE / "markdown" / f"{pdf_path.stem}.md"
    if md_path.exists():
        return md_path.read_text()
    md = parse_with_marker(pdf_path)
    md_path.write_text(md)
    return md

# ---------- Chunker ----------

tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
MAX_TOKENS, OVERLAP_TOKENS = 440, 60

def ntok(s):
    return len(tok.encode(s, add_special_tokens=False))

def clean_markdown(md):
    md = re.sub(r"!\[.*?\]\(.*?\)", "", md)
    md = re.sub(r"<span[^>]*>|</span>", "", md)
    md = re.sub(r"<sup>.*?</sup>", "", md)
    return md

def classify_section(path):
    p = path.lower()
    for key in ["abstract", "introduction", "related", "method", "approach",
                "experiment", "result", "discussion", "conclusion", "reference",
                "acknowledg", "appendix"]:
        if key in p:
            return key
    return "body"

def split_by_headers(md):
    blocks, path, buf = [], ["PREAMBLE"], []
    for line in md.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            if buf:
                blocks.append((" > ".join(path), "\n".join(buf).strip()))
                buf = []
            level, title = len(m.group(1)), m.group(2).strip()
            path = path[:level-1] + [title] if level > 1 else [title]
        else:
            buf.append(line)
    if buf:
        blocks.append((" > ".join(path), "\n".join(buf).strip()))
    return [b for b in blocks if b[1]]

def hard_split(p):
    if ntok(p) <= MAX_TOKENS:
        return [p]
    parts, cur, cur_t = [], [], 0
    for sent in re.split(r"(?<=[.!?])\s+", p):
        st = ntok(sent)
        if cur and cur_t + st > MAX_TOKENS:
            parts.append(" ".join(cur))
            cur, cur_t = [], 0
        cur.append(sent)
        cur_t += st
    if cur:
        parts.append(" ".join(cur))
    final = []
    for part in parts:
        ids = tok.encode(part, add_special_tokens=False)
        if len(ids) <= MAX_TOKENS:
            final.append(part)
        else:
            final += [tok.decode(ids[i:i + MAX_TOKENS])
                      for i in range(0, len(ids), MAX_TOKENS)]
    return final

def chunk_markdown(md, doc):
    md = clean_markdown(md)
    out = []
    for sec_path, text in split_by_headers(md):
        stype = classify_section(sec_path)
        if stype in ("reference", "acknowledg", "appendix") or sec_path == "PREAMBLE":
            continue

        paras = [piece for p in text.split("\n\n") if p.strip()
                 for piece in hard_split(p)]

        chunks_here, cur, cur_t = [], [], 0
        for p in paras:
            pt = ntok(p)
            if cur and cur_t + pt > MAX_TOKENS:
                chunks_here.append("\n\n".join(cur))
                tail, t = [], 0
                for q in reversed(cur):
                    if t + ntok(q) > OVERLAP_TOKENS:
                        break
                    tail.insert(0, q)
                    t += ntok(q)
                while tail and t + pt > MAX_TOKENS:
                    t -= ntok(tail.pop(0))
                cur, cur_t = tail + [p], t + pt
            else:
                cur.append(p)
                cur_t += pt
        if cur:
            chunks_here.append("\n\n".join(cur))

        for c in chunks_here:
            if ntok(c) < 30:
                continue
            out.append({
                "id": f'{doc["doc_id"]}::chunk_{len(out):04d}',
                "text": c,
                "embed_text": f'{doc["title"]} — {sec_path}\n\n{c}',
                "metadata": {"doc_id": doc["doc_id"],
                             "title": doc["title"][:80],
                             "section": sec_path[:80],
                             "section_type": stype,
                             "chunk_index": len(out)},
            })
    return out

# ---------- Embedder + ChromaDB ----------

emb_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
Q_PREFIX = "Represent this sentence for searching relevant passages: "

client = chromadb.PersistentClient(path=str(STORE / "chroma"))
collection = client.get_or_create_collection("research_chunks",
    metadata={"hnsw:space": "cosine"})

def chunk_and_index(markdown, record):
    doc = {"doc_id": record["doc_id"], "title": record["title"]}
    chunks = chunk_markdown(markdown, doc)
    if not chunks:
        return 0
    vecs = emb_model.encode([c["embed_text"] for c in chunks],
                            normalize_embeddings=True, show_progress_bar=False)
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=vecs.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks])
    return len(chunks)

# ---------- Ingestion functions ----------

def ingest_pdf(pdf_path, parse_fn, chunk_and_index_fn=None):
    pdf_path = Path(pdf_path)
    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
    if file_hash in manifest:
        return "skip"
    record = {"doc_id": file_hash, "title": pdf_path.stem.replace("_", " "),
              "source": pdf_path.name}
    try:
        md = parse_fn(pdf_path)
    except Exception as e:
        record["status"] = "parse_error"
        record["error"] = str(e)[:100]
        manifest[file_hash] = record
        save_manifest(manifest)
        return "error"
    if len(md) < 500:
        record["status"] = "quarantined"
        manifest[file_hash] = record
        save_manifest(manifest)
        return "quarantined"
    else:
        if chunk_and_index_fn is not None:
            record["n_chunks"] = chunk_and_index_fn(md, record)
        record["status"] = "ingested"
        record["ingested_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        print(f"  ingested: {record['title'][:60]}")
    manifest[file_hash] = record
    save_manifest(manifest)
    return "ingested"

def ingest_folder(folder, parse_fn, chunk_and_index_fn=None):
    results = [ingest_pdf(p, parse_fn, chunk_and_index_fn)
               for p in sorted(Path(folder).glob("*.pdf"))]
    print(f"\n{results.count('ingested')} ingested, "
          f"{results.count('skip')} skipped, "
          f"{results.count('quarantined')} quarantined, "
          f"{results.count('error')} errors")

def reset_index():
    global manifest
    collection.delete(where={"doc_id": {"$ne": ""}})
    manifest = {}
    save_manifest(manifest)
    print("index + manifest wiped")

# ---------- Abstract-only ingestion (runtime delta, no marker) ----------

def ingest_abstracts(papers):
    """Ingest paper abstracts as single chunks. No PDF, no marker."""
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
        vec = emb_model.encode(Q_PREFIX + embed_text, normalize_embeddings=True)
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

print(f"store ready | chroma: {collection.count()} chunks | "
      f"manifest: {len(manifest)} entries ✓")