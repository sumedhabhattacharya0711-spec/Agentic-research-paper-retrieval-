"""MCP server exposing research pipeline tools.
Run: python research_server.py
Agents connect via stdio and discover these tools at runtime."""

import os
import time
import json
from pathlib import Path

from fastmcp import FastMCP
import arxiv
import requests

mcp = FastMCP("research-tools")

# --- Semantic Scholar config ---
# Kept fully standalone (not importing from backend/core) since this server
# is meant to run as its own process — possibly spawned separately from the
# main API, possibly in a different environment entirely. Mirrors the same
# auth/caching/rate-limit behavior as backend/core/librarian.py's s2_get so
# this fallback path doesn't hit stricter unauthenticated rate limits or
# re-fetch things the main pipeline already cached.
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.environ.get("S2_API_KEY")
S2_FIELDS = "title,abstract,year,externalIds,url,referenceCount,citationCount"
S2_CACHE = Path(os.environ.get("DATA_DIR", "/app/store")) / "cache" / "s2"
S2_CACHE.mkdir(parents=True, exist_ok=True)

_last_call = 0

def _rate_limit(delay=0.2):
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_call = time.time()

def s2_get(endpoint, params=None):
    """Cached, authenticated, rate-limited S2 GET."""
    cache_key = endpoint.replace("/", "_") + "_" + json.dumps(params or {}, sort_keys=True)
    cache_key = "".join(c for c in cache_key if c.isalnum() or c in "_-")[:150]
    cache_path = S2_CACHE / f"{cache_key}.json"
    if cache_path.exists():
        raw = cache_path.read_text()
        if raw.strip() == "null":
            return None
        return json.loads(raw)
    _rate_limit()
    try:
        headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
        r = requests.get(f"{S2_BASE}{endpoint}", params=params,
                         headers=headers, timeout=15)
        if r.status_code == 429:
            time.sleep(3)
            r = requests.get(f"{S2_BASE}{endpoint}", params=params,
                             headers=headers, timeout=15)
        if r.status_code != 200:
            cache_path.write_text("null")
            return None
        data = r.json()
        cache_path.write_text(json.dumps(data))
        return data
    except Exception:
        cache_path.write_text("null")
        return None


@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> list:
    """Search arXiv for research papers matching a query.
    Returns titles, abstracts, years, and PDF URLs."""
    try:
        results = list(arxiv.Client().results(
            arxiv.Search(query=query, max_results=max_results,
                         sort_by=arxiv.SortCriterion.Relevance)))
        return [{
            "title": r.title,
            "abstract": r.summary[:500],
            "year": r.published.year,
            "pdf_url": r.pdf_url,
            "arxiv_id": r.entry_id.split("/")[-1],
        } for r in results]
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
def search_semantic_scholar(query: str, limit: int = 5) -> list:
    """Search Semantic Scholar for papers with citation data."""
    try:
        data = s2_get("/paper/search",
                      {"query": query, "limit": limit, "fields": S2_FIELDS})
        papers = data.get("data") or [] if data else []
        return [{
            "title": p.get("title"),
            "abstract": (p.get("abstract") or "")[:500],
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
        } for p in papers]
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
def get_paper_citations(s2_paper_id: str, limit: int = 10) -> list:
    """Get papers that cite a given paper (forward citation chase)."""
    try:
        data = s2_get(f"/paper/{s2_paper_id}/citations",
                      {"fields": S2_FIELDS, "limit": limit})
        citations = data.get("data") or [] if data else []
        return [{"title": c["citingPaper"].get("title"),
                 "year": c["citingPaper"].get("year")}
                for c in citations if c.get("citingPaper", {}).get("title")]
    except Exception as e:
        return [{"error": str(e)}]


if __name__ == "__main__":
    mcp.run()