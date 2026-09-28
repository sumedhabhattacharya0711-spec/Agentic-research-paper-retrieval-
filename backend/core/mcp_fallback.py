"""
mcp_fallback.py — spawns backend/mcp_server/research_server.py over stdio,
loads its tools into a LangGraph ReAct agent, and runs it as a fallback
paper search when librarian_v2's primary search comes up thin/empty for a
given hypothesis.

LOCAL MODE: subprocess spawning is fine on a real machine — the 512MB
constraint that forced the in-process rewrite on Render doesn't apply.
"""

import sys
import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from .config import big, safe_parse_json

_SERVER_PATH = Path(__file__).resolve().parent.parent / "mcp_server" / "research_server.py"

_MCP_CONFIG = {
    "research-tools": {
        "command": sys.executable,
        "args": [str(_SERVER_PATH)],
        "transport": "stdio",
    }
}

FALLBACK_PROMPT = """You are searching for research papers relevant to this hypothesis,
which the primary literature search failed to find enough for.

Problem statement context: {ps_context}
Hypothesis: {hypothesis}

Use the available tools to search arXiv and Semantic Scholar. If a direct
search on the hypothesis wording comes back empty, try broader or adjacent
terms — don't just repeat the same query. Stop once you find 3-5 genuinely
relevant papers, or after a reasonable number of attempts if nothing turns up.

Return ONLY JSON in this exact shape (no other text):
{{"papers": [
  {{"title": "...", "abstract": "...", "year": 2023, "pdf_url": "..."}}
]}}
If you find nothing relevant, return {{"papers": []}}."""


async def _run_mcp_fallback_async(hypothesis_text: str, ps_context: str) -> list:
    client = MultiServerMCPClient(_MCP_CONFIG)
    tools = await client.get_tools()

    agent = create_react_agent(big, tools)

    prompt = FALLBACK_PROMPT.format(
        ps_context=ps_context[:300],
        hypothesis=hypothesis_text)

    result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
    final_message = result["messages"][-1].content

    parsed = safe_parse_json(final_message)
    papers = parsed.get("papers", []) if parsed else []

    for p in papers:
        p["source"] = "mcp_fallback"
        p["source_hypothesis"] = hypothesis_text[:80]

    return papers


def run_mcp_fallback(hypothesis_text: str, ps_context: str = "") -> list:
    """
    Synchronous entry point. Returns [] on any failure rather than raising,
    since this is itself a fallback path.
    """
    try:
        return asyncio.run(_run_mcp_fallback_async(hypothesis_text, ps_context))
    except Exception as e:
        print(f"  MCP fallback error: {e}")
        return []
