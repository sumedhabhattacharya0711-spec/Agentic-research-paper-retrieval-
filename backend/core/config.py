"""
config.py — LLM clients, secrets, shared JSON-repair utility.
Corresponds to notebook Cell 1.
"""


import re
import os
from dotenv import load_dotenv
load_dotenv()
import time
import json
import hashlib
import datetime
import pickle
import urllib.request
import numpy as np
import networkx as nx
import requests
import arxiv
from pathlib import Path
from typing import TypedDict, List, Dict, Literal, Optional
from collections import defaultdict, Counter
from json_repair import repair_json
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- Secrets ---
# Kaggle notebooks pulled these from kaggle_secrets.UserSecretsClient, which
# only exists inside a Kaggle kernel. In production these are just env vars —
# set them via your host's secrets UI (Render/HF Spaces env vars, a local
# .env with python-dotenv, etc). GROQ_API_KEY is read automatically by
# ChatGroq() below; the other two are exposed as module-level constants so
# librarian.py can import them.
S2_API_KEY = os.environ.get("S2_API_KEY")  # optional — s2_get() degrades to unauthenticated rate limits if unset
S2_FIELDS = "title,abstract,year,externalIds,url,referenceCount,citationCount"
# --- LLM Setup ---
fast = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
big  = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- Utility ---
def safe_parse_json(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    m = re.search(r"\{.*", text, re.DOTALL)
    if not m:
        return None
    return repair_json(m.group(), return_objects=True)

print("imports + LLMs ready ✓ (local models)")
