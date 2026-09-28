"""
ps_parser.py — Parses a competition/hackathon Problem Statement (PS) PDF into
structured data: core problem text, tasks, evaluation weights, deliverables.

Runs entirely in the live request path (no offline/corpus dependency).
Text extraction is PyMuPDF (fitz) first; if a page has no usable text layer
(i.e. it's a scanned/photographed page), it falls back to Tesseract OCR.
A single cheap LLM call then structures the extracted text.

Deps: PyMuPDF, pytesseract, Pillow, tesseract-ocr (system package).
"""

import re
import io
import fitz               # PyMuPDF
import pytesseract
from PIL import Image

# Adjust these to match your actual package layout.
from .config import fast, safe_parse_json    # ChatGroq client + JSON repair helper
from .orchestrator import clean_term          # shared term-cleaning util


# ============================================================
# TEXT EXTRACTION — PyMuPDF, with Tesseract OCR fallback
# ============================================================

def page_text_with_ocr_fallback(page, min_chars=30, dpi=200):
    """
    Extract text from a single PDF page. If the page's text layer is empty
    or near-empty (typical of a scanned/photographed page), rasterize it
    and run Tesseract OCR instead.

    min_chars: below this, the page is treated as "no usable text layer".
    dpi: rasterization resolution for OCR — 200 is a reasonable
         accuracy/speed default; push to 300 only if you're seeing bad
         OCR output on small fonts.
    """
    text = page.get_text()
    if len(text.strip()) >= min_chars:
        return text, False

    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img), True


def extract_ps_from_pdf(pdf_path):
    """Extract text from a PS PDF with page tracking (OCR-aware)."""
    pdf = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(pdf):
        text, was_ocr = page_text_with_ocr_fallback(page)
        pages.append({
            "page": i + 1,
            "text": text,
            "ocr": was_ocr,
        })
    pdf.close()
    return pages


# ============================================================
# SECTION CLASSIFICATION — noise / high-priority / medium
# ============================================================

def classify_ps_section(text):
    """Classify a text block as important or noise."""
    text_lower = text.lower()[:200]

    # NOISE sections — skip entirely
    noise_patterns = [
        r"^about\s+(us|the company|.{3,30})\s*$",
        r"rules?\s*(and|&)\s*regulations?",
        r"eligibility\s*(criteria)?",
        r"registration",
        r"prize|reward|winner|award",
        r"timeline|schedule|important dates",
        r"contact\s*(us|info|details)",
        r"disclaimer|legal|terms\s*(and|&)\s*conditions",
        r"sponsor|partner|organiz",
        r"team\s*size|team\s*composition",
        r"code\s*of\s*conduct",
        r"frequently\s*asked|faq",
        r"copyright|all\s*rights\s*reserved",
    ]
    for p in noise_patterns:
        if re.search(p, text_lower):
            return "noise"

    # HIGH PRIORITY sections
    high_patterns = [
        r"problem\s*statement",
        r"task\s*\d|track\s*\d",
        r"objective|mission|challenge",
        r"evaluation\s*(criteria|metric|rubric)",
        r"scoring|grading|judging|weightage|marks",
        r"deliverable|submission\s*(checklist|format|requirement)",
        r"expected\s*(outcome|output)",
        r"technical\s*(requirement|specification|constraint)",
        r"dataset|data\s*description",
        r"architecture|system\s*design",
    ]
    for p in high_patterns:
        if re.search(p, text_lower):
            return "high"

    return "medium"


def extract_evaluation_weights(text):
    """Extract evaluation criteria with percentage weights."""
    weights = {}

    # Pattern: "Product Design (25%)" or "Product Design — 25%"
    for match in re.findall(
        r"([A-Za-z][A-Za-z\s&/,]+?)\s*[\(–\-—:]\s*(\d+)\s*%\s*\)?",
        text):
        name = clean_term(match[0].strip())
        if len(name) >= 3 and len(name) <= 60:
            weights[name] = int(match[1])

    # Pattern: "25% — Product Design" or "25% for Product Design"
    for match in re.findall(
        r"(\d+)\s*%\s*[\(–\-—:for]*\s*([A-Za-z][A-Za-z\s&/,]+?)(?:\n|$|\.)",
        text):
        name = clean_term(match[1].strip())
        if len(name) >= 3 and len(name) <= 60:
            weights[name] = int(match[0])

    # Pattern: "Product Design: 25 marks" or "Product Design: 25/100"
    for match in re.findall(
        r"([A-Za-z][A-Za-z\s&/,]+?)\s*:\s*(\d+)\s*(?:marks|points|/100)",
        text, re.I):
        name = clean_term(match[0].strip())
        if len(name) >= 3:
            weights[name] = int(match[1])

    return weights


def extract_deliverables(text):
    """Extract required deliverables and submission items."""
    deliverables = []

    for match in re.findall(
        r"(?:deliverable|submission\s*(?:checklist|requirement|format)|"
        r"must\s*(?:include|submit|provide)|expected\s*(?:outcome|output)|"
        r"you\s*(?:should|must|need\s*to)\s*submit)s?\s*[:\-–]\s*(.+?)(?:\n\n|\Z)",
        text, re.I | re.DOTALL):

        for item in re.split(r"\n\s*[-•▪◦]\s*|\n\s*\d+[.)]\s*|\n\s*[a-z][.)]\s*",
                              match):
            clean = item.strip()
            if len(clean) >= 15 and len(clean) <= 200:
                if not re.match(r"^[A-Z][a-z]+\s*$", clean):
                    deliverables.append(clean)

    return deliverables


def extract_tasks(text):
    """Extract task/track descriptions with numbers."""
    tasks = []

    for match in re.findall(
        r"(?:task|track|phase|stage)\s*(\d+)\s*[:\-–—]\s*(.+?)(?=(?:task|track|phase|stage)\s*\d|\Z)",
        text, re.I | re.DOTALL):
        desc = match[1].strip()
        desc = re.split(r"\n\n\n", desc)[0][:500]
        tasks.append({
            "number": int(match[0]),
            "description": desc.strip()
        })

    return tasks


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def parse_ps_pdf(pdf_path):
    """Extract structured PS from PDF using PyMuPDF (+ Tesseract fallback) + LLM filtering."""

    # Step 1: extract raw text, page by page, OCR fallback per page
    pdf = fitz.open(str(pdf_path))
    raw_text = ""
    n_pages = len(pdf)
    ocr_pages = 0
    for page in pdf:
        text, was_ocr = page_text_with_ocr_fallback(page)
        if was_ocr:
            ocr_pages += 1
        raw_text += text + "\n\n"
    pdf.close()

    print(f"  PDF: {n_pages} pages, {len(raw_text)} chars"
          + (f" ({ocr_pages} page(s) OCR'd)" if ocr_pages else ""))

    # Step 2: LLM extracts structured PS (1 cheap 8B call)
    prompt = f"""You are extracting a structured problem statement from a competition PDF.

RAW PDF TEXT:
{raw_text[:6000]}

Extract ONLY the following sections. IGNORE company descriptions, team rules,
eligibility, registration, prizes, timelines, FAQs, and legal text.

Return ONLY JSON:
{{
  "problem_statement": "the core technical problem description (2-3 paragraphs)",
  "tasks": [
    {{"number": 1, "name": "task name", "description": "what to build/do", "weight": 25}},
    {{"number": 2, "name": "task name", "description": "what to build/do", "weight": 15}}
  ],
  "deliverables": ["deliverable 1", "deliverable 2"],
  "evaluation_criteria": [
    {{"criterion": "name", "weight": 25, "description": "what is judged"}}
  ],
  "technical_requirements": "specific models, tools, constraints mentioned",
  "dataset_description": "any dataset details if mentioned"
}}"""

    try:
        out = fast.invoke(prompt)
        parsed = safe_parse_json(out.content)

        if not parsed:
            print("  LLM parse failed — falling back to raw text")
            return {"clean_text": raw_text, "weights": {},
                    "deliverables": [], "tasks": []}

        # build clean PS text from structured output
        clean_parts = []

        if parsed.get("problem_statement"):
            clean_parts.append(parsed["problem_statement"])

        if parsed.get("technical_requirements"):
            clean_parts.append(parsed["technical_requirements"])

        if parsed.get("tasks"):
            for t in parsed["tasks"]:
                clean_parts.append(
                    f"Task {t.get('number', '?')}: {t.get('name', '')}. "
                    f"{t.get('description', '')}")

        if parsed.get("dataset_description"):
            clean_parts.append(parsed["dataset_description"])

        clean_text = "\n\n".join(clean_parts)

        # weights: prefer evaluation_criteria, fall back to per-task weights
        weights = {}
        if parsed.get("evaluation_criteria"):
            for ec in parsed["evaluation_criteria"]:
                name = ec.get("criterion", "")
                w = ec.get("weight", 0)
                if name and w:
                    weights[name] = w

        if not weights and parsed.get("tasks"):
            for t in parsed["tasks"]:
                name = t.get("name", "")
                w = t.get("weight", 0)
                if w:
                    weights[name] = w

        deliverables = parsed.get("deliverables", [])

        if weights:
            print("\n  WEIGHTS:")
            for c, w in sorted(weights.items(), key=lambda kv: -kv[1]):
                print(f"    {w:3d}%  {c}")

        if deliverables:
            print("\n  DELIVERABLES:")
            for d in deliverables[:6]:
                print(f"    • {d[:80]}")

        print(f"\n  Clean PS: {len(clean_text)} chars")

        return {
            "clean_text": clean_text,
            "weights": weights,
            "deliverables": deliverables,
            "tasks": parsed.get("tasks", []),
        }

    except Exception as e:
        print(f"  parse_ps_pdf error: {e} — falling back to raw text")
        return {"clean_text": raw_text, "weights": {},
                "deliverables": [], "tasks": []}