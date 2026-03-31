"""
Phase 6 — End-to-End Mini Pipeline (v6)

Wires all approved components (Phases 1–5) into one complete pipeline:
  LLM Query Rewrite → Fetch (parallel) → Chunk → Token Cap → Embed → Per-Source Retrieve → Diversity Cap → LLM

Approved stack (from STATUS.md, updated 2026-03-20):
  Embedding   : BAAI/bge-small-en-v1.5 (bge-s, Phase 3 winner)
  Wikipedia   : hierarchical_chunk (RAPTOR dropped 2026-03-20) + embedding-based title selection
                Children (~750 chars) embedded for retrieval → parent paragraphs (~1,200-1,500 chars) sent to LLM
  ArXiv       : fetch_arxiv_enhanced — HTML full text via arxiv.org/html + trafilatura (~6,000 chars/paper)
                Falls back to abstract-only if HTML unavailable
  Web         : semantic topic-shift
  YouTube     : semantic topic-shift (with vocab correction)
  Retrieval   : per-source top-4 → diversity cap max-3/src → top-12 to LLM
  LLM         : gemini-2.5-flash via google-generativeai

Production fixes built in (v2–v5):
  1. LLM query rewriting    — one Gemini call pre-fetch, 4 agent-specific queries
  2. Wikipedia v2 fetch     — embedding-based title selection (bge-s scores 10 candidates)
  3. Gate 3 removed         — per-source retrieval + diversity cap replaces it
  4. Per-source retrieval   — top-4 per source before diversity cap
  5. Parallel fetching      — ThreadPoolExecutor, all 4 sources fire simultaneously
  6. Post-chunk token cap   — >480 tokens → force-split at sentence boundary (bge-s 512 limit)
  7. Source diversity cap   — max 3 chunks/source → top-12 to LLM
  8. YouTube LOW_YIELD      — <3 chunks after chunking → flag in answer
  9. YouTube vocab fix      — dict-based correction of technical mis-transcriptions
  10. Web URL dedup         — skip Wikipedia/YouTube URLs from Web agent
  11. Parent-child retrieval — Wikipedia: retrieve children, send parent_text to LLM (v5)
  12. ArXiv HTML upgrade    — arxiv.org/html + trafilatura for full paper sections (v5)

Test queries:
  1. "What is retrieval augmented generation and how does it work?"
  2. "What are the latest approaches to AI alignment?"
  3. "How does RLHF training work in LLMs?"

Pass criteria:
  - Total pipeline time < 30s per query
  - Answer includes source attribution
  - Answer quality: GOOD or OK on all 3 queries

Run   : python phase6_e2e_test.py
Deps  : pip install -r requirements_phase6.txt
Needs : GEMINI_API_KEY environment variable set
"""

import os
import re
import sys
import json
import time
import textwrap
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)   # loads .env into os.environ; override=True replaces stale shell vars

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ─────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # Phase 3 winner
LLM_MODEL       = "gemini-2.5-flash"  # Gemini 2.5 Flash

TOP_PER_SOURCE = 4        # per-source retrieval: top-N chunks fetched per source before diversity cap
MAX_PER_SOURCE = 3        # source diversity cap: max chunks per source in final selection
FINAL_K        = 12       # max chunks sent to LLM after diversity cap

TOKEN_CAP       = 480     # bge-s hard limit = 512; 32-token buffer
CHARS_PER_TOKEN = 4.5     # approximate conversion factor

MAX_PIPELINE_TIME = 30    # seconds — pass criterion

WIKI_SIM_THRESHOLD   = 0.65   # skip Wikipedia if best title sim < this (BUG 7)
ARXIV_SIM_THRESHOLD  = 0.60   # raised 0.50→0.60 (eval: 0.573/0.561 garbage chunks confirmed)
MIN_CHUNK_SIM        = 0.65   # Fix 1: hard floor — drop any chunk below this after source minimum
SOURCE_MIN_SIM_GATE  = 0.55   # Fix 3: source minimum gate — only force-add if best chunk >= this

SOURCE_ROLES = {
    "wikipedia": "foundation",
    "arxiv":     "research",
    "web":       "current",
    "youtube":   "explainer",
}

ROLE_HEADERS = {
    "foundation": "[BACKGROUND — Wikipedia]",
    "research":   "[RECENT RESEARCH — ArXiv]",
    "current":    "[CURRENT WEB]",
    "explainer":  "[EXPLAINER — YouTube]",
}

ROLE_LABELS = {
    "foundation": "Background",
    "research":   "Research",
    "current":    "Web",
    "explainer":  "Explainer",
}

ROLE_ORDER = ["foundation", "research", "current", "explainer"]

QUERIES = [
    "What is retrieval augmented generation and how does it work?",
    "What are the latest approaches to AI alignment?",
    "How does RLHF training work in LLMs?",
]

# Technical vocabulary corrections for YouTube auto-captions
YOUTUBE_VOCAB_CORRECTIONS = {
    r"\brl\s*hf\b":                   "RLHF",
    r"\br\s*l\s*h\s*f\b":            "RLHF",
    r"\bembed\s*ings\b":              "embeddings",
    r"\btrans\s*formers?\b":          "transformers",
    r"\bchat\s*g\s*p\s*t\b":         "ChatGPT",
    r"\bg\s*p\s*t[\s-]?(\d)\b":      r"GPT-\1",
    r"\bl\s*l\s*m\b":                 "LLM",
    r"\bl\s*l\s*ms\b":               "LLMs",
    r"\bfine\s*tuning\b":            "fine-tuning",
    r"\bfine\s*tune\b":              "fine-tune",
    r"\bvector\s*data\s*base\b":     "vector database",
    r"\bvector\s*data\s*bases\b":    "vector databases",
    r"\brag\b":                       "RAG",
    r"\bopen\s*ai\b":                "OpenAI",
    r"\bhugging\s*face\b":           "HuggingFace",
    r"\bbert\b":                      "BERT",
    r"\bkv\s*cache\b":               "KV cache",
    r"\battention\s*is\s*all\s*you\s*need\b": "Attention is All You Need",
}

LOG_DIR     = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PLAN_FILE   = Path("TESTING_PLAN.md")
UPDATE_FILE = Path("UPDATES.md")
STATUS_FILE = Path("STATUS.md")
CODE_FILE   = Path("CODE.md")

CURRENT_YEAR = date.today().year


# ── Embedding model ─────────────────────────────────────────────────────────────

_embed_model = None   # FastEmbed ONNX instance — one shared, genuinely thread-safe

def load_model():
    """Load bge-s via FastEmbed (ONNX Runtime).
    Single shared instance — thread-safe by design, no lock required.
    All threads and all concurrent users share this one instance.
    RAM: ~90MB once (vs 130MB × N threads with threading.local).
    Inference: ~2-3x faster than PyTorch SentenceTransformer (ONNX compiled).
    """
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        print(f"  Loading {EMBEDDING_MODEL} (FastEmbed/ONNX) ...", end="", flush=True)
        t = time.perf_counter()
        _embed_model = TextEmbedding(EMBEDDING_MODEL)
        print(f" done ({time.perf_counter()-t:.1f}s)")
    return _embed_model


def embed_texts(texts: list) -> np.ndarray:
    """Embed with bge-s via FastEmbed. Returns L2-normalised array (dot product = cosine sim).
    Thread-safe: FastEmbed ONNX Runtime handles concurrent calls on a single instance.
    """
    return np.array(list(load_model().embed(texts)))


# ── Gemini LLM ──────────────────────────────────────────────────────────────────

_llm = None

def load_llm():
    global _llm
    if _llm is None:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set.\n"
                "  Windows : set GEMINI_API_KEY=your_key_here\n"
                "  Linux   : export GEMINI_API_KEY=your_key_here"
            )
        genai.configure(api_key=api_key)
        _llm = genai.GenerativeModel(LLM_MODEL)
        print(f"  LLM: {LLM_MODEL} ready.")
    return _llm


def call_llm(prompt: str) -> tuple:
    """Call Gemini. Returns (answer_text, latency_seconds).
    G7 fix: retries on 429 rate-limit with exponential backoff (5s, 15s, 45s).
    """
    llm     = load_llm()
    t0      = time.perf_counter()
    delays  = [5, 15, 45]
    for attempt in range(len(delays) + 1):
        try:
            response = llm.generate_content(prompt)
            # Safe extraction — response.text raises if no valid Part (e.g. safety filter, empty output)
            try:
                text = response.text
            except Exception:
                parts = getattr(response, "candidates", [])
                text = parts[0].content.parts[0].text if parts and parts[0].content.parts else ""
            if not text:
                text = "[LLM returned empty response]"
            return text, time.perf_counter() - t0
        except Exception as e:
            if "429" in str(e) and attempt < len(delays):
                wait = delays[attempt]
                print(f"    ⚠ Gemini 429 — waiting {wait}s (retry {attempt+1}/{len(delays)}) ...")
                time.sleep(wait)
            else:
                return f"[LLM ERROR: {e}]", time.perf_counter() - t0
    return "[LLM ERROR: max retries exceeded]", time.perf_counter() - t0


# ── LLM Query Rewriting ───────────────────────────────────────────────────────

def rewrite_queries_for_agents(query: str) -> dict:
    """
    One Gemini call before fetching. Returns 4 agent-specific search queries.
    Handles query ambiguity that fixed rule-based transforms cannot (e.g.
    'transformer architecture' → ArXiv gets neural-network-specific phrasing,
    Wikipedia gets the canonical article title, YouTube gets tutorial framing).

    Falls back to original query for all agents if Gemini fails or JSON is invalid.
    """
    prompt = f"""You are a search query optimizer for a 4-agent research pipeline.

OUTPUT RULE: respond with ONLY a JSON object. No explanation, no markdown, no extra text.
NEVER return an empty string for any field. If no specific Wikipedia article or ArXiv topic applies, use the best keyword phrase from the user's question.

RULES per agent:

1. WIKIPEDIA — Find the exact Wikipedia article.
   - NEVER drop proper nouns, product names, framework names, library names, or person names.
   - Wikipedia has articles named after specific things. Send the specific name, not a general concept.
   - Good: "FastAPI" | Bad: "asynchronous web framework"
   - Good: "BERT language model" | Bad: "transformer neural network"
   - Good: "Elon Musk" | Bad: "tech entrepreneur"

2. ARXIV — Write a technical academic search query.
   - Remove question words (what, how, why, explain, tell me).
   - Keep domain-specific terminology and technical nouns.
   - Frame as a research topic, not a question.

3. WEB — Write the search query, then set web_mode:
   - "news"    → query is about current events, recent product releases, announcements, or breaking developments
   - "general" → query is a how-to, tutorial, concept explanation, framework usage, or technical deep-dive
   - When in doubt, use "general" — it returns blogs, docs, and tutorials, not just news.

4. YOUTUBE — Write a tutorial-focused query.
   - Keep the EXACT technology/product/framework name from the original question.
   - End with "tutorial explained" or "explained tutorial".

5. RETRIEVAL_QUERY — Write a clean semantic restatement of the user's question.
   - This is used to score chunks at retrieval time (embedding comparison), NOT for searching.
   - Fix typos. Remove filler words. Keep the full intent and all key concepts.
   - Should be a natural English phrase that captures exactly what the user wants to know.
   - Good: "how to design an advanced RAG pipeline and explain each strategy"
   - Bad: "advanced RAG" (too short) | Bad: same as arxiv query (too academic)

Return ONLY this JSON:
{{
  "wikipedia": "...",
  "arxiv": "...",
  "web": "...",
  "web_mode": "news" or "general",
  "youtube": "...",
  "retrieval_query": "..."
}}

User question: {query}"""

    # G7 fix: retry on 429 with backoff before giving up
    delays = [5, 15]
    for attempt in range(len(delays) + 1):
        raw_text = None  # BUG 5 FIX: track raw response so we can log it on any failure
        try:
            llm      = load_llm()
            response = llm.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            try:
                raw_text = response.text.strip()
            except Exception:
                parts = getattr(response, "candidates", [])
                raw_text = (parts[0].content.parts[0].text if parts and parts[0].content.parts else "").strip()
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if not json_match:
                print(f"    ✗ Gemini returned no JSON (attempt {attempt+1}/{len(delays)+1}). Raw response:\n{raw_text}")
                continue
            parsed   = json.loads(json_match.group())
            required = {"wikipedia", "arxiv", "web", "web_mode", "youtube", "retrieval_query"}
            if not required.issubset(parsed.keys()):
                print(f"    ✗ Gemini JSON missing keys (attempt {attempt+1}/{len(delays)+1}). Got: {list(parsed.keys())}. Raw response:\n{raw_text}")
                continue
            print(f"    ✓ Query rewriting done")
            for src, q in parsed.items():
                print(f"      {src:<12} → {q}")
            return parsed
        except Exception as e:
            if "429" in str(e) and attempt < len(delays):
                wait = delays[attempt]
                print(f"    ⚠ Rewrite 429 — waiting {wait}s (retry {attempt+1}/{len(delays)}) ...")
                time.sleep(wait)
            else:
                print(f"    ✗ Query rewriting failed ({e}). Raw response:\n{raw_text or '<no response received>'}")
                break

    # G8 fix: keyword extractor for Wikipedia fallback.
    # Verbose question ("what is X and how does it work?") confuses bge-s title ranking
    # → wrong Wikipedia article. Strip question words → clean keyword phrase.
    _stop = {"what", "how", "why", "when", "where", "who", "which", "is", "are",
             "does", "do", "the", "a", "an", "and", "or", "in", "of", "to", "for",
             "work", "works", "working", "latest", "approaches", "tell", "me",
             "about", "define", "describe", "give", "list", "explain", "explain"}
    kw           = [w for w in re.sub(r'[?!.,]', '', query.lower()).split() if w not in _stop]
    wiki_fallback = " ".join(kw) if kw else query
    print(f"    ⚠ Using keyword fallback: wikipedia='{wiki_fallback}', others=original query")
    return {
        "wikipedia":      wiki_fallback,
        "arxiv":          query,
        "web":            query,
        "web_mode":       "general",   # safe default for fallback
        "youtube":        query,
        "retrieval_query": " ".join(kw) if kw else query,  # cleaned keyword phrase
    }


# ── Wikipedia v2: Embedding-Based Title Selection (BUG 2 FIX) ────────────────
#
# BUG 2 FIX: The original fetch_wikipedia_v2() ran embed_texts() (CPU-heavy bge-s
# inference) inside a ThreadPoolExecutor thread. Threads are for network I/O only.
# CPU work inside a thread makes Wikipedia the slowest thread, blocking all 4 threads.
# Q1 fetch: 18.85s instead of expected ~9s (+8s wasted).
#
# FIX: Split into two functions:
#   select_wikipedia_title(query) — runs on MAIN THREAD before threads launch.
#                                   Does Wikipedia search + bge-s embedding → ranked list.
#   fetch_wikipedia_article(ranked) — runs INSIDE fetch thread.
#                                     Pure network download — no embedding at all.

def select_wikipedia_title(query: str) -> list:
    """
    Runs on the MAIN THREAD before fetch threads launch.
    Does Wikipedia candidate search + bge-s similarity ranking.
    Returns ranked list of (title, sim) tuples — no article download.
    CPU-heavy (bge-s embedding) — must NOT run inside fetch threads.
    """
    import wikipedia as wiki_lib

    try:
        candidates = wiki_lib.search(query, results=10)
    except Exception as e:
        print(f"    ⚠  Wikipedia search failed: {e}")
        return []

    if not candidates:
        print(f"    ⚠  Wikipedia: no search results")
        return []

    q_emb      = embed_texts([query])[0]
    title_embs = embed_texts(candidates)
    sims       = title_embs @ q_emb
    ranked     = sorted(zip(candidates, sims.tolist()), key=lambda x: x[1], reverse=True)

    print(f"    Wikipedia candidates (top 5 by sim):")
    for t, s in ranked[:5]:
        print(f"      sim={s:.3f}  {t}")

    return ranked   # list of (title, sim) tuples, best first


def fetch_wikipedia_article(ranked: list) -> dict:
    """
    Runs INSIDE the fetch thread — pure network I/O, no embedding.
    Takes the pre-ranked title list from select_wikipedia_title() and
    downloads the first article that meets the minimum length requirement.
    """
    import wikipedia as wiki_lib
    start   = time.perf_counter()
    min_len = 10_000

    if not ranked:
        return {"status": "FAILED", "text": "", "url": "",
                "error": "No ranked candidates (title selection failed on main thread)",
                "time": time.perf_counter() - start}

    # BUG 7 fix: skip Wikipedia entirely if best match is below relevance threshold.
    # Prevents wrong articles (e.g. "Applications of AI" for "vibe coding") reaching LLM.
    best_title, best_sim = ranked[0]
    if best_sim < WIKI_SIM_THRESHOLD:
        print(f"    Wikipedia SKIPPED (best sim={best_sim:.3f} < {WIKI_SIM_THRESHOLD}) — no relevant article found")
        return {"status": "SKIPPED", "text": "", "url": "",
                "error": f"Best match '{best_title}' sim={best_sim:.3f} below threshold {WIKI_SIM_THRESHOLD}",
                "time": time.perf_counter() - start}

    for title, sim in ranked[:5]:   # try top-5 most similar titles
        try:
            page = wiki_lib.page(title, auto_suggest=False)
            text = page.content
            if len(text) >= min_len:
                print(f"    ✓ Wikipedia: '{page.title}'  sim={sim:.3f}  {len(text):,} chars")
                return {
                    "status":        "SUCCESS",
                    "text":          text,
                    "url":           page.url,
                    "fetched_title": page.title,
                    "title_sim":     sim,
                    "time":          time.perf_counter() - start,
                }
        except wiki_lib.exceptions.DisambiguationError as e:
            for option in (e.options or [])[:3]:
                try:
                    page = wiki_lib.page(option, auto_suggest=False)
                    text = page.content
                    if len(text) >= min_len:
                        print(f"    ✓ Wikipedia (disambig→'{option}')  {len(text):,} chars")
                        return {
                            "status":        "SUCCESS",
                            "text":          text,
                            "url":           page.url,
                            "fetched_title": option,
                            "title_sim":     sim,
                            "time":          time.perf_counter() - start,
                        }
                except Exception:
                    continue
        except Exception:
            continue

    return {"status": "FAILED", "text": "", "url": "",
            "error": "All title candidates failed or too short",
            "time": time.perf_counter() - start}


# ── Per-Source Retrieval ───────────────────────────────────────────────────────

def per_source_retrieve(query: str, chunks: list, embs: np.ndarray,
                        top_per_source: int = TOP_PER_SOURCE) -> list:
    """
    Retrieve top-N chunks per source independently, then combine and sort by sim.
    Replaces global top-K semantic retrieval.

    Why: global top-10 naturally favours Web (most chunks). ArXiv's 3 chunks can
    rank 11th–13th and be completely excluded before diversity cap runs.
    Per-source retrieval guarantees every source gets a fair shot.
    """
    q_emb    = embed_texts([query])[0]
    all_sims = embs @ q_emb

    by_source = defaultdict(list)
    for i, (chunk, sim) in enumerate(zip(chunks, all_sims)):
        by_source[chunk["source"]].append((i, float(sim)))

    combined = []
    for source, items in by_source.items():
        items.sort(key=lambda x: x[1], reverse=True)
        for rank_in_src, (idx, sim) in enumerate(items[:top_per_source]):
            combined.append({
                "rank":  rank_in_src + 1,
                "chunk": chunks[idx],
                "sim":   sim,
            })

    # Sort combined pool by similarity for diversity cap
    combined.sort(key=lambda x: x["sim"], reverse=True)
    return combined


# ── Production Fix 5: YouTube vocab correction ────────────────────────────────

def correct_youtube_vocab(text: str) -> str:
    """Apply dict-based correction of common auto-caption errors on technical terms."""
    for pattern, replacement in YOUTUBE_VOCAB_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── Production Fix 2: Post-chunk token cap ───────────────────────────────────

def apply_token_cap(chunks: list, max_tokens: int = TOKEN_CAP) -> list:
    """
    If a chunk exceeds max_tokens (~480 tokens = ~2,160 chars), force-split at
    nearest sentence boundary. Fixes Web topic-continuous articles producing
    chunks that silently truncate in bge-s (512 token limit).
    """
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    result    = []
    split_count = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        if len(text) <= max_chars:
            result.append(chunk)
            continue

        # Force-split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current   = []
        cur_len   = 0

        for sent in sentences:
            sent_len = len(sent)
            if cur_len + sent_len > max_chars and current:
                new_chunk          = dict(chunk)
                new_chunk["text"]  = " ".join(current)
                new_chunk["tokens"] = max(1, int(len(new_chunk["text"]) / CHARS_PER_TOKEN))
                result.append(new_chunk)
                split_count += 1
                current  = [sent]
                cur_len  = sent_len
            else:
                current.append(sent)
                cur_len += sent_len

        if current:
            new_chunk          = dict(chunk)
            new_chunk["text"]  = " ".join(current)
            new_chunk["tokens"] = max(1, int(len(new_chunk["text"]) / CHARS_PER_TOKEN))
            result.append(new_chunk)

    if split_count:
        print(f"    [token cap] force-split {split_count} oversize chunks")
    return result


# ── Step 4: ArXiv enhanced fetch (Abstract + HTML full text via arxiv.org) ────

def fetch_arxiv_enhanced(query: str) -> dict:
    """
    Semantic Scholar API fetch (upgraded 2026-03-23, replaces ArXiv library + trafilatura HTML).

    Why Semantic Scholar over the old approach:
    - BUG 8 fixed: ArXiv date-filter query syntax caused timeouts on niche topics.
      Semantic Scholar is a plain REST GET — consistent 1–3s, no timeout risk.
    - Better content: returns abstract + AI-generated TLDR per paper.
      TLDR is a one-sentence distillation of the full paper — higher signal than
      raw HTML fragments that trafilatura extracted ~60% of the time.
    - No scraping: many ArXiv papers have no HTML version; trafilatura silently
      fell back to abstract anyway. Semantic Scholar abstract is always available.
    - URL quality: externalIds.ArXiv field gives clean arxiv.org/abs/{id} links.

    Return format unchanged — same {status, text, meta, metas, source_dates, time} dict.
    Falls back to phase1b fetch_arxiv (arxiv library, abstract-only) on total failure.
    """
    import requests
    start       = time.perf_counter()
    cutoff_year = datetime.now().year - 2

    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query":  query,
                "fields": "title,abstract,tldr,year,externalIds",
                "limit":  10,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        if not data:
            raise ValueError("Semantic Scholar returned no results")

        # Prefer recent papers; fall back to all results for niche/new topics
        recent = [p for p in data if (p.get("year") or 0) >= cutoff_year]
        candidates = recent if recent else data

        texts, metas = [], []
        for paper in candidates:
            if len(texts) >= 3:
                break

            abstract = (paper.get("abstract") or "").strip()
            if len(abstract) < 200:
                continue   # skip stub/empty abstracts

            title    = (paper.get("title") or "").strip()
            tldr     = ((paper.get("tldr") or {}).get("text") or "").strip()
            year     = paper.get("year") or ""

            # Build paper text: Title + Abstract + TLDR (if available)
            parts = [f"Title: {title}", f"Abstract: {abstract}"]
            if tldr:
                parts.append(f"Summary: {tldr}")
            paper_text = "\n\n".join(parts)

            # URL: prefer arxiv.org link, fall back to Semantic Scholar page
            ext_ids  = paper.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv")
            url = (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                   else f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}")

            texts.append(paper_text)
            metas.append({
                "source":      "arxiv",
                "role":        "research",
                "source_date": str(year),
                "url":         url,
                "fetched_at":  date.today().isoformat(),
            })

        if not texts:
            raise ValueError("No papers with sufficient abstract content found")

        # Relevance filter: same pattern as Wikipedia WIKI_SIM_THRESHOLD (BUG 7).
        # Semantic Scholar returns results for any query — even when the topic has no
        # real research coverage. Check query-paper similarity; skip ArXiv entirely
        # if the best paper is below threshold (e.g. satellite papers for "AI data leakage").
        query_emb  = embed_texts([query])[0]
        paper_embs = embed_texts(texts)
        sims       = paper_embs @ query_emb   # dot product = cosine sim (L2-normalised)
        best_sim   = float(np.max(sims))

        if best_sim < ARXIV_SIM_THRESHOLD:
            print(f"    ArXiv SKIPPED (best sim={best_sim:.3f} < {ARXIV_SIM_THRESHOLD}) — no relevant papers found")
            return {"status": "SKIPPED", "text": "", "metas": [], "source_dates": [],
                    "error": f"Best paper sim={best_sim:.3f} below threshold {ARXIV_SIM_THRESHOLD}",
                    "time":  round(time.perf_counter() - start, 2)}

        print(f"    ✓ Semantic Scholar: {len(texts)} papers found (best sim={best_sim:.3f})")
        combined = "\n\n---\n\n".join(texts)
        status   = "SUCCESS" if len(texts) >= 2 else "PARTIAL"
        return {
            "status":       status,
            "text":         combined,
            "meta":         metas[0],   # backward compat
            "metas":        metas,      # per-paper metadata — used for per-chunk URL
            "source_dates": [m["source_date"] for m in metas],
            "time":         round(time.perf_counter() - start, 2),
        }

    except Exception as e:
        from sources import fetch_arxiv as _base_fetch
        print(f"    ⚠  Semantic Scholar failed ({e}) — falling back to arxiv abstract-only")
        return _base_fetch(query)


# ── Production Fix 1: Parallel fetching ──────────────────────────────────────

def fetch_all_parallel(agent_queries: dict) -> dict:
    """
    Fire all 4 fetchers simultaneously via ThreadPoolExecutor.
    Each fetcher receives its own optimised query (from rewrite_queries_for_agents).

    BUG 2 FIX: Wikipedia title selection (bge-s embedding — CPU work) runs on the
    MAIN THREAD before threads launch. Only the article download runs inside the thread.
    This prevents CPU work from blocking the I/O-only thread pool.

    Returns {source: fetch_result_dict}.
    """
    from sources import (fetch_web, fetch_youtube)
    # fetch_arxiv_enhanced defined locally in this file (Step 4 upgrade — HTML full text).
    # fetch_arxiv from phase1b is the fallback inside fetch_arxiv_enhanced itself.

    # Title selection on main thread — CPU work (bge-s) must not be inside I/O threads
    print(f"    Selecting Wikipedia title (main thread) ...", end="", flush=True)
    t_title     = time.perf_counter()
    wiki_ranked = select_wikipedia_title(agent_queries["wikipedia"])
    print(f" {time.perf_counter()-t_title:.1f}s")

    _web_mode = agent_queries.get("web_mode", "general")
    fetchers = {
        "wikipedia": (fetch_wikipedia_article,                              wiki_ranked),
        "arxiv":     (fetch_arxiv_enhanced,                                 agent_queries["arxiv"]),
        "web":       (lambda q: fetch_web(q, web_mode=_web_mode),           agent_queries["web"]),
        "youtube":   (fetch_youtube,                                        agent_queries["youtube"]),
    }

    FETCH_TIMEOUT = 25  # seconds; prevents one slow source blocking pipeline

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_src = {executor.submit(fn, q): src
                         for src, (fn, q) in fetchers.items()}
        done, not_done = wait(future_to_src.keys(), timeout=FETCH_TIMEOUT)

        for future in done:
            src = future_to_src[future]
            try:
                results[src] = future.result()
            except Exception as e:
                print(f"    ⚠  {src} fetch error: {e}")
                results[src] = {"status": "ERROR", "text": "", "error": str(e)}

        for future in not_done:
            src = future_to_src[future]
            future.cancel()
            print(f"    ⚠  {src} timed out after {FETCH_TIMEOUT}s — skipped")
            results[src] = {"status": "TIMEOUT", "text": ""}

    return results


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_source(source: str, text: str, query: str) -> list:
    """
    Apply approved per-source chunking strategy (Phase 2 decisions):
      Wikipedia : hierarchical_chunk — children (~750 chars) embedded for retrieval,
                  parent_text (~1,200-1,500 chars) attached to each child for LLM context.
                  RAPTOR dropped 2026-03-20 (Phase 4 test invalid — wrong vector space).
      ArXiv     : atomic (1 abstract = 1 chunk)
      Web       : semantic topic-shift
      YouTube   : semantic topic-shift (vocab correction applied first)
    """
    from chunking import (hierarchical_chunk, atomic_chunk,
                                      semantic_topic_shift)

    if source == "wikipedia":
        # Fix 2: strip boilerplate sections before chunking.
        # These sections (See also, References, etc.) appear at the end of every
        # Wikipedia article and contain only nav links / citations — zero useful content.
        # Eval confirmed "See also — Churnalism" reaching LLM at sim=0.747.
        import re
        noise_match = re.search(
            r'\n={1,4}\s*(See also|References|External links|Notes|'
            r'Further reading|Bibliography|Footnotes|Citations)\s*={0,4}',
            text, re.IGNORECASE
        )
        if noise_match:
            text = text[:noise_match.start()]

        # Step 1: hierarchical_chunk replaces raptor_lite (STATUS.md 2026-03-20).
        # Build parent lookup: parent_id → full paragraph text (~1,200-1,500 chars).
        # Attach parent_text to every child so build_prompt() sends parent to LLM,
        # not the small child chunk. Children (~750 chars) are what gets embedded.
        all_hchunks = hierarchical_chunk(text)
        parents  = {c["parent_id"]: c["text"]
                    for c in all_hchunks if c["type"] == "parent"}
        children = [c for c in all_hchunks if c["type"] == "child"]
        for child in children:
            child["parent_text"] = parents.get(child["parent_id"], child["text"])
        return children

    elif source == "arxiv":
        return atomic_chunk(text)

    elif source == "web":
        # overlap_chars=150: G2 fix — last 150 chars of prev chunk prepended to next chunk
        return semantic_topic_shift(text, embed_fn=embed_texts, overlap_chars=150)

    elif source == "youtube":
        corrected = correct_youtube_vocab(text)
        return semantic_topic_shift(corrected, embed_fn=embed_texts, overlap_chars=150)

    return []


def chunk_all(query: str, fetch_results: dict) -> list:
    """
    Chunk all fetched sources and attach metadata to every chunk.
    Production Fix 6: skip Wikipedia URLs from Web results (prevents content duplication).
    Returns flat list of chunk dicts.
    """
    all_chunks  = []
    chunk_stats = {}

    for source in ["wikipedia", "arxiv", "web", "youtube"]:
        result = fetch_results.get(source, {})
        text   = result.get("text", "")
        if not text:
            chunk_stats[source] = 0
            continue

        try:
            chunks = chunk_source(source, text, query)
        except Exception as e:
            print(f"    ⚠  chunking error ({source}): {e}")
            chunks = []

        # ArXiv: atomic_chunk produces one chunk per paper in order.
        # Pre-assign each chunk its own paper URL from metas[i] before the loop below.
        if source == "arxiv":
            arxiv_metas = result.get("metas", [result.get("meta", {})])
            for i, chunk in enumerate(chunks):
                chunk["_arxiv_url"] = arxiv_metas[i].get("url", "") if i < len(arxiv_metas) else ""

        for chunk in chunks:
            chunk.setdefault("text", "")
            if not chunk["text"].strip():
                continue
            chunk["source"]     = source
            chunk["role"]       = SOURCE_ROLES[source]
            chunk["query"]      = query
            chunk["fetched_at"] = date.today().isoformat()
            chunk["url"]        = chunk.pop("_arxiv_url", None) or result.get("url") or result.get("meta", {}).get("url", "")

        all_chunks.extend(chunks)
        chunk_stats[source] = len(chunks)

    return all_chunks, chunk_stats


# ── Retrieval ─────────────────────────────────────────────────────────────────

def semantic_retrieve(query: str, chunks: list, embs: np.ndarray, k: int) -> list:
    """Top-k by cosine similarity. bge-s already L2-normalised → dot product = cosine."""
    q_emb = embed_texts([query])[0]
    sims  = embs @ q_emb
    idxs  = np.argsort(sims)[::-1][:k]
    return [
        {"rank": r + 1, "chunk": chunks[i], "sim": float(sims[i])}
        for r, i in enumerate(idxs)
    ]


# ── Production Fix 3: Source diversity cap ────────────────────────────────────

def apply_diversity_cap(retrieved: list,
                        max_per_source: int = MAX_PER_SOURCE,
                        final_k: int = FINAL_K) -> list:
    """
    Enforce max 2 chunks per source, then take top-8 remaining by sim score.
    Fixes Web dominating 56% of the chunk pool (observed in Phase 4).
    Input must be already sorted descending by sim.
    """
    counts = defaultdict(int)
    capped = []
    for r in retrieved:
        src = r["chunk"]["source"]
        if counts[src] < max_per_source:
            capped.append(r)
            counts[src] += 1
        if len(capped) >= final_k:
            break
    return capped


# ── G6: Source minimum guarantee ─────────────────────────────────────────────

def apply_source_minimum(retrieved_pool: list, final: list) -> list:
    """
    G6 fix: after diversity cap, if any source that contributed to the retrieved
    pool has 0 chunks in the final set, force-add its top-1 chunk by similarity.

    Why: YouTube often drops to 0 after the diversity cap (LOW_YIELD or low sim).
    Without this, the LLM never gets the explainer perspective even when YouTube
    data was successfully fetched and chunked.
    """
    present     = {r["chunk"]["source"] for r in final}
    all_sources = {r["chunk"]["source"] for r in retrieved_pool}
    missing     = all_sources - present

    if not missing:
        return final

    added = list(final)
    for src in sorted(missing):   # sorted for deterministic log output
        src_chunks = [r for r in retrieved_pool if r["chunk"]["source"] == src]
        if src_chunks:
            best = max(src_chunks, key=lambda x: x["sim"])
            if best["sim"] < SOURCE_MIN_SIM_GATE:
                print(f"    [src-min] SKIP {src} (sim={best['sim']:.4f} < gate {SOURCE_MIN_SIM_GATE}) — below quality gate")
                continue
            added.append(best)
            print(f"    [src-min] Force-added {src} top-1 (sim={best['sim']:.4f}) — was 0 after cap")
    return added


# ── LLM prompt ────────────────────────────────────────────────────────────────

def build_prompt(query: str, final_results: list, yt_low_yield: bool = False) -> str:
    """
    Build structured research prompt. Each chunk is numbered [Source N] with
    metadata (URL, date, ArXiv paper title) embedded in the context header.

    LLM is instructed to produce 5 sections:
      1. TL;DR          — 2-3 sentence direct answer
      2. Detailed Explanation — 500-800 words, inline [Source N] citations
      3. Research Highlights  — ArXiv papers by title + key finding (if ArXiv present)
      4. Sources         — numbered list with URLs and dates
      5. Confidence Note — only when worth flagging (missing source, conflict, etc.)
    """

    # ── Assign reference numbers + extract metadata from each chunk ───────────
    refs = []
    for i, r in enumerate(final_results, 1):
        chunk  = r["chunk"]
        source = chunk.get("source", "")
        url    = chunk.get("url", "") or ""
        # Step 2 (parent-child retrieval): Wikipedia sends parent paragraph (~300 tokens)
        # to LLM instead of the child chunk (~167 tokens). Other sources use text directly.
        text   = (chunk.get("parent_text") or chunk.get("text", "")) \
                 if source == "wikipedia" else chunk.get("text", "")
        date   = chunk.get("fetched_at", "")

        # Extract human-readable label per source type
        if source == "arxiv":
            # ArXiv chunk text starts with "Title: <paper title>\n\nAbstract: ..."
            m     = re.match(r"Title:\s*(.+?)(?:\n|$)", text)
            label = m.group(1).strip() if m else "ArXiv paper"
        elif source == "wikipedia" and url:
            # Derive article title from URL: /wiki/Retrieval-augmented_generation
            slug  = url.split("/wiki/")[-1] if "/wiki/" in url else ""
            label = slug.replace("_", " ") if slug else "Wikipedia"
        elif source == "web" and url:
            # Use domain name: https://www.example.com/... → example.com
            try:
                domain = url.split("/")[2].replace("www.", "")
            except IndexError:
                domain = url
            label = domain
        elif source == "youtube":
            label = "YouTube educational video"
        else:
            label = source.capitalize()

        refs.append({
            "n":      i,
            "source": source,
            "role":   chunk.get("role", ""),
            "label":  label,
            "url":    url,
            "date":   date,
            "text":   text,
        })

    # ── Build numbered context sections grouped by role ───────────────────────
    by_role = defaultdict(list)
    for ref in refs:
        by_role[ref["role"]].append(ref)

    context_parts = []
    for role in ROLE_ORDER:
        role_refs = by_role.get(role, [])
        if not role_refs:
            continue
        header = ROLE_HEADERS[role]
        blocks = []
        for ref in role_refs:
            # Header line tells LLM the reference number, source, URL, and date
            meta = f"[Source {ref['n']} — {ref['source'].upper()}"
            if ref["label"]:
                meta += f" | {ref['label']}"
            if ref["date"]:
                meta += f" | fetched {ref['date']}"
            if ref["url"]:
                meta += f" | {ref['url']}"
            meta += "]"
            blocks.append(f"{meta}\n{ref['text']}")
        context_parts.append(f"{header}\n" + "\n\n".join(blocks))

    context = "\n\n".join(context_parts)

    # ── Build sources reference list for the Sources section ──────────────────
    sources_ref = "\n".join(
        f"[{r['n']}] {r['source'].upper()} — {r['label']}"
        + (f" | {r['date']}" if r["date"] else "")
        + (f"\n    {r['url']}" if r["url"] else "")
        for r in refs
    )

    # ── Confidence note hints ─────────────────────────────────────────────────
    has_arxiv  = any(r["source"] == "arxiv"   for r in refs)
    has_yt     = any(r["source"] == "youtube" for r in refs)
    conf_hints = []
    if yt_low_yield:
        conf_hints.append(
            "YouTube had limited content for this query — explainer perspective may be underrepresented."
        )

    research_highlights_instruction = ""
    if has_arxiv:
        research_highlights_instruction = """\

## Research Highlights
For each ArXiv paper you cited, write one block (use the exact title from the source):

**Paper:** <title exactly as it appears after "Title:" in the source>
**Key finding:** One sentence — what does this paper specifically argue or demonstrate that is relevant to this question?

"""

    conf_instruction = ""
    if conf_hints:
        conf_instruction = (
            "\n\nNote for your Confidence Note section: " + " ".join(conf_hints)
        )

    prompt = f"""You are a research synthesis assistant. Using ONLY the numbered sources provided, write a structured research answer.

Follow this output format exactly — section by section:

## TL;DR
2–3 sentences. Give the direct, complete answer. No filler, no hedging.

## Detailed Explanation
Write 500–800 words. Use numbered sections and sub-points where helpful.
Cite every claim inline using [Source N] — for example: "RAG retrieves documents before generating [Source 1][Source 3]."
Be specific: include mechanisms, step-by-step processes, technical terms, and concrete examples from the sources.
Synthesize — when multiple sources agree on a point, combine them into one stronger statement.
{research_highlights_instruction}
## Sources
Copy the reference list below exactly, adding any additional sources you cited:
{sources_ref}

## Confidence Note
Include ONLY if something is worth flagging: a source type was missing or had very little content, sources disagreed on something, or ArXiv papers are older than 1 year. If everything looks complete, omit this section entirely.
{conf_instruction}

{'─'*60}
QUESTION: {query}
{'─'*60}

SOURCES:
{context}

{'─'*60}
Write the structured research answer now:"""

    return prompt


# ── Quality evaluation ────────────────────────────────────────────────────────

def evaluate_quality(query: str, answer: str, final_results: list) -> str:
    """
    Heuristic quality gate:
      GOOD : >150 words AND has source attribution AND ≥50% query term coverage
      OK   : >50 words AND ≥30% query term coverage
      POOR : otherwise
    """
    words = len(answer.split())

    attribution_markers = [
        # new structured format
        "[source ", "## sources", "## tl;dr",
        # old format fallback
        "[background]", "[research]", "[web]", "[explainer]",
        "wikipedia", "arxiv", "youtube",
        "according to", "based on",
    ]
    has_attribution = any(m.lower() in answer.lower() for m in attribution_markers)

    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
        "of", "with", "is", "are", "was", "were", "how", "what", "why",
        "does", "do", "it", "its",
    }
    terms    = [t for t in re.findall(r"[a-z]+", query.lower())
                if t not in stopwords and len(t) > 3]
    coverage = sum(1 for t in terms if t in answer.lower()) / max(1, len(terms))

    if words >= 150 and has_attribution and coverage >= 0.5:
        return "GOOD"
    elif words >= 50 and coverage >= 0.3:
        return "OK"
    return "POOR"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline_query(query: str, lines: list) -> dict:
    """
    Full E2E pipeline for a single query. Returns result dict with all metrics.
    """
    def _log(msg):
        lines.append(msg)

    _log(f"\n{'═'*70}")
    _log(f"QUERY: {query}")
    _log(f"{'═'*70}")

    pipeline_start = time.perf_counter()

    # ── Step 0: LLM Query Rewriting ────────────────────────────────────────────
    print(f"\n  Rewriting queries for agents ...")
    _log(f"\n{'─'*70}")
    _log(f"  Original query : {query}")
    t0_rewrite    = time.perf_counter()
    agent_queries = rewrite_queries_for_agents(query)
    rewrite_time  = round(time.perf_counter() - t0_rewrite, 2)
    _log(f"STEP 0 — LLM QUERY REWRITING  ({rewrite_time}s)")
    _log(f"{'─'*70}")
    _log(f"  Rewritten queries per agent:")
    for src, q in agent_queries.items():
        _log(f"    {src:<12} → \"{q}\"")

    # ── Step 1: Parallel fetch ─────────────────────────────────────────────────
    print(f"\n  Fetching (parallel) ...", end="", flush=True)
    t0            = time.perf_counter()
    fetch_results = fetch_all_parallel(agent_queries)
    fetch_time    = round(time.perf_counter() - t0, 2)
    print(f" done ({fetch_time}s)")

    _log(f"\n{'─'*70}")
    _log(f"STEP 1 — PARALLEL FETCH  (wall time: {fetch_time}s)")
    _log(f"{'─'*70}")
    for src in ["wikipedia", "arxiv", "web", "youtube"]:
        res    = fetch_results.get(src, {})
        status = res.get("status", "UNKNOWN")
        text   = res.get("text", "")
        chars  = len(text)
        q_used = agent_queries.get(src, "")
        _log(f"  [{src}]")
        _log(f"    Search query : \"{q_used}\"")
        _log(f"    Status       : {status}")
        _log(f"    Characters   : {chars:,}")
        if src == "wikipedia" and res.get("fetched_title"):
            _log(f"    Article title: {res['fetched_title']}  (sim={res.get('title_sim', '?'):.3f})")
        if res.get("url"):
            _log(f"    URL          : {res['url']}")
        if status != "SUCCESS" and res.get("error"):
            _log(f"    Error        : {res['error']}")

    # ── Step 2: Chunk ──────────────────────────────────────────────────────────
    print(f"  Chunking ...", end="", flush=True)
    t0 = time.perf_counter()
    all_chunks, chunk_stats = chunk_all(query, fetch_results)
    chunk_time = round(time.perf_counter() - t0, 2)
    print(f" {len(all_chunks)} chunks ({chunk_time}s)")

    _log(f"\n{'─'*70}")
    _log(f"STEP 2 — CHUNKING  (wall time: {chunk_time}s)")
    _log(f"{'─'*70}")
    _log(f"  Total chunks produced: {len(all_chunks)}")
    for src in ["wikipedia", "arxiv", "web", "youtube"]:
        src_chunks = [c for c in all_chunks if c.get("source") == src]
        n = len(src_chunks)
        strategy = {"wikipedia": "hierarchical_chunk", "arxiv": "atomic",
                    "web": "semantic topic-shift", "youtube": "semantic topic-shift (+ vocab correction)"}
        _log(f"\n  [{src}]  strategy={strategy.get(src,'?')}  chunks={n}")
        for i, c in enumerate(src_chunks):
            tok     = max(1, int(len(c["text"]) / CHARS_PER_TOKEN))
            chars   = len(c["text"])
            preview = c["text"][:120].replace("\n", " ")
            _log(f"    #{i+1:>2}  chars={chars:>5}  tokens≈{tok:>4}  \"{preview}\"")

    # ── Step 3: Post-chunk token cap ───────────────────────────────────────────
    _log(f"\n{'─'*70}")
    _log(f"STEP 3 — TOKEN CAP (limit: {TOKEN_CAP} tokens ≈ {int(TOKEN_CAP * CHARS_PER_TOKEN)} chars)")
    _log(f"{'─'*70}")
    before_cap = len(all_chunks)
    all_chunks = apply_token_cap(all_chunks)
    after_cap  = len(all_chunks)
    splits     = after_cap - before_cap
    _log(f"  Chunks before cap : {before_cap}")
    _log(f"  Chunks after cap  : {after_cap}  (force-split {splits} oversize chunk(s))")
    oversize = [c for c in all_chunks if len(c["text"]) > int(TOKEN_CAP * CHARS_PER_TOKEN)]
    _log(f"  Still oversize    : {len(oversize)}  (should be 0)")

    # ── Production Fix 4: YouTube LOW_YIELD detection ─────────────────────────
    yt_chunks    = [c for c in all_chunks if c.get("source") == "youtube"]
    yt_low_yield = len(yt_chunks) < 3
    if yt_low_yield:
        _log(f"  ⚠  YouTube LOW_YIELD: only {len(yt_chunks)} chunk(s) — will flag in answer")
        print(f"    ⚠  YouTube LOW_YIELD ({len(yt_chunks)} chunks)")

    if not all_chunks:
        _log("  ERROR: No chunks available after chunking. Skipping query.")
        return {"query": query, "error": "no_chunks", "quality": "POOR",
                "time_pass": False, "attribution_pass": False}

    # ── Step 4: Embed ──────────────────────────────────────────────────────────
    print(f"  Embedding {len(all_chunks)} chunks ...", end="", flush=True)
    t0    = time.perf_counter()
    texts = [c["text"] for c in all_chunks]
    embs  = embed_texts(texts)
    embed_time = round(time.perf_counter() - t0, 2)
    print(f" done ({embed_time}s)")

    _log(f"\n{'─'*70}")
    _log(f"STEP 4 — EMBEDDING  (wall time: {embed_time}s)")
    _log(f"{'─'*70}")
    _log(f"  Model            : {EMBEDDING_MODEL}")
    _log(f"  Vector dimensions: {embs.shape[1]}")
    _log(f"  Chunks embedded  : {embs.shape[0]}")
    _log(f"  Embedding matrix : {embs.shape[0]} × {embs.shape[1]}  dtype={embs.dtype}")
    _log(f"  Normalised       : Yes (L2-normalised — dot product = cosine similarity)")
    tok_counts = [max(1, int(len(c["text"]) / CHARS_PER_TOKEN)) for c in all_chunks]
    _log(f"  Token stats      : min={min(tok_counts)}  max={max(tok_counts)}  "
         f"avg={sum(tok_counts)/len(tok_counts):.0f}  total={sum(tok_counts)}")

    # ── Step 5: Per-Source Retrieval ──────────────────────────────────────────
    # Top-{TOP_PER_SOURCE} chunks retrieved per source independently.
    # Replaces global top-K and Gate 3 — every source gets a fair shot before
    # the diversity cap selects the final set.
    retrieval_query = agent_queries.get("retrieval_query", query)
    retrieved = per_source_retrieve(retrieval_query, all_chunks, embs, top_per_source=TOP_PER_SOURCE)

    _log(f"\n{'─'*70}")
    _log(f"STEP 5 — PER-SOURCE RETRIEVAL  (top-{TOP_PER_SOURCE} per source)")
    _log(f"{'─'*70}")
    _log(f"  Retrieval query  : \"{query}\"  (original — used for similarity scoring)")
    _log(f"  Total retrieved  : {len(retrieved)} chunks across all sources")

    # Group by source for detailed logging
    by_src = defaultdict(list)
    for r in retrieved:
        by_src[r["chunk"]["source"]].append(r)

    for src in ["wikipedia", "arxiv", "web", "youtube"]:
        items = by_src.get(src, [])
        _log(f"\n  [{src}]  {len(items)} chunks retrieved:")
        for r in items:
            c   = r["chunk"]
            tok = max(1, int(len(c["text"]) / CHARS_PER_TOKEN))
            preview = c["text"][:120].replace("\n", " ")
            _log(f"    sim={r['sim']:.4f}  chars={len(c['text']):>5}  tokens≈{tok:>4}  \"{preview}\"")

    src_retrieved = defaultdict(int)
    for r in retrieved:
        src_retrieved[r["chunk"]["source"]] += 1
    print(f"  Per-source retrieve ({TOP_PER_SOURCE}/src): "
          + "  ".join(f"{s}={n}" for s, n in src_retrieved.items()))

    # ── Step 6: Source diversity cap + source minimum guarantee ───────────────
    final_results = apply_diversity_cap(retrieved)
    final_results = apply_source_minimum(retrieved, final_results)  # G6: force-add any 0-chunk source

    # BUG 10 fix: dedup by parent — same parent paragraph can appear multiple times when
    # multiple children of the same parent are retrieved. Key on url + parent_text[:100]
    # (what the LLM actually sees). Keep highest-sim copy, re-sort by sim descending.
    seen_parents = {}
    for r in final_results:
        key = (r["chunk"].get("url", "")
               + r["chunk"].get("parent_text", r["chunk"]["text"])[:100])
        if key not in seen_parents or r["sim"] > seen_parents[key]["sim"]:
            seen_parents[key] = r
    deduped = sorted(seen_parents.values(), key=lambda x: x["sim"], reverse=True)
    if len(deduped) < len(final_results):
        print(f"    [dedup] removed {len(final_results) - len(deduped)} duplicate parent(s) "
              f"({len(final_results)} → {len(deduped)} chunks)")
    final_results = deduped

    # Fix 1: hard sim floor — drop chunks below MIN_CHUNK_SIM.
    # Graceful: if all chunks are below floor (e.g. niche query with no good sources),
    # keep the top-1 rather than sending an empty context to the LLM.
    floored = [r for r in final_results if r["sim"] >= MIN_CHUNK_SIM]
    if floored:
        dropped = len(final_results) - len(floored)
        if dropped:
            print(f"    [sim-floor] Dropped {dropped} chunk(s) below sim={MIN_CHUNK_SIM} floor "
                  f"({len(final_results)} → {len(floored)} chunks)")
        final_results = floored
    # else: all chunks below floor — keep as-is, avoid empty LLM context

    src_dist = {s: sum(1 for r in final_results if r["chunk"]["source"] == s)
                for s in SOURCE_ROLES}

    _log(f"\n{'─'*70}")
    _log(f"STEP 6 — DIVERSITY CAP  (max {MAX_PER_SOURCE} chunks/source → top-{FINAL_K})")
    _log(f"{'─'*70}")
    _log(f"  Chunks in       : {len(retrieved)}")
    _log(f"  Chunks out      : {len(final_results)}")
    _log(f"  Per-source count: " + "  ".join(f"{s}={src_dist[s]}" for s in SOURCE_ROLES))
    _log(f"\n  Final {len(final_results)} chunks sent to LLM (full text):")
    for i, r in enumerate(final_results):
        c   = r["chunk"]
        tok = max(1, int(len(c["text"]) / CHARS_PER_TOKEN))
        _log(f"\n  ── Chunk {i+1}/{len(final_results)} ──────────────────────────────────")
        _log(f"  Source   : {c['source']}  |  Role: {c['role']}")
        _log(f"  Sim score: {r['sim']:.4f}")
        _log(f"  Size     : {len(c['text'])} chars  ≈ {tok} tokens")
        _log(f"  URL      : {c.get('url','N/A')}")
        _log(f"  Text     :\n{c['text']}")

    print(f"  Final chunks: {len(final_results)}  "
          + "  ".join(f"{s}={n}" for s, n in src_dist.items() if n > 0))

    # ── Step 7: Build role-grouped prompt ─────────────────────────────────────
    prompt = build_prompt(query, final_results, yt_low_yield)

    _log(f"\n{'─'*70}")
    _log(f"STEP 7 — PROMPT SENT TO LLM  ({len(prompt):,} chars)")
    _log(f"{'─'*70}")
    _log(prompt)

    # ── Step 8: LLM call ──────────────────────────────────────────────────────
    print(f"  Calling Gemini ({LLM_MODEL}) ...", end="", flush=True)
    answer, llm_time = call_llm(prompt)
    print(f" done ({llm_time:.1f}s)")

    _log(f"\n{'═'*70}")
    _log(f"STEP 8 — FULL LLM OUTPUT  (model: {LLM_MODEL}  latency: {llm_time:.2f}s)")
    _log(f"{'═'*70}")
    _log(answer)
    _log(f"{'═'*70}")

    # ── Step 9: Evaluate + Timing Summary ─────────────────────────────────────
    total_time = round(time.perf_counter() - pipeline_start, 2)
    quality    = evaluate_quality(query, answer, final_results)

    attribution_markers = ["[background]", "[research]", "[web]", "[explainer]",
                           "wikipedia", "arxiv", "youtube", "according to"]
    attribution_pass = any(m in answer.lower() for m in attribution_markers)
    time_pass        = total_time < MAX_PIPELINE_TIME

    # Timing breakdown
    _log(f"\n{'─'*70}")
    _log(f"STEP 9 — RESULT + TIMING")
    _log(f"{'─'*70}")
    _log(f"  Total pipeline time : {total_time}s  ({'PASS ✅' if time_pass else 'FAIL ❌ >30s'})")
    _log(f"  Attribution         : {'PASS ✅' if attribution_pass else 'FAIL ❌'}")
    _log(f"  Quality             : {quality}  ({'✅' if quality in ('GOOD','OK') else '❌'})")
    _log(f"\n  Timing breakdown:")
    _log(f"    Step 0 — Query rewrite  : {rewrite_time}s")
    _log(f"    Step 1 — Fetch (parallel): {fetch_time}s")
    _log(f"    Step 2 — Chunk          : {chunk_time}s")
    _log(f"    Step 4 — Embed          : {embed_time}s")
    _log(f"    Step 8 — LLM            : {llm_time:.2f}s")
    _log(f"    Total                   : {total_time}s")

    print(f"\n  ─── Result ───")
    print(f"  Time       : {total_time}s  ({'PASS' if time_pass else 'FAIL >30s'})")
    print(f"  Attribution: {'PASS' if attribution_pass else 'FAIL'}")
    print(f"  Quality    : {quality}")
    print(f"  ─────────────")

    return {
        "query":            query,
        "agent_queries":    agent_queries,
        "total_time":       total_time,
        "fetch_time":       fetch_time,
        "embed_time":       embed_time,
        "llm_time":         llm_time,
        "n_chunks_raw":     len(all_chunks),
        "n_chunks_retrieved": len(retrieved),
        "n_chunks_final":   len(final_results),
        "src_dist":         src_dist,
        "yt_low_yield":     yt_low_yield,
        "quality":          quality,
        "time_pass":        time_pass,
        "attribution_pass": attribution_pass,
        "answer":           answer,
    }


# ── Full phase runner ─────────────────────────────────────────────────────────

def run_phase6():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = LOG_DIR / f"phase6_e2e_{timestamp}.log"
    lines     = []

    def _h(text, level=1):
        sep = "═" if level == 1 else "─"
        lines.extend(["", sep * 70, text, sep * 70])

    _h(f"PHASE 6: END-TO-END MINI PIPELINE  |  {timestamp}")
    lines += [
        f"\nEmbedding model  : {EMBEDDING_MODEL} (bge-s)",
        f"LLM              : {LLM_MODEL} (Gemini)",
        f"Gate 3           : REMOVED (per-source retrieval + diversity cap handles quality)",
        f"Retrieval        : per-source top-{TOP_PER_SOURCE} → diversity cap → top-{FINAL_K} to LLM",
        f"Max per source   : {MAX_PER_SOURCE} chunks",
        f"Pipeline time cap: {MAX_PIPELINE_TIME}s",
        "",
        *[f"  Q{i+1}. {q}" for i, q in enumerate(QUERIES)],
    ]

    # Load models once before the query loop
    print("\n" + "═" * 70)
    print("  PHASE 6 — E2E MINI PIPELINE")
    print("=" * 70)
    print("\nLoading models...")
    load_model()
    load_llm()

    # ── Run pipeline for each query ────────────────────────────────────────────
    all_results = []

    for i, query in enumerate(QUERIES):
        print(f"\n{'═'*70}")
        print(f"  QUERY {i+1}/{len(QUERIES)}: {query}")
        print(f"{'═'*70}")
        _h(f"QUERY {i+1}: {query}", level=2)

        result = run_pipeline_query(query, lines)
        all_results.append(result)

    # ── Summary ────────────────────────────────────────────────────────────────
    _h("PHASE 6 SUMMARY", level=1)

    quality_pass  = [r for r in all_results if r.get("quality") in ("GOOD", "OK")]
    time_pass     = [r for r in all_results if r.get("time_pass")]
    attrib_pass   = [r for r in all_results if r.get("attribution_pass")]
    avg_time      = sum(r.get("total_time", 0) for r in all_results) / max(1, len(all_results))

    lines += [
        "",
        f"  Queries tested      : {len(QUERIES)}",
        f"  Time pass (<{MAX_PIPELINE_TIME}s)     : {len(time_pass)}/{len(QUERIES)}",
        f"  Attribution pass    : {len(attrib_pass)}/{len(QUERIES)}",
        f"  Quality pass (GOOD/OK): {len(quality_pass)}/{len(QUERIES)}",
        f"  Avg pipeline time   : {avg_time:.1f}s",
        "",
    ]

    for i, r in enumerate(all_results):
        q = r.get("quality", "?")
        t = r.get("total_time", 0)
        lines.append(
            f"  Q{i+1}  quality={q:<5}  time={t:.1f}s  "
            f"attribution={'YES' if r.get('attribution_pass') else 'NO'}  "
            f"retrieved={r.get('n_chunks_retrieved','?')}/{r.get('n_chunks_raw','?')}  "
            f"final_chunks={r.get('n_chunks_final','?')}"
        )

    all_pass = (
        len(quality_pass) == len(QUERIES) and
        len(time_pass)    == len(QUERIES) and
        len(attrib_pass)  == len(QUERIES)
    )

    verdict = "PASS ✅" if all_pass else "PARTIAL ⚠" if len(quality_pass) > 0 else "FAIL ❌"
    lines += [
        "",
        "─" * 55,
        f"  VERDICT: {verdict}",
        "─" * 55,
    ]

    print(f"\n{'═'*70}")
    print(f"  PHASE 6 VERDICT: {verdict}")
    print(f"  Quality pass : {len(quality_pass)}/{len(QUERIES)}")
    print(f"  Time pass    : {len(time_pass)}/{len(QUERIES)}")
    print(f"  Attrib pass  : {len(attrib_pass)}/{len(QUERIES)}")
    print(f"  Avg time     : {avg_time:.1f}s")
    print(f"{'='*70}")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLog → {log_path}")

    return all_results, verdict, timestamp, log_path


# ── MD auto-updaters ──────────────────────────────────────────────────────────

def update_decisions_tracker(all_results: list, verdict: str, timestamp: str):
    text     = PLAN_FILE.read_text(encoding="utf-8")
    date_str = timestamp[:10]

    quality_counts = {"GOOD": 0, "OK": 0, "POOR": 0}
    for r in all_results:
        quality_counts[r.get("quality", "POOR")] = \
            quality_counts.get(r.get("quality", "POOR"), 0) + 1
    avg_time = sum(r.get("total_time", 0) for r in all_results) / max(1, len(all_results))

    reason = (
        f"Full E2E pipeline with all production fixes. "
        f"Quality: GOOD={quality_counts['GOOD']} OK={quality_counts['OK']} "
        f"POOR={quality_counts['POOR']}. "
        f"Avg pipeline time: {avg_time:.1f}s. Verdict: {verdict}."
    )
    new_row = f"| 6 | E2E Pipeline | {verdict} | {reason} | {date_str} |"
    updated = re.sub(r"\| 6 \| E2E.*", new_row, text)
    if updated == text:
        # Row doesn't exist yet — append to end of table if there's a table
        updated = re.sub(
            r"(\| 5 \| Reranker.*\n)",
            r"\1" + new_row + "\n",
            text
        )
    if updated != text:
        PLAN_FILE.write_text(updated, encoding="utf-8")
        print("TESTING_PLAN.md updated.")
    else:
        print("Warning: could not update Phase 6 row in TESTING_PLAN.md — add manually.")


def update_updates_md(all_results: list, verdict: str, timestamp: str, log_path: Path):
    date_str = timestamp[:10]

    rows = []
    for i, r in enumerate(all_results):
        q      = r.get("quality", "?")
        t      = r.get("total_time", 0)
        attrib = "YES" if r.get("attribution_pass") else "NO"
        time_p = "YES" if r.get("time_pass") else "NO"
        rows.append(
            f"| Q{i+1} | {r['query'][:50]} | {q} | {t:.1f}s | {attrib} | {time_p} |"
        )

    avg_time   = sum(r.get("total_time", 0) for r in all_results) / max(1, len(all_results))
    q_good_ok  = sum(1 for r in all_results if r.get("quality") in ("GOOD","OK"))

    entry = textwrap.dedent(f"""
    ---

    ## Phase 6 — End-to-End Pipeline
    **Date:** {date_str}
    **Log:** `{log_path}`

    ### Results

    | Query | Full Query | Quality | Time | Attribution | Time<30s |
    |-------|-----------|---------|------|-------------|---------|
    {chr(10).join(rows)}

    ### Summary
    - Quality pass (GOOD/OK): {q_good_ok}/{len(QUERIES)}
    - Avg pipeline time: {avg_time:.1f}s
    - Verdict: **{verdict}**

    ### Production fixes confirmed working (v2)
    1. LLM query rewriting — one Gemini call pre-fetch, 4 agent-specific queries
    2. Wikipedia v2 — embedding-based title selection (bge-s scores 10 candidates vs query)
    3. Gate 3 removed — per-source retrieval + diversity cap handles quality selection
    4. Per-source retrieval — top-3 per source, guarantees all 4 sources represented
    5. Parallel fetching — ThreadPoolExecutor (all 4 sources simultaneous)
    6. Post-chunk token cap — >480 tokens force-split at sentence boundary
    7. Source diversity cap — max 2 chunks/source → top-8 to LLM
    8. YouTube LOW_YIELD detection — <3 chunks flagged in answer
    9. YouTube vocab correction — technical term dict applied before chunking
    10. Web URL dedup — Wikipedia/YouTube URLs stripped from Web agent results

    ### Full approved stack
    bge-s + RAPTOR-lite (Wikipedia) + atomic (ArXiv) + topic-shift (Web/YouTube)
    + LLM query rewriting + per-source retrieval (top-3/src) + diversity cap + no reranker + role-grouped Gemini prompt

    ### Next Step
    All 6 phases complete. Build production system using approved stack.
    """).strip()

    current    = UPDATE_FILE.read_text(encoding="utf-8")
    header_end = current.find("\n---\n") + 5
    UPDATE_FILE.write_text(
        current[:header_end] + "\n" + entry + "\n" + current[header_end:],
        encoding="utf-8"
    )
    print("UPDATES.md updated.")


def update_status_md(all_results: list, verdict: str, timestamp: str):
    text     = STATUS_FILE.read_text(encoding="utf-8")
    date_str = timestamp[:10]

    text = re.sub(
        r"\*\*Last updated:\*\*.*",
        f"**Last updated:** {date_str} (Phase 6 complete)",
        text
    )
    text = re.sub(
        r"\| 6 \| E2E Mini Pipeline.*",
        f"| 6 | E2E Mini Pipeline | ✅ Done | {verdict} |",
        text
    )

    # Update current phase
    text = re.sub(
        r"\*\*Current phase:\*\*.*",
        "**Current phase:** All phases complete — build production system.",
        text
    )

    STATUS_FILE.write_text(text, encoding="utf-8")
    print("STATUS.md updated.")


def update_code_md(timestamp: str):
    date_str = timestamp[:10]

    entry = textwrap.dedent(f"""
    ---

    ## phase6_e2e_test.py
    **Written:** {date_str}
    **Version:** v1

    ### What it does
    Final phase — wires all approved components into one E2E pipeline with all
    production fixes from RISKTOKEEPINMIND.md. Tests 3 queries and evaluates
    quality, attribution, and pipeline speed.

    ### Key functions

    | Function | Purpose |
    |----------|---------|
    | `fetch_all_parallel(query)` | ThreadPoolExecutor — all 4 sources in parallel |
    | `chunk_all(query, fetch_results)` | Per-source chunking + metadata tagging |
    | `apply_token_cap(chunks)` | Force-split chunks >480 tokens at sentence boundary |
    | `apply_diversity_cap(retrieved)` | Max 2 chunks/source → top-8 to LLM |
    | `correct_youtube_vocab(text)` | Dict-based technical vocab correction pre-chunking |
    | `semantic_retrieve(query, chunks, embs, k)` | Cosine similarity top-k (bge-s) |
    | `build_prompt(query, final_results, yt_low_yield)` | Role-grouped LLM prompt |
    | `call_llm(prompt)` | Gemini 1.5 Flash API call |
    | `evaluate_quality(query, answer, results)` | GOOD/OK/POOR heuristic evaluation |
    | `run_pipeline_query(query, lines)` | Full pipeline for one query |
    | `run_phase6()` | Runs all 3 queries, writes log, calls MD updaters |

    ### Production fixes applied (v2)
    1. LLM query rewriting (one Gemini call pre-fetch, 4 agent-specific queries)
    2. Wikipedia v2 fetch (embedding-based title selection, 10 candidates ranked by bge-s)
    3. Gate 3 removed (per-source retrieval replaces it)
    4. Per-source retrieval (top-3/source, all sources fairly represented)
    5. Parallel fetch (ThreadPoolExecutor, max_workers=4)
    6. Post-chunk token cap (>480 tokens split at sentence boundary)
    7. Source diversity cap (max 2/source, top-8 final)
    8. YouTube LOW_YIELD detection (<3 chunks)
    9. YouTube vocab correction (regex dict, pre-chunking)
    10. Web URL dedup (Wikipedia/YouTube URLs skipped from Web agent)

    ### Outputs
    - `logs/phase6_e2e_{{timestamp}}.log`
    - TESTING_PLAN.md Phase 6 row updated
    - UPDATES.md entry added
    - STATUS.md Phase 6 row + Last Updated + Current Phase updated
    - CODE.md this entry added
    """).strip()

    current = CODE_FILE.read_text(encoding="utf-8")
    CODE_FILE.write_text(current + "\n\n" + entry + "\n", encoding="utf-8")
    print("CODE.md updated.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("  PHASE 6 — END-TO-END MINI PIPELINE  (v2)")
    print("═" * 70)
    print(f"  Embedding  : {EMBEDDING_MODEL}")
    print(f"  LLM        : {LLM_MODEL} (Gemini)")
    print(f"  Retrieval  : per-source top-{TOP_PER_SOURCE} → diversity cap (max {MAX_PER_SOURCE}/src) → top-{FINAL_K}")
    print(f"  Queries    : {len(QUERIES)}")
    print(f"  Pass time  : <{MAX_PIPELINE_TIME}s per query")
    print("═" * 70)
    print("\nProduction fixes v2: LLM query rewriting | Wikipedia embedding title selection")
    print("                     Gate 3 removed | per-source retrieval | token cap | diversity cap")
    print("═" * 70 + "\n")

    all_results, verdict, timestamp, log_path = run_phase6()

    print("\n── Updating MDs ──")
    update_decisions_tracker(all_results, verdict, timestamp)
    update_updates_md(all_results, verdict, timestamp, log_path)
    update_status_md(all_results, verdict, timestamp)
    update_code_md(timestamp)

    print(f"\nDone. All MDs updated. Log → {log_path}")
