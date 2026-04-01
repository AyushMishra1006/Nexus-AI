# Nexus AI — Multi-Agent Research Synthesis Engine

<img width="1916" height="971" alt="image" src="https://github.com/user-attachments/assets/2db1ff4c-6b3f-410e-a03d-9b3e7b6dce93" />

<img width="1919" height="966" alt="image" src="https://github.com/user-attachments/assets/3d49166b-5a8f-48c3-a04d-e2ceabfa9057" />

A production-grade RAG pipeline that answers research questions by coordinating four specialised AI agents in parallel — each targeting a different knowledge source — then synthesising a single sourced, attributed answer via Gemini 2.5 Flash.

Built through a structured **6-phase testing framework**. Every architectural decision is data-backed and recorded before being locked in.

> **Validated across 34 queries — 91% GOOD quality, ~33 seconds end-to-end on typical queries (includes ~17s Gemini free-tier synthesis).**

---

## What it does

You ask a research question. The system:

1. **Rewrites your query** — one Gemini call generates four agent-specific search queries, not the same query sent four times
2. **Fetches in parallel** — four agents fire simultaneously across Wikipedia, ArXiv, Tavily Web, and YouTube
3. **Chunks intelligently** — each source gets a strategy matched to its structure, not a one-size-fits-all token split
4. **Embeds and retrieves** — semantic search with per-source diversity caps so no single source dominates
5. **Synthesises** — Gemini receives role-grouped context and streams an attributed answer to the frontend

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│   LLM Query Rewrite (Gemini)        │  ← one call, four agent-specific queries
│   + web_mode classification         │    + Wikipedia keyword fallback
└──────────────────┬──────────────────┘
                   │
       ┌───────────┼───────────┬──────────────┐
       ▼           ▼           ▼              ▼
  Wikipedia     ArXiv       Tavily/DDG    YouTube
  Foundation    Research    Current       Explainer
  agent         agent       agent         agent
  (bge-s        (sort by    (topic=news   ("explained
  title         date,       or general,   tutorial"
  selection)    10s cap)    days=30)      + captions)
       │           │           │              │
       ▼           ▼           ▼              ▼
  Hierarchical  Atomic     Semantic       Semantic
  parent-child  (1 paper   topic-shift    topic-shift
  chunking      = 1 chunk) + overlap      + vocab fix
       │           │           │              │
       └───────────┴───────────┴──────────────┘
                   │
                   ▼
       FastEmbed ONNX  (bge-small-en-v1.5)
       Single shared instance — genuinely thread-safe
                   │
                   ▼
       Per-source semantic retrieval
       top-4 per source → diversity cap (max 3/src) → top-12 to LLM
                   │
                   ▼
       ┌──────────────────────────────┐
       │   Gemini 2.5 Flash           │  ← context grouped by role, not flat
       │   Role-grouped prompt        │    [BACKGROUND] [RESEARCH] [CURRENT] [EXPLAINER]
       └──────────────────────────────┘
                   │
                   ▼
       SSE streamed answer → frontend typewriter render
```

---

## Agent Design — 3-Mechanism Specialisation

Each agent stays in its lane through three things working together:

| Agent | Query transformation | Source config | Role tag |
|-------|---------------------|---------------|----------|
| Wikipedia | unchanged | 1 article, section-scored vs query | `foundation` |
| ArXiv | unchanged | sort by `SubmittedDate` DESC, 10s hard cap | `research` |
| Web | `query + year` | Tavily `topic=news`, `days=30` | `current` |
| YouTube | `"explained tutorial " + query` | captions, top 2 videos | `explainer` |

Every chunk carries `source`, `role`, `source_date`, `fetched_at`, `url`, `query` metadata. The LLM prompt groups context by role — it knows what kind of source it is reading before it synthesises.

---

## Engineering Decisions (Data-Backed)

These were not guesses. Each was tested, measured, and locked in before moving to the next phase.

**Chunk size — 750 chars for Wikipedia children**
Phase 2 tested 450-char children first. Token distribution showed 115-token average — below the 150-token retrieval floor. Raised to 750 chars → 167-token average. Parent paragraphs (~1,200–1,500 chars) are what the LLM reads; children are search-only.

**Embedding model — `bge-small-en-v1.5` over MiniLM and bge-base**
Phase 3 benchmarked four models. MiniLM rejected: 256-token limit truncated 18.7% of chunks (39/209). bge-base rejected: 3× slower (6–7s vs 2.2s avg) with marginal quality gain. bge-s: 512-token limit, 2.2s avg embed, 130MB, built for retrieval. Unanimous choice.

**RAPTOR dropped for Wikipedia**
Phase 2 built and tested RAPTOR (recursive abstractive clustering). Dropped: requires LLM calls during indexing, adds cost and latency at index time. Hierarchical parent-child achieves equivalent precision/context tradeoff without the overhead.

**ArXiv 10s hard cap — data-driven**
31 queries of telemetry showed ArXiv averaging 12.1s arrived time, with 42% of queries exceeding the 25s global wall due to Semantic Scholar 429s on cloud IPs. `fetch_arxiv_enhanced` now runs inside a worker thread with `future.result(timeout=10)`. Fast responses still contribute peer-reviewed context. Slow responses are skipped cleanly. Pipeline never waits more than 10s for ArXiv.

**Per-source retrieval before diversity cap**
Early pipeline sent all chunks into one ranked list — ArXiv abstracts (short, high similarity) always won, YouTube never reached the LLM. Fixed with per-source top-4 first, then diversity cap (max 3/source), then global top-12. Every agent gets a proportional slot based on quality, not chunk size.

---

## Evaluation Results

Validated across 34 real queries (local pipeline, post all fixes):

| Metric | Result |
|--------|--------|
| Quality: GOOD | **91%** (31/34 queries) |
| Quality: POOR | 9% (3 niche/ambiguous queries) |
| Avg sim score across chunks sent to LLM | **0.78** |
| Typical end-to-end time | **~33 seconds** |
| Timing breakdown | Rewrite 3.6s · Fetch+Embed ~14s · LLM 17.3s |
| Wikipedia fetch success | 97% |
| Web fetch success | **100%** |
| YouTube fetch success | 97% |
| ArXiv fetch success | 42%* |
| Noise chunks reaching LLM | **Zero** across all 34 queries (sim floor = 0.65) |

*ArXiv rate-limits cloud datacenter IPs. Hard 10s cap ensures fast responses contribute without dragging pipeline speed.




---

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | FastAPI + SSE | Streaming responses, async I/O |
| LLM | Gemini 2.5 Flash | Fast reasoning, 1M context window |
| Embeddings | FastEmbed ONNX (`bge-small-en-v1.5`) | Thread-safe, no GIL issues, 2.2s avg |
| Web search | Tavily API + DDG fallback | Tavily for news freshness; DDG for resilience |
| ArXiv | Semantic Scholar REST + abstract fallback | No scraping, stable API, TLDR field |
| YouTube | `youtube-transcript-api` | Caption extraction, no API quota cost |
| Database | Supabase PostgreSQL | Query logs, user history, zero infra |
| Rate limiting | slowapi (5 req/min per IP) | API key protection |

---

## Project Structure

```
app/
├── main.py        # FastAPI app — SSE pipeline, rate limiting, Supabase logging
├── pipeline.py    # Core RAG logic — fetch, chunk, embed, retrieve, prompt, LLM
├── sources.py     # Four data fetchers with role configs and metadata tagging
├── chunking.py    # Per-source chunking: hierarchical, atomic, semantic topic-shift
└── static/        # Frontend — dark theme, typewriter streaming, expandable sources
requirements.txt
render.yaml
.python-version    # Pinned to 3.11.9
```

---

## Local Setup

```bash
pip install -r requirements.txt
```

Create a `.env` in the project root:
```
GEMINI_API_KEY=...
TAVILY_API_KEY=...
YOUTUBE_DATA_API_KEY=...
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
```

Run:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`

---

## Build Methodology

The pipeline was not assembled all at once. It went through a structured 6-phase testing framework — each component isolated, tested, and approved before the next was built:

| Phase | Component tested | Decision |
|-------|-----------------|----------|
| 1 | All 4 data sources | All kept — each serves a distinct knowledge role |
| 1b | Role configs + metadata tagging | 3-mechanism agent specialisation locked in |
| 2 | Chunking strategies per source | Hierarchical (Wiki), atomic (ArXiv), topic-shift (Web/YT) |
| 3 | Embedding models | `bge-small-en-v1.5` selected over 3 alternatives with data |
| 4 | Retrieval methods | Per-source semantic retrieval with diversity cap |
| 5 | Cross-encoder reranker | Tested and skipped — diversity cap sufficient at this scale |
| 6 | End-to-end pipeline | 91% GOOD quality, ~33s typical, all 4 sources validated |

Full decision log, test results, and architectural rationale tracked in internal session docs (not in this repo).
