# Nexus AI — Research Synthesis Engine

A multi-agent RAG pipeline that takes a research question, fetches content from four sources in parallel, and synthesizes a sourced, attributed answer using Gemini 2.5 Flash.

## How it works

```
Query → Agent rewrite → Parallel fetch (4 sources)
      → Chunk + Embed → Semantic retrieval → LLM synthesis → Streamed answer
```

Each source has a dedicated role:

| Agent | Source | Role |
|-------|--------|------|
| Foundation | Wikipedia | Background definitions |
| Research | ArXiv + Semantic Scholar | Peer-reviewed papers |
| Current | Tavily Web Search | Recent news and articles |
| Explainer | YouTube Transcripts | Tutorials and talks |

## Stack

- **Backend** — FastAPI, SSE streaming
- **LLM** — Gemini 2.5 Flash (query rewriting + synthesis)
- **Embeddings** — `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX, thread-safe)
- **Retrieval** — Semantic search with per-source diversity cap
- **Chunking** — Hierarchical (Wikipedia), atomic (ArXiv), semantic topic-shift (Web/YouTube)
- **Database** — Supabase PostgreSQL (query logs + user history)
- **Deploy** — Render

## Project structure

```
app/
├── main.py        # FastAPI app, SSE pipeline, API endpoints
├── pipeline.py    # Core RAG pipeline: retrieve, rank, prompt, LLM
├── sources.py     # Data fetching: Wikipedia, ArXiv, Web, YouTube
├── chunking.py    # Chunking strategies per source type
└── static/        # Frontend: HTML, CSS, JS
requirements.txt
render.yaml
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=...
TAVILY_API_KEY=...
YOUTUBE_DATA_API_KEY=...
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
```

Run locally:
```bash
cd app
uvicorn main:app --reload --port 8081
```

## Deploy

Configured for Render via `render.yaml`. Set the five environment variables in the Render dashboard and connect the GitHub repo — it deploys automatically on push.
