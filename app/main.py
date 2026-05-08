"""
Research Synthesis Engine — Localhost App (v2)

Fixes applied:
  Fix 2 — Pre-load bge-s + Gemini at startup (no cold-start penalty)
  Fix 3 — Per-source streaming pipeline: chunk+embed each source as its fetch completes
           (overlaps embed with fetch instead of waiting for all 4 sources)
  Fix 4 — Stream Gemini tokens to frontend (first words appear ~2s after LLM starts)

Threading: FastEmbed (ONNX Runtime) is thread-safe — no lock required.
Pipeline threads run in a bounded ThreadPoolExecutor (max_workers=3).
"""

import asyncio
import json
import numpy as np
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime, date, timezone
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv
load_dotenv(APP_DIR.parent / ".env", override=True)

from supabase import create_client, Client as SupabaseClient

# ── Env var validation — fail fast at startup if critical keys are missing ──────
_REQUIRED_KEYS = ["GEMINI_API_KEY", "TAVILY_API_KEY"]
_missing = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
if _missing:
    raise RuntimeError(
        f"[startup] Missing required environment variables: {', '.join(_missing)}. "
        "Set them in your .env file (local) or Render dashboard (production)."
    )

# ── Supabase client (optional — degrades gracefully if keys not set) ────────────
_SUPABASE_URL  = os.getenv("SUPABASE_URL", "")
_SUPABASE_KEY  = os.getenv("SUPABASE_ANON_KEY", "")
_supabase: SupabaseClient | None = None
if _SUPABASE_URL and _SUPABASE_KEY:
    try:
        _supabase = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        print("  [startup] Supabase client initialised.")
    except Exception as _e:
        print(f"  [startup] WARNING: Supabase init failed: {_e}. DB logging disabled.")
else:
    print("  [startup] SUPABASE_URL/SUPABASE_ANON_KEY not set — DB logging disabled.")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Pipeline imports ────────────────────────────────────────────────────────────
from pipeline import (
    load_model,
    load_llm,
    embed_texts,
    rewrite_queries_for_agents,
    select_wikipedia_title,
    fetch_wikipedia_article,
    fetch_arxiv_enhanced,
    _fetch_arxiv_timed,
    chunk_source,
    apply_token_cap,
    per_source_retrieve,
    apply_diversity_cap,
    apply_source_minimum,
    build_prompt,
    evaluate_quality,
    SOURCE_ROLES,
    TOP_PER_SOURCE,
    MAX_PER_SOURCE,
    FINAL_K,
    MIN_CHUNK_SIM,
    WIKI_SIM_THRESHOLD,
)
from sources import fetch_web, fetch_youtube

# fastembed (ONNX Runtime) is thread-safe — no lock needed.
# Sources embed concurrently as each fetch completes.

# Bounded executor: max 3 pipeline threads at once.
# 4th request queues rather than spawning unbounded threads → predictable RAM.
_pipeline_executor = ThreadPoolExecutor(max_workers=3)

# ── App setup ───────────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
LOGS_DIR   = Path(__file__).parent / "query_logs"
LOGS_DIR.mkdir(exist_ok=True)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Research Synthesis Engine", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# TODO: after Render deploy, replace "*" with your exact Render domain e.g.
# allow_origins=["https://nexus-ai.onrender.com"]
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_SRC_SHORT   = {"wikipedia": "wiki", "arxiv": "arxiv", "web": "web", "youtube": "yt"}
_STATUS_NORM = {
    "SUCCESS": "ok",    "PARTIAL": "ok",
    "FAILED":  "error", "ERROR":   "error", "UNKNOWN": "error",
    "TIMEOUT": "timeout",
    "SKIPPED": "skipped",
    "EMPTY":   "empty",
}


# ── Fix 2: Pre-load models at startup (with timeouts) ──────────────────────────
@app.on_event("startup")
async def startup():
    loop = asyncio.get_running_loop()

    print("\n  [startup] Pre-loading bge-s embedding model...")
    try:
        await asyncio.wait_for(loop.run_in_executor(None, load_model), timeout=180)
        print("  [startup] bge-s loaded.")
    except asyncio.TimeoutError:
        raise RuntimeError(
            "[startup] FATAL: bge-s model load timed out after 180s — aborting."
        )
    except Exception as e:
        raise RuntimeError(f"[startup] FATAL: bge-s model load failed: {e}")

    print("  [startup] Pre-loading Gemini LLM client...")
    try:
        await asyncio.wait_for(loop.run_in_executor(None, load_llm), timeout=30)
        print("  [startup] Gemini client loaded.")
    except asyncio.TimeoutError:
        raise RuntimeError(
            "[startup] FATAL: Gemini client init timed out after 30s — aborting."
        )
    except Exception as e:
        raise RuntimeError(f"[startup] FATAL: Gemini client init failed: {e}")

    print("  [startup] Models ready. Accepting requests.\n")


# ── Routes ──────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    user_id: str = ""   # localStorage UUID from frontend; empty string = anonymous


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/query")
@limiter.limit("5/minute")
async def run_query(request: Request, body: QueryRequest):
    query   = body.query.strip()
    user_id = body.user_id.strip() or "anonymous"
    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)
    if len(query) > 500:
        return JSONResponse({"error": "Query too long (max 500 characters)"}, status_code=413)

    query_id = str(uuid.uuid4())[:8]
    loop     = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def emit(event_type: str, **kwargs):
        payload = {"type": event_type, "query_id": query_id, **kwargs}
        asyncio.run_coroutine_threadsafe(q.put(payload), loop)

    def run_pipeline():
        # Safe defaults — must be initialised before the try block so the except
        # crash-insert never hits NameError on a partial pipeline failure.
        log_lines        = []
        llm_retry_count  = 0
        avg_sim          = None
        min_sim          = None
        rewrite_success  = True
        _pipeline_error  = None
        sources_json     = {}
        result_data      = {
            "query_id":  query_id,
            "query":     query,
            "timestamp": datetime.now().isoformat(),
            "steps":     {},
        }

        try:
            pipeline_start = time.perf_counter()
            emit("start", query=query, timestamp=result_data["timestamp"])

            # ── Step 0: Query Rewrite ──────────────────────────────────────
            emit("step_start", step="rewrite",
                 message="Rewriting queries for each agent with Gemini...")
            t0 = time.perf_counter()
            agent_queries   = rewrite_queries_for_agents(query)
            rewrite_time    = round(time.perf_counter() - t0, 2)
            rewrite_success = not agent_queries.get("_fallback", False)
            emit("step_done", step="rewrite", elapsed=rewrite_time,
                 queries=agent_queries)
            log_lines.append(f"STEP 0 — Rewrite ({rewrite_time}s)")
            for src, qt in agent_queries.items():
                log_lines.append(f"  {src}: {qt}")
            result_data["steps"]["rewrite"] = {
                "elapsed": rewrite_time, "queries": agent_queries
            }

            # ── Step 1: Wikipedia title selection ──────────────────────────────
            emit("step_start", step="fetch",
                 message="Selecting Wikipedia title then launching all 4 fetches...")
            t_fetch_start = time.perf_counter()

            wiki_ranked = select_wikipedia_title(agent_queries["wikipedia"])

            # BUG 7 fix: skip Wikipedia if best match sim < WIKI_SIM_THRESHOLD (imported from pipeline)
            wiki_feed = wiki_ranked  # what gets passed to fetch_wikipedia_article
            if wiki_ranked:
                best_title, best_sim = wiki_ranked[0]
                if best_sim < WIKI_SIM_THRESHOLD:
                    wiki_feed = []   # fetch_wikipedia_article returns SKIPPED on empty list
                    emit("source_arrived", source="wikipedia",
                         status="SKIPPED",
                         title=best_title,
                         title_sim=round(float(best_sim), 3),
                         chars=0, fetch_elapsed=0,
                         error=f"sim={best_sim:.3f} below threshold {WIKI_SIM_THRESHOLD}")
                else:
                    emit("source_arrived", source="wikipedia",
                         status="TITLE_SELECTED",
                         title=best_title,
                         title_sim=round(float(best_sim), 3),
                         chars=0, fetch_elapsed=0)

            # ── Fix 3: Per-source streaming pipeline ───────────────────────
            # Launch all 4 fetches in parallel.
            # As each source completes: chunk + embed it immediately (under lock).
            # All chunk+embed work overlaps with remaining fetches.

            FETCH_TIMEOUT = 25

            _web_mode = agent_queries.get("web_mode", "general")

            fetchers = {
                "wikipedia": (fetch_wikipedia_article,                        wiki_feed),
                "arxiv":     (_fetch_arxiv_timed,                             agent_queries["arxiv"]),
                "web":       (lambda q: fetch_web(q, web_mode=_web_mode),     agent_queries["web"]),
                "youtube":   (fetch_youtube,                                  agent_queries["youtube"]),
            }

            # Accumulators
            all_chunks:    list       = []
            all_embs_list: list       = []   # list of np.ndarray, one per source
            source_summary:  dict     = {}
            per_src_chunks:  dict     = {}

            executor = ThreadPoolExecutor(max_workers=4)
            try:
                future_to_src = {
                    executor.submit(fn, arg): src
                    for src, (fn, arg) in fetchers.items()
                }

                completed_futures = set()

                try:
                    for future in as_completed(future_to_src,
                                               timeout=FETCH_TIMEOUT + 5):
                        src = future_to_src[future]
                        completed_futures.add(future)
                        elapsed_so_far = round(
                            time.perf_counter() - t_fetch_start, 2
                        )

                        try:
                            res = future.result()
                        except Exception as e:
                            res = {"status": "ERROR", "text": "",
                                   "error": str(e)}

                        text = res.get("text", "")
                        source_summary[src] = {
                            "status":    res.get("status", "UNKNOWN"),
                            "chars":     len(text),
                            "url":       res.get("url", ""),
                            "title":     res.get("fetched_title", ""),
                            "title_sim": round(
                                float(res.get("title_sim", 0) or 0), 3),
                            "error":     res.get("error", ""),
                            "arrived_at": elapsed_so_far,
                        }

                        emit("source_arrived", source=src,
                             status=res.get("status", "UNKNOWN"),
                             chars=len(text),
                             fetch_elapsed=elapsed_so_far,
                             title=res.get("fetched_title", ""),
                             title_sim=round(
                                 float(res.get("title_sim", 0) or 0), 3),
                             error=res.get("error", ""))

                        log_lines.append(
                            f"  [{src}] {res.get('status','?')} "
                            f"{len(text)//1000}K chars @{elapsed_so_far}s"
                        )

                        if not text.strip():
                            per_src_chunks[src] = 0
                            continue

                        # Chunk + embed immediately — fastembed is thread-safe,
                        # no lock needed. Sources run concurrently.
                        t_ce = time.perf_counter()
                        try:
                            chunks = chunk_source(src, text, query)
                        except Exception as e:
                            log_lines.append(
                                f"    chunk error [{src}]: {e}")
                            chunks = []

                        # Attach metadata
                        # ArXiv: one chunk per paper — assign each chunk its own URL from metas[i]
                        arxiv_metas = res.get("metas", []) if src == "arxiv" else []
                        valid = []
                        for i, c in enumerate(chunks):
                            if not c.get("text", "").strip():
                                continue
                            c["source"]     = src
                            c["role"]       = SOURCE_ROLES[src]
                            c["query"]      = query
                            c["fetched_at"] = date.today().isoformat()
                            if src == "arxiv" and i < len(arxiv_metas):
                                c["url"] = arxiv_metas[i].get("url", "")
                            else:
                                c["url"] = res.get("url") or ""
                            valid.append(c)
                        chunks = valid

                        # Token cap (no embed, just text splitting)
                        chunks = apply_token_cap(chunks)

                        # Embed this source's chunks
                        if chunks:
                            texts_to_embed = [c["text"] for c in chunks]
                            embs = embed_texts(texts_to_embed)
                        else:
                            embs = None

                        ce_elapsed = round(time.perf_counter() - t_ce, 2)
                        per_src_chunks[src] = len(chunks)

                        if chunks and embs is not None:
                            all_chunks.extend(chunks)
                            all_embs_list.append(embs)

                        emit("source_processed", source=src,
                             chunks=len(chunks), ce_elapsed=ce_elapsed)
                        log_lines.append(
                            f"    → {len(chunks)} chunks, embed {ce_elapsed}s"
                        )

                except FuturesTimeoutError:
                    # Cancel any fetches still running after timeout
                    for future, src in future_to_src.items():
                        if future not in completed_futures:
                            future.cancel()
                            source_summary[src] = {
                                "status": "TIMEOUT", "chars": 0,
                                "url": "", "error": "fetch timeout"
                            }
                            per_src_chunks[src] = 0
                            emit("source_arrived", source=src,
                                 status="TIMEOUT", chars=0,
                                 fetch_elapsed=FETCH_TIMEOUT, error="timeout")

            finally:
                # shutdown(wait=False): don't block on stuck threads.
                # Threads that are hung on network calls continue in the background
                # but we don't wait for them — pipeline proceeds immediately.
                executor.shutdown(wait=False)

            fetch_time = round(time.perf_counter() - t_fetch_start, 2)
            emit("step_done", step="fetch", elapsed=fetch_time,
                 sources=source_summary)
            log_lines.append(f"STEP 1 — Fetch wall time: {fetch_time}s")
            result_data["steps"]["fetch"] = {
                "elapsed": fetch_time, "sources": source_summary
            }

            if not all_chunks:
                emit("error", message="No chunks produced from any source.")
                return

            # Combine all per-source embeddings into one matrix
            all_embs = np.vstack(all_embs_list)
            total_chunks = len(all_chunks)

            # Emit combined chunk/embed summary
            emit("step_done", step="chunk", elapsed=0,
                 total=total_chunks, per_source=per_src_chunks)
            emit("step_done", step="embed", elapsed=0,
                 chunks=total_chunks, dims=int(all_embs.shape[1]))

            result_data["steps"]["chunk"] = {
                "elapsed": 0, "total": total_chunks,
                "per_source": per_src_chunks
            }
            result_data["steps"]["embed"] = {"elapsed": 0}

            yt_chunks = [c for c in all_chunks
                         if c.get("source") == "youtube"]
            yt_low    = len(yt_chunks) < 3

            # ── Step 5: Retrieve ───────────────────────────────────────────
            emit("step_start", step="retrieve",
                 message="Per-source retrieval → diversity cap → source minimum...")
            retrieval_query = agent_queries.get("retrieval_query", query)
            retrieved = per_source_retrieve(
                retrieval_query, all_chunks, all_embs, top_per_source=TOP_PER_SOURCE
            )
            final = apply_diversity_cap(retrieved)
            final = apply_source_minimum(retrieved, final)

            # BUG 10 fix: dedup by parent — key on url + parent_text[:100]
            seen_parents = {}
            for r in final:
                key = (r["chunk"].get("url", "")
                       + r["chunk"].get("parent_text", r["chunk"]["text"])[:100])
                if key not in seen_parents or r["sim"] > seen_parents[key]["sim"]:
                    seen_parents[key] = r
            final = sorted(seen_parents.values(), key=lambda x: x["sim"], reverse=True)

            # Fix 1: hard sim floor — drop chunks below MIN_CHUNK_SIM (0.65).
            # Graceful: if all chunks fall below floor (niche query), keep top-1.
            floored = [r for r in final if r["sim"] >= MIN_CHUNK_SIM]
            if floored:
                final = floored

            sim_scores = [float(r["sim"]) for r in final]
            avg_sim    = round(sum(sim_scores) / len(sim_scores), 4) if sim_scores else None
            min_sim    = round(min(sim_scores), 4) if sim_scores else None

            retrieve_summary = {
                src: len([r for r in final
                          if r["chunk"].get("source") == src])
                for src in ["wikipedia", "arxiv", "web", "youtube"]
            }

            # Build JSONB sources — one key per agent, status normalized to lowercase.
            # Built here so all three inputs (source_summary, per_src_chunks,
            # retrieve_summary) are fully populated.
            sources_json = {}
            for src, key in _SRC_SHORT.items():
                summ   = source_summary.get(src, {})
                status = _STATUS_NORM.get(
                    summ.get("status", "UNKNOWN").upper(), "error"
                )
                arrived = (0    if status == "skipped" else
                           None if status in ("timeout", "error") else
                           summ.get("arrived_at"))
                sources_json[key] = {
                    "status":       status,
                    "arrived_at_s": arrived,
                    "chars":        summ.get("chars", 0),
                    "chunks_prod":  per_src_chunks.get(src, 0),
                    "chunks_llm":   retrieve_summary.get(src, 0),
                    "error":        (summ.get("error") or "")[:100] or None,
                }

            emit("step_done", step="retrieve",
                 per_source=retrieve_summary, total=len(final))
            log_lines.append(
                f"STEP 5 — Retrieve: {retrieve_summary} → {len(final)} final"
            )
            result_data["steps"]["retrieve"] = {
                "per_source": retrieve_summary, "total": len(final)
            }

            chunks_for_ui = [
                {
                    "source":  r["chunk"].get("source"),
                    "role":    r["chunk"].get("role"),
                    "url":     r["chunk"].get("url", ""),
                    "preview": r["chunk"].get(
                        "parent_text", r["chunk"]["text"])[:300],
                    "sim":     round(float(r.get("sim", 0)), 3),
                }
                for r in final
            ]

            # ── Step 6: Build prompt + stream LLM ─────────────────────────
            prompt = build_prompt(query, final, yt_low_yield=yt_low)
            emit("step_start", step="llm",
                 message="Gemini 2.5 Flash generating answer (streaming)...")

            t_llm           = time.perf_counter()
            llm_model       = load_llm()
            full_answer     = ""
            delays          = [5, 15, 45]
            llm_retry_count = 0

            for attempt in range(len(delays) + 1):
                try:
                    # Fix 4: stream=True — tokens arrive as Gemini generates
                    response = llm_model.generate_content(prompt, stream=True)
                    for chunk in response:
                        token = getattr(chunk, "text", None)
                        if token:
                            full_answer += token
                            emit("llm_token", token=token)
                    break  # success
                except Exception as e:
                    retryable = "429" in str(e) or "503" in str(e) or "UNAVAILABLE" in str(e)
                    if retryable and attempt < len(delays):
                        wait_s = delays[attempt]
                        emit("llm_retry", attempt=attempt + 1, wait=wait_s,
                             message=f"Gemini busy — retrying in {wait_s}s")
                        time.sleep(wait_s)
                        llm_retry_count += 1
                        full_answer = ""
                    else:
                        raise

            llm_time = round(time.perf_counter() - t_llm, 2)
            emit("step_done", step="llm", elapsed=llm_time)
            log_lines.append(f"STEP 6 — LLM stream ({llm_time}s)")
            result_data["steps"]["llm"] = {"elapsed": llm_time}

            # ── Evaluate & save ────────────────────────────────────────────
            total_time = round(time.perf_counter() - pipeline_start, 2)
            quality    = evaluate_quality(query, full_answer, final)
            has_attr   = any(
                m in full_answer
                for m in ["[Background]", "[Research]", "[Web]", "[Explainer]",
                          "Wikipedia", "ArXiv", "according to", "Source"]
            )

            sources_used = []
            seen_urls    = set()
            for r in final:
                c   = r["chunk"]
                url = c.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources_used.append({
                        "source": c.get("source"),
                        "role":   c.get("role", ""),
                        "url":    url,
                        "title":  c.get("fetched_title", "") or url,
                    })

            result_data.update({
                "answer":             full_answer,
                "quality":            quality,
                "attribution":        has_attr,
                "total_time":         total_time,
                "sources_used":       sources_used,
                "chunks_sent_to_llm": chunks_for_ui,
                "log_lines":          log_lines,
                "timing": {
                    "rewrite":              rewrite_time,
                    "fetch_plus_embed":     fetch_time,
                    "llm":                  llm_time,
                    "total":                total_time,
                },
            })

            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = LOGS_DIR / f"{query_id}_{ts}.json"
            log_path.write_text(
                json.dumps(result_data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            if _supabase:
                try:
                    _supabase.table("queries").insert({
                        "query_id":          query_id,
                        "user_id":           user_id,
                        "query_text":        query,
                        "timestamp":         datetime.now(timezone.utc).isoformat(),
                        "quality":           quality,
                        "total_time_s":      total_time,
                        "rewrite_time_s":    rewrite_time,
                        "fetch_wall_time_s": fetch_time,
                        "llm_time_s":        llm_time,
                        "llm_retry_count":   llm_retry_count,
                        "rewrite_success":   rewrite_success,
                        "web_mode":          _web_mode,
                        "chunks_to_llm":     len(final),
                        "avg_sim_score":     avg_sim,
                        "min_sim_score":     min_sim,
                        "sources":           sources_json or None,
                        "error_log":         None,
                        "answer_text":       full_answer,
                    }).execute()
                except Exception as e:
                    print(f"  [db] WARNING: queries insert failed: {e}")

            emit("answer", sources=sources_used, chunks=chunks_for_ui)
            emit("done",
                 quality=quality,
                 attribution=has_attr,
                 total_time=total_time,
                 time_pass=total_time < 30,
                 log_file=log_path.name,
                 timing=result_data["timing"])

        except Exception as exc:
            _pipeline_error = traceback.format_exc()
            print(f"[pipeline error] query_id={query_id}: {_pipeline_error}")
            emit("error", message=str(exc))
            # Minimal crash row — monitoring agent uses error_log IS NOT NULL to count errors.
            if _supabase:
                try:
                    _supabase.table("queries").insert({
                        "query_id":   query_id,
                        "user_id":    user_id,
                        "query_text": query,
                        "timestamp":  datetime.now(timezone.utc).isoformat(),
                        "error_log":  _pipeline_error,
                    }).execute()
                except Exception as db_e:
                    print(f"  [db] WARNING: crash row insert failed: {db_e}")

        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    _pipeline_executor.submit(run_pipeline)

    async def generate():
        while True:
            event = await q.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield 'data: {"type":"stream_end"}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.get("/api/logs")
async def list_logs():
    logs = []
    for f in sorted(LOGS_DIR.glob("*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            logs.append({
                "query_id":   data.get("query_id"),
                "query":      data.get("query"),
                "timestamp":  data.get("timestamp"),
                "quality":    data.get("quality"),
                "total_time": data.get("total_time"),
                "filename":   f.name,
            })
        except Exception:
            continue
    return JSONResponse(logs)


@app.get("/api/logs/{filename}")
async def get_log(filename: str):
    # Path traversal guard — only allow plain filenames, no slashes or dots traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    log_file = LOGS_DIR / filename
    if not log_file.exists() or not log_file.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(json.loads(log_file.read_text(encoding="utf-8")))


@app.delete("/api/logs/{filename}")
async def delete_log(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    log_file = LOGS_DIR / filename
    if log_file.exists() and log_file.is_file():
        log_file.unlink()
    return JSONResponse({"deleted": filename})


@app.get("/api/history/{user_id}")
async def get_user_history(user_id: str, limit: int = 20):
    """Return the last N research queries for a given user (localStorage UUID)."""
    if not _supabase:
        return JSONResponse({"error": "Database not configured"}, status_code=503)
    try:
        resp = (
            _supabase.table("queries")
            .select("query_id, query_text, answer_text, total_time_s, quality, timestamp")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        rows = resp.data or []
        for row in rows:
            row["total_time"] = row.pop("total_time_s", None)
        return JSONResponse(rows)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  Research Synthesis Engine — Localhost v2")
    print("="*60)
    port = int(os.getenv("PORT", 8081))
    print(f"  URL  : http://localhost:{port}")
    print("  Logs : app/query_logs/")
    print("="*60 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
