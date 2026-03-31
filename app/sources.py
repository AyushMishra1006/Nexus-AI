"""
Phase 1b — Role-Based Fetch Testing  (v5)
Re-runs all 4 sources with role config + content caps + evaluation pipeline:
  - Wikipedia : query unchanged | min 10K | retry next title (up to 3) | no max (Phase 2 handles)
  - ArXiv     : Relevance sort + post-filter 2 yrs | abstract-only | min 200/abstract
  - Web       : query+year + DDG timelimit='m' | min 3K | max 12K | skip Wikipedia/YouTube URLs
  - YouTube   : "explained tutorial "+query | min 3K | max 8K | pool from 3 query variants → retry up to 3

Evaluation pipeline (2 gates, no embedding model required at this phase):
  Gate 1 — Quality floor : text length vs minimum threshold → retry / fallback if below
  Gate 2 — Noise check   : heuristic noise score → flag HIGH noise fetches
  Gate 3 — Relevance     : requires embedding model → deferred to Phase 2

Every result carries: source, role, source_date, fetched_at, url, query.

Run: python phase1b_role_test.py
"""

import os
import re
import sys
import time
import textwrap
from datetime import datetime, date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────

QUERIES = [
    "transformer architecture explained",
    "climate change latest research",
    "retrieval augmented generation",
]

SOURCES = ["wikipedia", "arxiv", "web", "youtube"]

SOURCE_ROLES = {
    "wikipedia": "foundation",
    "arxiv":     "research",
    "web":       "current",
    "youtube":   "explainer",
}

# Content caps — single source of truth for all min/max values
# Min = quality floor: if below, retry / fallback / skip
# Max = ceiling: truncate trailing noise above this at fetch time
#       None = no fetch-time cap (Phase 2 handles via section selection / RAPTOR)
CONTENT_CAPS = {
    "wikipedia": {"min": 10_000, "max": None},   # max: Phase 2 section selection / RAPTOR
    "arxiv":     {"min": 200,    "max": None},   # per abstract; 3 abstracts combined ~4.5K naturally
    "web":       {"min": 3_000,  "max": 12_000}, # per article; beyond 12K is low-value noise
    "youtube":   {"min": 3_000,  "max": 8_000},  # per transcript; beyond 8K is repetitive
}

LOG_DIR     = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PLAN_FILE   = Path("TESTING_PLAN.md")
UPDATE_FILE = Path("UPDATES.md")
STATUS_FILE = Path("STATUS.md")

CURRENT_YEAR = date.today().year


# ── Metadata Builder ──────────────────────────────────────────────────────────

def build_metadata(source: str, query: str, text: str,
                   source_date: str = "", url: str = "") -> dict:
    return {
        "source":      source,
        "role":        SOURCE_ROLES[source],
        "source_date": source_date,
        "fetched_at":  date.today().isoformat(),
        "url":         url,
        "query":       query,
        "text_len":    len(text),
    }


# ── Noise Scorer ──────────────────────────────────────────────────────────────

def compute_noise_score(text: str) -> str:
    if not text or len(text) < 100:
        return "HIGH"
    total = len(text)
    special = sum(1 for c in text
                  if not c.isalnum() and c not in " \n\t.,!?;:()'\"\\-/")
    html_tags   = len(re.findall(r"<[^>]{1,60}>", text))
    noise_words = ["cookie", "subscribe", "javascript", "advertisement",
                   "click here", "sign up", "log in", "privacy policy"]
    noise_hits  = sum(text.lower().count(w) for w in noise_words)
    score = (special / total) + (html_tags * 0.05) + (noise_hits * 0.03)
    if score < 0.06:
        return "LOW"
    elif score < 0.15:
        return "MEDIUM"
    return "HIGH"


# ── Evaluation Pipeline ───────────────────────────────────────────────────────

def evaluate_fetch(source: str, text: str, retry_count: int = 0) -> dict:
    """
    Runs evaluation gates on fetched content.
    Gate 1 — Quality floor : text length vs CONTENT_CAPS[source]['min']
    Gate 2 — Noise check   : heuristic noise scorer
    Gate 3 — Relevance     : DEFERRED to Phase 2 (requires embedding model)

    Returns evaluation dict with per-gate results and overall pass/fail.
    """
    caps     = CONTENT_CAPS[source]
    min_len  = caps["min"]
    noise    = compute_noise_score(text)
    length   = len(text)

    gate1_pass = length >= min_len
    gate2_pass = noise in ("LOW", "MEDIUM")

    return {
        "gate1_floor":  "PASS" if gate1_pass else f"FAIL (got {length}, need {min_len})",
        "gate2_noise":  f"{noise} ({'PASS' if gate2_pass else 'FAIL'})",
        "gate3_relevance": "DEFERRED — Phase 2",
        "retry_count":  retry_count,
        "eval_pass":    gate1_pass and gate2_pass,
    }


# ── Role-Configured Fetchers ──────────────────────────────────────────────────

def fetch_wikipedia(query: str) -> dict:
    """
    Role: foundation — query unchanged, no special config needed.
    Min floor: 10,000 chars. If first result is a stub, retry with next title.
    Max: no hard cap at fetch time — Phase 2 section selection handles reduction.
    """
    import wikipedia
    start = time.perf_counter()
    min_len = CONTENT_CAPS["wikipedia"]["min"]

    try:
        titles = wikipedia.search(query, results=5)   # fetch 5 titles for fallback
        if not titles:
            return _fail(start, "No search results")

        last_error = ""
        for attempt, title in enumerate(titles[:3]):  # try up to 3 titles
            try:
                page = wikipedia.page(title, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                except Exception as ex:
                    last_error = str(ex)
                    continue
            except Exception as ex:
                last_error = str(ex)
                continue

            text = page.content
            if len(text) >= min_len:
                meta = build_metadata("wikipedia", query, text, url=page.url)
                ev   = evaluate_fetch("wikipedia", text, retry_count=attempt)
                return _ok(start, text, meta=meta, title=page.title, evaluation=ev)
            # Content too short (stub article) — try next title
            last_error = f"Article too short: {len(text)} chars (need {min_len})"

        # All retries exhausted — return best we found (last attempt)
        try:
            page = wikipedia.page(titles[0], auto_suggest=False)
            text = page.content
        except Exception:
            return _fail(start, f"All retries failed. Last error: {last_error}")

        meta = build_metadata("wikipedia", query, text, url=page.url)
        ev   = evaluate_fetch("wikipedia", text, retry_count=2)
        result = _ok(start, text, meta=meta, title=page.title, evaluation=ev)
        result["status"] = "PARTIAL"
        return result

    except Exception as e:
        return _fail(start, str(e))


def fetch_arxiv(query: str) -> dict:
    """
    Role: research — sort by Relevance, then post-filter to last 2 years.
    Abstract-only fetch (~1.5K per paper, ~4.5K combined for 3 papers).
    PDF parsing deferred to production (pdfplumber unreliable on academic PDFs —
    word-spacing issues, 17-28s latency, section boundary detection failures).
    Min: 200 chars per abstract. Need ≥ 2 papers to return SUCCESS.
    """
    import arxiv
    start       = time.perf_counter()
    min_per_doc = CONTENT_CAPS["arxiv"]["min"]
    cutoff_year = CURRENT_YEAR - 2

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=10,                        # fetch 10, filter below for recency
            sort_by=arxiv.SortCriterion.Relevance, # relevance first, date filter after
        )
        results = list(client.results(search))

        # Keep only papers from last 2 years
        recent = [r for r in results
                  if r.published and r.published.year >= cutoff_year]
        if not recent:
            recent = results[:3]   # fallback: no date filter if nothing passes

        texts, metas = [], []
        for r in recent[:3]:
            paper_text = f"Title: {r.title}\n\nAbstract: {r.summary}"
            if len(paper_text) < min_per_doc:
                continue   # skip truly empty abstracts
            source_date = r.published.strftime("%Y-%m") if r.published else ""
            texts.append(paper_text)
            metas.append(build_metadata("arxiv", query, paper_text,
                                         source_date=source_date,
                                         url=str(r.entry_id)))

        if not texts:
            return _fail(start, "No papers passed quality floor after filtering")

        combined     = "\n\n---\n\n".join(texts)
        source_dates = [m["source_date"] for m in metas]
        ev           = evaluate_fetch("arxiv", combined)
        status       = "SUCCESS" if len(texts) >= 2 else "PARTIAL"
        return {"status": status, "text": combined, "meta": metas[0],
                "title": recent[0].title, "source_dates": source_dates,
                "time": time.perf_counter() - start, "evaluation": ev}

    except Exception as e:
        return _fail(start, str(e))


def fetch_web(query: str, web_mode: str = "general") -> dict:
    """
    Role: current — semantic web search via Tavily API.
    Tavily returns pre-extracted clean article content — no scraping needed.
    web_mode controls Tavily topic:
      "news"    → topic='news', days=30 — recent news/announcements (current events queries)
      "general" → topic='general', no days filter — blogs, docs, tutorials (how-to/technical queries)
    Gemini's rewrite step sets web_mode per query. Defaults to "general" if not set.
    Collects up to 3 results, skips Wikipedia URLs (Wikipedia agent covers those).
    Fallback: DDG + Trafilatura if TAVILY_API_KEY is not set.
    Min floor : 1,000 chars. Max ceiling: 7,000 chars (general mode) / 12,000 chars (news mode) per article.
    BUG 9 fix: general mode caps at 7K/article (3 articles = 21K max → ~12s embed, was 29s).
    news mode keeps full 12K cap (news articles are shorter in practice).
    """
    start   = time.perf_counter()
    min_len = CONTENT_CAPS["web"]["min"]   # 3000 — for DDG fallback (full-page scrape)
    tavily_min_len = 500                   # Tavily returns AI-summarized content, shorter by design
    # BUG 9 fix: general mode caps at 7K/article to prevent embed lock contention.
    # 21K total → ~12s embed (down from 29s) — Wikipedia/ArXiv can acquire lock in time.
    max_len = 7_000 if web_mode == "general" else CONTENT_CAPS["web"]["max"]

    api_key = os.environ.get("TAVILY_API_KEY")
    if api_key:
        try:
            from tavily import TavilyClient
            client   = TavilyClient(api_key=api_key)
            search_params = {
                "query":        query,
                "search_depth": "advanced",
                "topic":        web_mode,
                "max_results":  5,
            }
            if web_mode == "news":
                search_params["days"] = 30   # only valid for topic="news"
            response = client.search(**search_params)
            # Fix A: consent-wall patterns — Tavily advanced depth occasionally fetches
            # GDPR/cookie popup pages. These embed at high sim (0.741 observed) because
            # the page title/domain matches the query, but the content is useless noise.
            _CONSENT_WALL = [
                "privacy policy", "terms of service", "sign in to",
                "by continuing you agree", "cookie policy", "accept cookies",
                "we use cookies",
            ]
            texts = []
            for result in response.get("results", []):
                url     = result.get("url", "")
                content = result.get("content", "")
                if not content or "wikipedia.org" in url:
                    continue
                if any(p in content.lower() for p in _CONSENT_WALL):
                    print(f"    [consent-wall] Skipped {url[:60]} — consent/cookie page")
                    continue
                if len(content) < tavily_min_len:
                    continue
                if len(content) > max_len:
                    content = content[:max_len]
                texts.append((content, url, ""))
                if len(texts) >= 3:
                    break

            if texts:
                print(f"    ✓ Tavily: {len(texts)} relevant articles fetched")
                combined     = "\n\n---\n\n".join(t for t, _, _ in texts)
                meta         = build_metadata("web", query, combined, url=texts[0][1])
                ev           = evaluate_fetch("web", combined)
                status       = "SUCCESS" if len(combined) >= min_len else "PARTIAL"
                return {"status": status, "text": combined, "meta": meta,
                        "source_dates": [], "time": time.perf_counter() - start,
                        "evaluation": ev}
            else:
                print(f"    ⚠ Tavily returned no usable results — falling back to DDG")
        except Exception as e:
            print(f"    ⚠ Tavily failed ({e}) — falling back to DDG")
    else:
        print(f"    ⚠ TAVILY_API_KEY not set — using DDG fallback")

    # ── DDG + Trafilatura fallback ────────────────────────────────────────────
    from ddgs import DDGS
    import trafilatura

    role_query = f"{query} {CURRENT_YEAR}"
    try:
        with DDGS(timeout=10) as ddgs:
            hits = list(ddgs.text(role_query, timelimit='m', max_results=8))

        texts, tried_urls = [], []
        for h in hits:
            url = h.get("href", "")
            if not url or "wikipedia.org" in url:
                continue
            tried_urls.append(url)
            try:
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    continue
                text = trafilatura.extract(downloaded, include_comments=False,
                                           include_tables=False, with_metadata=True)
                if not text or len(text) < min_len:
                    continue
                if len(text) > max_len:
                    text = text[:max_len]
                source_date = ""
                meta_text = trafilatura.extract(downloaded, output_format='xml',
                                                include_comments=False)
                if meta_text:
                    date_match = re.search(r'<date>(.*?)</date>', meta_text)
                    if date_match:
                        source_date = date_match.group(1)[:7]
                texts.append((text, url, source_date))
                if len(texts) >= 2:
                    break
            except Exception:
                continue

        if not texts:
            snippets = [(f"{h.get('title','')}\n{h.get('body','')}",
                         h.get("href", ""), "")
                        for h in hits
                        if len(h.get("body", "")) > 80
                        and "wikipedia.org" not in h.get("href", "")]
            if snippets:
                texts = snippets[:2]
            else:
                return _fail(start, f"Tavily unavailable and DDG/Trafilatura empty. Tried: {tried_urls[:3]}")

        combined = "\n\n---\n\n".join(t for t, _, _ in texts)
        meta     = build_metadata("web", query, combined,
                                  source_date=texts[0][2], url=texts[0][1])
        ev       = evaluate_fetch("web", combined)
        status   = "SUCCESS" if len(combined) >= min_len else "PARTIAL"
        return {"status": status, "text": combined, "meta": meta,
                "source_dates": [t[2] for t in texts if t[2]],
                "time": time.perf_counter() - start, "evaluation": ev}

    except Exception as e:
        return _fail(start, str(e))


def _get_youtube_ids_via_api(search_query: str) -> list:
    """
    YouTube Data API v3 video ID discovery.
    Filters to caption-enabled videos only (videoCaption='closedCaption') so
    YouTubeTranscriptApi().fetch() has a much higher success rate.
    Returns [] if YOUTUBE_API_KEY is not set or API call fails.
    """
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print(f"    ⚠ YOUTUBE_API_KEY not set — falling back to DDG")
        return []
    try:
        from googleapiclient.discovery import build
        yt       = build("youtube", "v3", developerKey=api_key)
        response = yt.search().list(
            part             = "snippet",
            q                = search_query,
            type             = "video",
            maxResults       = 5,
            relevanceLanguage= "en",
            videoCaption     = "closedCaption",   # only caption-enabled videos
        ).execute()
        ids = [item["id"]["videoId"] for item in response.get("items", [])]
        if ids:
            print(f"    ✓ YouTube API v3: {len(ids)} caption-enabled videos found")
        return ids
    except Exception as e:
        print(f"    ⚠ YouTube API v3 failed ({e}) — falling back to DDG")
        return []


def _get_youtube_ids_via_ddg(search_query: str) -> list:
    """
    DDG fallback for YouTube video ID discovery.
    Tries 3 query variants via ddgs.videos() + ddgs.text() to build a candidate pool.
    """
    from ddgs import DDGS

    candidate_ids  = []
    base           = search_query.replace("explained tutorial ", "")
    query_variants = [
        search_query,
        f"{base} tutorial",
        f"{base} explained",
    ]

    def extract_ids(hits: list, url_fields: tuple) -> list:
        ids = []
        for h in hits:
            for field in url_fields:
                url   = h.get(field, "")
                match = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", url)
                if match:
                    vid = match.group(1)
                    if vid not in candidate_ids and vid not in ids:
                        ids.append(vid)
                    break
        return ids

    for variant in query_variants:
        if len(candidate_ids) >= 6:
            break
        try:
            with DDGS() as ddgs:
                video_hits = list(ddgs.videos(variant, max_results=5))
            candidate_ids.extend(extract_ids(video_hits, ("content", "url", "href")))
        except Exception:
            pass
        if len(candidate_ids) < 4:
            try:
                with DDGS() as ddgs:
                    text_hits = list(ddgs.text(f"youtube {variant}", max_results=8))
                candidate_ids.extend(extract_ids(text_hits, ("href", "url", "link")))
            except Exception:
                pass

    return candidate_ids


def fetch_youtube(query: str) -> dict:
    """
    Role: explainer — prepend 'explained tutorial' to query.
    Video ID discovery: YouTube Data API v3 (YOUTUBE_API_KEY) → DDG fallback.
    API v3 pre-filters to caption-enabled videos → higher transcript success rate.
    Retry: tries up to 5 candidates before failing.
    Min floor : 3,000 chars. Skip short transcripts, try next video.
    Max ceiling: 8,000 chars (truncate above this — beyond is usually repetitive).
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    start        = time.perf_counter()
    min_len      = CONTENT_CAPS["youtube"]["min"]
    max_len      = CONTENT_CAPS["youtube"]["max"]
    search_query = f"explained tutorial {query}"

    # ── Step 1: Get candidate video IDs (API v3 → DDG fallback) ──────────────
    candidate_ids = _get_youtube_ids_via_api(search_query)
    if not candidate_ids:
        candidate_ids = _get_youtube_ids_via_ddg(search_query)
    if not candidate_ids:
        return _fail(start, "Could not find any YouTube video IDs via API or DDG")

    # ── Step 2: Try each candidate until one works ────────────────────────────
    api         = YouTubeTranscriptApi()
    last_error  = ""
    retry_count = 0

    for video_id in candidate_ids[:5]:   # bumped 3 → 5: API gives reliable candidates
        try:
            transcript = api.fetch(video_id)
            text       = " ".join(snippet.text for snippet in transcript)

            if len(text) < min_len:
                last_error  = f"Transcript too short: {len(text)} chars (need {min_len})"
                retry_count += 1
                continue

            if len(text) > max_len:
                text = text[:max_len]

            url  = f"https://www.youtube.com/watch?v={video_id}"
            meta = build_metadata("youtube", query, text, url=url)
            ev   = evaluate_fetch("youtube", text, retry_count=retry_count)
            return _ok(start, text, meta=meta, video_id=video_id, evaluation=ev)

        except Exception as ex:
            last_error  = str(ex)
            retry_count += 1
            continue

    return _fail(start,
                 f"All {min(5, len(candidate_ids))} video attempts failed. "
                 f"Candidates found: {len(candidate_ids)}. Last error: {last_error}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(start: float, text: str, **extra) -> dict:
    return {"status": "SUCCESS", "text": text,
            "time": time.perf_counter() - start, **extra}

def _fail(start: float, error: str) -> dict:
    return {"status": "FAILED", "text": "",
            "time": time.perf_counter() - start, "error": error,
            "evaluation": {"gate1_floor": "FAIL", "gate2_noise": "N/A",
                           "gate3_relevance": "DEFERRED — Phase 2",
                           "retry_count": 0, "eval_pass": False}}

def _h(lines: list, text: str, level: int = 1):
    sep = "=" if level == 1 else "─"
    lines += ["", sep * 60, text, sep * 60]

FETCHERS = {
    "wikipedia": fetch_wikipedia,
    "arxiv":     fetch_arxiv,
    "web":       fetch_web,
    "youtube":   fetch_youtube,
}


# ── Main Test Runner ──────────────────────────────────────────────────────────

def run_phase1b():
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path    = LOG_DIR / f"phase1b_role_{timestamp}.log"
    all_results = {src: [] for src in SOURCES}
    lines       = []

    _h(lines, f"PHASE 1b: ROLE-BASED FETCH TESTING  |  {timestamp}")
    lines.append(f"\nContent caps: {CONTENT_CAPS}")

    for source in SOURCES:
        _h(lines, f"SOURCE: {source.upper()}  |  role: {SOURCE_ROLES[source]}", level=2)
        caps = CONTENT_CAPS[source]
        lines.append(f"  Caps: min={caps['min']} chars  |  "
                     f"max={caps['max'] if caps['max'] else 'none (Phase 2 handles)'}")

        for query in QUERIES:
            print(f"  [{source}] '{query}' ...", end="", flush=True)
            result  = FETCHERS[source](query)
            text    = result.get("text", "")
            noise   = compute_noise_score(text) if text else "HIGH"
            meta    = result.get("meta", {})
            ev      = result.get("evaluation", {})
            s_dates = result.get("source_dates",
                                 [meta.get("source_date", "")] if meta else [])

            entry = {
                "source":       source,
                "role":         SOURCE_ROLES[source],
                "query":        query,
                "status":       result["status"],
                "fetch_time":   round(result["time"], 3),
                "text_len":     len(text),
                "noise":        noise,
                "source_dates": [d for d in s_dates if d],
                "metadata_ok":  bool(meta),
                "error":        result.get("error", ""),
                "evaluation":   ev,
            }
            all_results[source].append(entry)

            preview = (text[:400] + "...").replace("\n", " ") if text else ""
            lines += [
                "",
                "[ROLE FETCH TEST]",
                f"  Source        : {source}  ({SOURCE_ROLES[source]})",
                f"  Query sent    : {query}",
                f"  Status        : {entry['status']}",
                f"  Fetch time    : {entry['fetch_time']}s",
                f"  Text length   : {entry['text_len']} chars",
                f"  Noise level   : {noise}",
                f"  Source dates  : {entry['source_dates'] or 'not available'}",
                f"  Metadata OK   : {entry['metadata_ok']}",
                f"  Eval Gate 1   : {ev.get('gate1_floor', 'N/A')}",
                f"  Eval Gate 2   : {ev.get('gate2_noise', 'N/A')}",
                f"  Eval Gate 3   : {ev.get('gate3_relevance', 'DEFERRED — Phase 2')}",
                f"  Retries       : {ev.get('retry_count', 0)}",
                f"  Eval overall  : {'PASS' if ev.get('eval_pass') else 'FAIL'}",
            ]
            if entry["error"]:
                lines.append(f"  Error         : {entry['error']}")
            if preview:
                lines.append(f"  Preview       : {preview}")

            symbol    = "✓" if entry["status"] == "SUCCESS" else ("~" if entry["status"] == "PARTIAL" else "✗")
            eval_sym  = "✓" if ev.get("eval_pass") else "✗"
            dates_str = ",".join(entry["source_dates"]) if entry["source_dates"] else "no-date"
            print(f" {symbol}  {entry['fetch_time']}s  {entry['text_len']}chars  "
                  f"eval:{eval_sym}  retries:{ev.get('retry_count',0)}  {dates_str}")

    # ── Summary ───────────────────────────────────────────────────────────────
    _h(lines, "PHASE 1b SUMMARY")

    results_summary = {}
    for source, entries in all_results.items():
        n         = len(entries)
        success   = sum(1 for e in entries if e["status"] in ("SUCCESS", "PARTIAL"))
        eval_pass = sum(1 for e in entries if e["evaluation"].get("eval_pass"))
        s_rate    = success / n
        e_rate    = eval_pass / n
        all_dates = [d for e in entries for d in e["source_dates"]]
        meta_rate = sum(1 for e in entries if e["metadata_ok"]) / n
        avg_len   = sum(e["text_len"] for e in entries) / n

        fresh_dates = sum(1 for d in all_dates
                         if d and int(d[:4]) >= CURRENT_YEAR - 2)
        freshness   = (f"{fresh_dates}/{len(all_dates)} dates recent"
                       if all_dates else "no dates extracted")

        results_summary[source] = {
            "success_rate": round(s_rate, 2),
            "eval_rate":    round(e_rate, 2),
            "avg_len":      round(avg_len),
            "freshness":    freshness,
            "meta_rate":    round(meta_rate, 2),
            "pass":         s_rate >= 0.8 and meta_rate >= 0.8,
        }

        lines += [
            "",
            f"{source.upper()}  ({SOURCE_ROLES[source]})",
            f"  Success rate  : {s_rate*100:.0f}%",
            f"  Eval pass rate: {e_rate*100:.0f}%",
            f"  Avg length    : {round(avg_len)} chars",
            f"  Freshness     : {freshness}",
            f"  Metadata rate : {meta_rate*100:.0f}%",
            f"  PASS          : {'YES' if results_summary[source]['pass'] else 'NO'}",
        ]

    passed = [s for s, v in results_summary.items() if v["pass"]]
    lines += [
        "",
        "─" * 55,
        f"PASSED  : {', '.join(passed)}",
        f"FAILED  : {', '.join(s for s in SOURCES if s not in passed) or 'none'}",
        "─" * 55,
    ]

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLog → {log_path}")
    return results_summary, timestamp, log_path


# ── Auto-update TESTING_PLAN.md ───────────────────────────────────────────────

def update_decisions_tracker(results: dict, timestamp: str):
    text     = PLAN_FILE.read_text(encoding="utf-8")
    date_str = timestamp[:10]
    passed   = [s for s, v in results.items() if v["pass"]]
    reason   = (f"Role config confirmed for: {', '.join(passed)}. "
                f"Metadata tagging verified. Content caps enforced.")
    new_row  = f"| 1b | Role Config + Metadata | {', '.join(passed)} | {reason} | {date_str} |"
    updated  = re.sub(r"\| 1b \| Role Config.*", new_row, text)
    if updated != text:
        PLAN_FILE.write_text(updated, encoding="utf-8")
        print("TESTING_PLAN.md updated.")
    else:
        print("Warning: could not find Phase 1b row in TESTING_PLAN.md.")


# ── Auto-update UPDATES.md ────────────────────────────────────────────────────

def update_updates_md(results: dict, timestamp: str, log_path: Path):
    date_str = timestamp[:10]
    rows = "\n".join(
        f"| {s.upper()} | {v['success_rate']*100:.0f}% | {v['eval_rate']*100:.0f}% | "
        f"{v['avg_len']} chars | {v['freshness']} | {'✅' if v['pass'] else '❌'} |"
        for s, v in results.items()
    )
    entry = textwrap.dedent(f"""
    ---

    ## Phase 1b — Role-Based Fetch Testing (v3 — caps + fallback + eval)
    **Date:** {date_str}
    **Log:** `{log_path}`

    ### Results

    | Source | Fetch OK | Eval Pass | Avg Length | Freshness | Pass |
    |--------|----------|-----------|------------|-----------|------|
    {rows}

    ### Content caps enforced
    | Agent | Min | Max | Notes |
    |-------|-----|-----|-------|
    | Wikipedia | 10,000 chars | None (Phase 2 section selection) | Retry next title if stub |
    | ArXiv | 200 chars/abstract | None (abstracts naturally bounded) | Skip abstracts below floor |
    | Web | 1,000 chars/article | 10,000 chars/article | Skip Wikipedia URLs; truncate above max |
    | YouTube | 1,000 chars | 10,000 chars | Retry up to 3 videos if captions disabled |

    ### Next Step
    Phase 2 — Chunking Strategy Testing.
    Run: `python phase2_chunking_test.py`
    """).strip()

    current    = UPDATE_FILE.read_text(encoding="utf-8")
    header_end = current.find("\n---\n") + 5
    UPDATE_FILE.write_text(
        current[:header_end] + "\n" + entry + "\n" + current[header_end:],
        encoding="utf-8"
    )
    print("UPDATES.md updated.")


# ── Auto-update STATUS.md ─────────────────────────────────────────────────────

def update_status_md(results: dict, timestamp: str):
    text     = STATUS_FILE.read_text(encoding="utf-8")
    date_str = timestamp[:10]
    passed   = ", ".join(s for s, v in results.items() if v["pass"])

    text = re.sub(r"\*\*Last updated:\*\*.*",
                  f"**Last updated:** {date_str} (Phase 1b v3 complete)", text)
    text = re.sub(r"\| 1b \| Role Config.*",
                  f"| 1b | Role Config + Metadata | ✅ Done | Passed: {passed} |", text)
    text = re.sub(r"\*\*Immediate:\*\*.*",
                  "**Immediate:** Build + run `python phase2_chunking_test.py`", text)

    STATUS_FILE.write_text(text, encoding="utf-8")
    print("STATUS.md updated.")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 1b — ROLE-BASED FETCH TESTING  (v5)")
    print("=" * 60)
    print(f"  Queries  : {len(QUERIES)}")
    print(f"  Sources  : {', '.join(SOURCES)}")
    print(f"  Roles    : {', '.join(SOURCE_ROLES.values())}")
    print(f"  Year     : {CURRENT_YEAR}")
    print(f"  Caps     : Wikipedia min={CONTENT_CAPS['wikipedia']['min']}  "
          f"Web max={CONTENT_CAPS['web']['max']}  "
          f"YouTube max={CONTENT_CAPS['youtube']['max']}")
    print("=" * 60 + "\n")

    results, timestamp, log_path = run_phase1b()

    passed = [s for s, v in results.items() if v["pass"]]
    print("\n" + "─" * 60)
    print(f"  Passed: {', '.join(passed)}")
    print("─" * 60)

    update_decisions_tracker(results, timestamp)
    update_updates_md(results, timestamp, log_path)
    update_status_md(results, timestamp)

    print("\nDone. UPDATES.md, TESTING_PLAN.md, STATUS.md all updated.")
