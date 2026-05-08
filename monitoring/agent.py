"""Nexus AI — nightly monitoring agent.

Cron (EC2): 0 1 * * * /home/ubuntu/Nexus-AI/venv/bin/python /home/ubuntu/Nexus-AI/monitoring/agent.py
"""

import json
import os
import random
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from statistics import mean

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SOURCES          = ("wiki", "arxiv", "web", "yt")
LOOK_BACK_HOURS  = 24
BASELINE_DAYS    = 7
MAX_OUTLIER_ROWS = 10
SAMPLE_SIZE      = 30
GEMINI_MODEL     = "gemini-2.5-flash"

# Percentile bucket boundaries (seconds)
PCTILE_FAST    = 30   # below this = suspiciously fast
PCTILE_HEALTHY = 45   # P30–P45 = normal range
PCTILE_SLOW    = 65   # P45–P65 = starting to degrade
PCTILE_OUTLIER = 90   # above this = true outlier

# Outlier trigger thresholds — all pure Python, no AI involvement
P90_EXTREME_FACTOR  = 1.5  # total_time_s > P90 * 1.5  → extreme slow
MIN_SIM_GATE        = 0.65 # avg_sim_score below this   → poor retrieval quality
THIN_CONTEXT_CHUNKS = 3    # chunks_to_llm <= this AND quality==POOR → silent failure (BUG G)
CASCADE_FAIL_MIN    = 2    # ≥ 2 sources failed         → cascade failure
MIN_RETRY_OUTLIER   = 2    # llm_retry_count >= this    → Gemini was struggling


# ── Supabase ───────────────────────────────────────────────────────────────────
def _get_supabase() -> SupabaseClient:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])


# ── Data layer ─────────────────────────────────────────────────────────────────
def fetch_rows(db: SupabaseClient) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(hours=LOOK_BACK_HOURS)).isoformat()
    resp = (
        db.table("queries")
        .select(
            "query_id,user_id,query_text,timestamp,quality,"
            "total_time_s,rewrite_time_s,fetch_wall_time_s,llm_time_s,"
            "llm_retry_count,rewrite_success,web_mode,"
            "chunks_to_llm,avg_sim_score,min_sim_score,sources,error_log"
            # answer_text deliberately excluded — ~700 tokens/row blows Gemini budget (BUG C)
        )
        .gte("timestamp", since)
        .order("timestamp", desc=False)
        .execute()
    )
    return resp.data or []


def fetch_baseline(db: SupabaseClient) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=BASELINE_DAYS)).date().isoformat()
    resp = (
        db.table("daily_summaries")
        .select("date,total_queries,p45_time_s,p90_time_s,quality_dist,src_failures,avg_sim_score")
        .gte("date", since)
        .order("date", desc=False)
        .execute()
    )
    return resp.data or []


# ── Outlier detection ──────────────────────────────────────────────────────────
def _is_outlier(row: dict, p30: float, p90: float) -> bool:
    t       = row.get("total_time_s") or 0.0
    quality = row.get("quality") or ""
    chunks  = row.get("chunks_to_llm") or 0
    sim     = row.get("avg_sim_score") or 1.0
    retries = row.get("llm_retry_count") or 0
    sources = row.get("sources") or {}

    failed_count = sum(
        1 for s in sources.values()
        if s.get("status") not in ("ok", "skipped")
    )

    return any([
        t > p90 * P90_EXTREME_FACTOR,
        quality == "POOR" and t < p30,                      # fast + POOR = silent failure
        failed_count >= CASCADE_FAIL_MIN,                    # cascade source failure
        chunks <= THIN_CONTEXT_CHUNKS and quality == "POOR", # thin context only flags if answer was POOR (BUG G)
        sim < MIN_SIM_GATE,
        retries >= MIN_RETRY_OUTLIER,
        row.get("error_log") is not None,
    ])


# ── Metrics computation ────────────────────────────────────────────────────────
def compute_metrics(rows: list[dict]) -> dict:
    times = [r["total_time_s"] for r in rows if r.get("total_time_s") is not None]

    # Crash-only rows have no timing — treat as 0 so percentiles don't blow up
    if not times:
        times = [0.0]

    p30 = float(np.percentile(times, PCTILE_FAST))
    p45 = float(np.percentile(times, PCTILE_HEALTHY))
    p65 = float(np.percentile(times, PCTILE_SLOW))
    p90 = float(np.percentile(times, PCTILE_OUTLIER))

    quality_dist: dict[str, int] = {"GOOD": 0, "OK": 0, "POOR": 0}
    for r in rows:
        q = r.get("quality") or "POOR"
        if q in quality_dist:
            quality_dist[q] += 1

    src_failures: dict[str, int]        = {s: 0 for s in SOURCES}
    src_times:    dict[str, list[float]] = {s: [] for s in SOURCES}

    for r in rows:
        for src, data in (r.get("sources") or {}).items():
            if src not in SOURCES or src == "yt":
                continue  # yt is permanently blocked on EC2 AWS IPs — never count as real failure
            if data.get("status") not in ("ok", "skipped"):
                src_failures[src] += 1
            arrived = data.get("arrived_at_s")
            if isinstance(arrived, (int, float)) and data.get("status") == "ok":
                src_times[src].append(float(arrived))

    # BUG F: never divide by zero — use None when a source had zero successful fetches
    src_avg_time = {
        s: round(mean(src_times[s]), 2) if src_times[s] else None
        for s in SOURCES
    }

    sim_vals    = [r["avg_sim_score"] for r in rows if r.get("avg_sim_score") is not None]
    retry_flags = [r["llm_retry_count"] > 0 for r in rows if r.get("llm_retry_count") is not None]
    outliers    = [r for r in rows if _is_outlier(r, p30, p90)]

    return {
        "total_queries":  len(rows),
        "p30_time_s":     round(p30, 2),
        "p45_time_s":     round(p45, 2),
        "p65_time_s":     round(p65, 2),
        "p90_time_s":     round(p90, 2),
        "quality_dist":   quality_dist,
        "src_failures":   src_failures,
        "src_avg_time":   src_avg_time,
        "avg_sim_score":  round(mean(sim_vals), 4) if sim_vals else None,
        "llm_retry_rate": round(mean([float(f) for f in retry_flags]), 4) if retry_flags else 0.0,
        "error_count":    sum(1 for r in rows if r.get("error_log") is not None),
        "outlier_count":  len(outliers),
        # Private — used internally by segment_rows and gemini_diagnose, not written to DB
        "_outliers": outliers,
        "_p30": p30,
        "_p45": p45,
        "_p90": p90,
    }


def segment_rows(rows: list[dict], p30: float, p45: float, p90: float) -> dict[str, list[dict]]:
    fast, slow, extreme = [], [], []
    for r in rows:
        t = r.get("total_time_s") or 0.0
        if t < p30:
            fast.append(r)
        elif t > p90:
            extreme.append(r)
        elif t > p45:
            slow.append(r)
    return {"fast": fast, "slow": slow, "extreme": extreme}


# ── Gemini diagnosis ───────────────────────────────────────────────────────────
_GEMINI_STRIP = frozenset({"answer_text", "user_id", "min_sim_score", "web_mode", "rewrite_success"})

def _strip_for_gemini(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _GEMINI_STRIP}


# Failure fingerprinting — priority order: first matching condition wins.
# Each outlier gets exactly one label so no failure type crowds out another.
_FAILURE_TYPES = (
    "CRASH",         # error_log is not null
    "CASCADE",       # 2+ sources failed simultaneously
    "EXTREME_SLOW",  # total_time_s > P90 × 1.5
    "ARXIV_TIMEOUT",
    "WIKI_TIMEOUT",
    "WEB_TIMEOUT",
    "LOW_SIM",       # avg_sim_score < 0.65
    "LLM_OVERLOAD",  # llm_retry_count >= 2
    "FAST_POOR",     # quality==POOR and time < P30 (silent failure)
    "THIN_CONTEXT",  # chunks_to_llm <= 3 and quality==POOR
)


def _fingerprint_outlier(row: dict, p30: float, p90: float) -> str:
    t       = row.get("total_time_s") or 0.0
    quality = row.get("quality") or ""
    chunks  = row.get("chunks_to_llm") or 0
    sim     = row.get("avg_sim_score") or 1.0
    retries = row.get("llm_retry_count") or 0
    sources = row.get("sources") or {}

    failed_count = sum(
        1 for s in sources.values()
        if s.get("status") not in ("ok", "skipped")
    )

    if row.get("error_log") is not None:                                          return "CRASH"
    if failed_count >= CASCADE_FAIL_MIN:                                          return "CASCADE"
    if t > p90 * P90_EXTREME_FACTOR:                                             return "EXTREME_SLOW"
    if (sources.get("arxiv") or {}).get("status") in ("timeout", "error"):       return "ARXIV_TIMEOUT"
    if (sources.get("wiki")  or {}).get("status") in ("timeout", "error"):       return "WIKI_TIMEOUT"
    if (sources.get("web")   or {}).get("status") in ("timeout", "error"):       return "WEB_TIMEOUT"
    if sim < MIN_SIM_GATE:                                                        return "LOW_SIM"
    if retries >= MIN_RETRY_OUTLIER:                                              return "LLM_OVERLOAD"
    if quality == "POOR" and t < p30:                                            return "FAST_POOR"
    return "THIN_CONTEXT"


def _count_conditions(row: dict, p30: float, p90: float) -> int:
    """Count how many outlier conditions are true — used to pick the worst row per group."""
    t       = row.get("total_time_s") or 0.0
    quality = row.get("quality") or ""
    chunks  = row.get("chunks_to_llm") or 0
    sim     = row.get("avg_sim_score") or 1.0
    retries = row.get("llm_retry_count") or 0
    sources = row.get("sources") or {}

    failed_count = sum(
        1 for s in sources.values()
        if s.get("status") not in ("ok", "skipped")
    )

    return sum([
        row.get("error_log") is not None,
        failed_count >= CASCADE_FAIL_MIN,
        t > p90 * P90_EXTREME_FACTOR,
        (sources.get("arxiv") or {}).get("status") in ("timeout", "error"),
        (sources.get("wiki")  or {}).get("status") in ("timeout", "error"),
        (sources.get("web")   or {}).get("status") in ("timeout", "error"),
        sim < MIN_SIM_GATE,
        retries >= MIN_RETRY_OUTLIER,
        quality == "POOR" and t < p30,
        chunks <= THIN_CONTEXT_CHUNKS and quality == "POOR",
    ])


def _group_outliers_by_type(
    outliers: list[dict], p30: float, p90: float
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in outliers:
        ftype = _fingerprint_outlier(row, p30, p90)
        groups.setdefault(ftype, []).append(row)
    return groups


def gemini_diagnose(
    outliers: list[dict],
    all_rows: list[dict],
    metrics:  dict,
    baseline: list[dict],
) -> str:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)

    # BUG E: on the first ever run daily_summaries is empty — never crash on this
    baseline_text = (
        json.dumps(baseline, indent=2, default=str)
        if baseline
        else "First run — no historical baseline data yet."
    )

    # Exclude internal _* keys from the prompt
    prompt_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}

    # Failure fingerprinting — one representative row per failure type (max 10 types → max 10 rows)
    p30 = metrics["_p30"]
    p90 = metrics["_p90"]
    groups = _group_outliers_by_type(outliers, p30, p90)
    fingerprinted_outliers = [
        {
            "failure_type":   ftype,
            "count":          len(group_rows),
            "representative": _strip_for_gemini(
                max(group_rows, key=lambda r: _count_conditions(r, p30, p90))
            ),
        }
        for ftype, group_rows in groups.items()
    ]

    outlier_ids   = {r.get("query_id") for r in outliers}
    sample_pool   = [r for r in all_rows if r.get("query_id") not in outlier_ids]
    random_sample = [
        _strip_for_gemini(r)
        for r in random.sample(sample_pool, min(SAMPLE_SIZE, len(sample_pool)))
    ]

    prompt = f"""You are a site reliability analyst reviewing the daily health of Nexus AI — a multi-agent RAG system that fetches from Wikipedia, ArXiv, Web (Tavily), and YouTube in parallel, then synthesises an answer with Gemini 2.5 Flash.

Note: YouTube always fails on EC2 AWS IPs (blocked by YouTube for cloud datacenter ranges). Treat yt status="error" as expected — do not flag it as a system issue.

## Today's computed metrics
{json.dumps(prompt_metrics, indent=2, default=str)}

## 7-day historical baseline
{baseline_text}

## Outlier queries — fingerprinted by failure type (one representative per type, PII stripped)
Each entry shows the failure type, how many queries had it today, and the single worst representative row.
{json.dumps(fingerprinted_outliers, indent=2, default=str)}

## Random sample ({len(random_sample)} queries, PII stripped)
{json.dumps(random_sample, indent=2, default=str)}

Write a concise daily health report with exactly these sections:

1. SUMMARY — 2–3 sentences: overall health, any regressions vs baseline.
2. FAST QUERIES (<P30={metrics['p30_time_s']}s) — are these suspiciously fast (skipped sources, thin context) or just efficient?
3. SLOW QUERIES (P45–P90) — identify which source is the bottleneck per query if possible.
4. OUTLIERS (>P90={metrics['p90_time_s']}s) — root cause per query, be specific.
5. RECOMMENDATIONS — max 3 bullets, actionable only. Skip if nothing needs fixing.

If a section has nothing notable, write "None today." Be concise. No filler."""

    resp = model.generate_content(prompt)
    return resp.text


# ── Database write ─────────────────────────────────────────────────────────────
def write_daily_summary(db: SupabaseClient, today: str, metrics: dict) -> None:
    row = {
        "date":           today,
        "total_queries":  metrics["total_queries"],
        "p30_time_s":     metrics["p30_time_s"],
        "p45_time_s":     metrics["p45_time_s"],
        "p65_time_s":     metrics["p65_time_s"],
        "p90_time_s":     metrics["p90_time_s"],
        "quality_dist":   metrics["quality_dist"],
        "src_failures":   metrics["src_failures"],
        "src_avg_time":   metrics["src_avg_time"],
        "avg_sim_score":  metrics["avg_sim_score"],
        "llm_retry_rate": metrics["llm_retry_rate"],
        "error_count":    metrics["error_count"],
        "outlier_count":  metrics["outlier_count"],
    }
    db.table("daily_summaries").upsert(row, on_conflict="date").execute()


# ── Email ──────────────────────────────────────────────────────────────────────
def _build_email_body(
    metrics:   dict,
    narrative: str,
    today:     str,
    segments:  dict[str, list[dict]],
) -> str:
    m = metrics
    src_lines = "\n".join(
        "  yt    Dropped (AWS IP block — expected)"
        if s == "yt"
        else (
            f"  {s:<5} {m['src_failures'][s]} failures | avg fetch "
            + (f"{m['src_avg_time'].get(s)}s" if m['src_avg_time'].get(s) is not None else "N/A")
        )
        for s in SOURCES
    )

    return (
        f"NEXUS AI — Daily Health Report ({today})\n"
        f"{'=' * 52}\n\n"
        f"QUERIES      : {m['total_queries']} in last 24h\n"
        f"QUALITY      : GOOD={m['quality_dist']['GOOD']}  OK={m['quality_dist']['OK']}  POOR={m['quality_dist']['POOR']}\n"
        f"AVG SIM      : {m['avg_sim_score']}\n"
        f"LLM RETRIES  : {m['llm_retry_rate'] * 100:.1f}% of queries\n"
        f"ERRORS       : {m['error_count']}\n\n"
        f"TIMING PERCENTILES\n"
        f"  P30={m['p30_time_s']}s  P45={m['p45_time_s']}s  P65={m['p65_time_s']}s  P90={m['p90_time_s']}s\n\n"
        f"SOURCE HEALTH\n"
        f"{src_lines}\n\n"
        f"SEGMENTS\n"
        f"  FAST  (<{m['p30_time_s']}s) : {len(segments['fast'])} queries\n"
        f"  SLOW  (>{m['p45_time_s']}s) : {len(segments['slow'])} queries\n"
        f"  P90+  (>{m['p90_time_s']}s) : {len(segments['extreme'])} queries\n"
        f"  FLAGGED OUTLIERS : {m['outlier_count']}\n\n"
        f"{'─' * 52}\n"
        f"GEMINI ANALYSIS\n"
        f"{'─' * 52}\n"
        f"{narrative}\n\n"
        f"{'─' * 52}\n"
        f"Sent by monitoring/agent.py at "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
    )


def send_email(subject: str, body: str) -> None:
    sender    = os.environ["GMAIL_SENDER"]
    receivers = [r.strip() for r in os.environ["GMAIL_RECEIVER"].split(",")]
    password  = os.environ["GMAIL_APP_PASSWORD"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = ", ".join(receivers)
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(sender, password)
        smtp.sendmail(sender, receivers, msg.as_string())


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    print(f"[agent] Starting — date={today}", flush=True)

    db   = _get_supabase()
    rows = fetch_rows(db)
    print(f"[agent] Fetched {len(rows)} rows from last {LOOK_BACK_HOURS}h", flush=True)

    # BUG D: zero queries — notify and exit cleanly (np.percentile([],n) would raise ValueError)
    if not rows:
        send_email(
            subject=f"Nexus AI — No Queries ({today})",
            body=f"No queries received in the last {LOOK_BACK_HOURS} hours.\n\nDate: {today}",
        )
        print("[agent] Zero queries today — email sent.", flush=True)
        return

    metrics  = compute_metrics(rows)
    segments = segment_rows(rows, metrics["_p30"], metrics["_p45"], metrics["_p90"])
    baseline = fetch_baseline(db)

    print(
        f"[agent] Metrics done — outliers={metrics['outlier_count']} "
        f"baseline_rows={len(baseline)}",
        flush=True,
    )

    # Gemini failure is non-fatal — Python metrics are complete without it
    try:
        narrative = gemini_diagnose(metrics["_outliers"], rows, metrics, baseline)
        print("[agent] Gemini narrative received.", flush=True)
    except Exception as exc:
        narrative = f"[Gemini unavailable — {exc}]\nPython-computed metrics above are complete."
        print(f"[agent] WARNING: Gemini failed: {exc}", flush=True)

    write_daily_summary(db, today, metrics)
    print("[agent] daily_summaries row written.", flush=True)

    body = _build_email_body(metrics, narrative, today, segments)
    send_email(
        subject=(
            f"Nexus AI — Daily Report ({today}) | "
            f"{metrics['total_queries']} queries | "
            f"GOOD={metrics['quality_dist']['GOOD']}"
        ),
        body=body,
    )
    print("[agent] Email sent. Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[agent] FATAL: {exc}", flush=True)
        sys.exit(1)
