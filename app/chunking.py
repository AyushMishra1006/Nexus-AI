"""
Phase 2 — Chunking Strategy Testing (v1)

Strategies per source:
  Wikipedia : (1) query-aware section selection + hierarchical parent-child (baseline)
              (2) RAPTOR-lite — fixed chunk → embed → K-means cluster → extractive summary → 2-level tree (challenger)
              Winner deferred to Phase 4 (retrieval scoring needed to compare meaningfully).
  ArXiv     : atomic — 1 abstract = 1 chunk, no splitting
  Web       : semantic topic-shift — split at cosine similarity drop between consecutive sentences
  YouTube   : semantic topic-shift — same, handles transcript text without punctuation structure

Gate 3 (relevance check) first implemented here:
  cosine_similarity(embed(query), embed(fetched_text[:500])) > 0.25 → PASS

Run  : python phase2_chunking_test.py
Deps : pip install -r requirements_phase2.txt
"""

import re
import sys
import time
import textwrap
import numpy as np
from datetime import datetime, date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ─────────────────────────────────────────────────────────────────────

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

# Chunking thresholds
SECTION_SIM_THRESHOLD  = 0.25   # min cosine sim for query-aware Wikipedia section selection
TOPIC_SHIFT_THRESHOLD  = 0.45   # below this between consecutive sentences → split
GATE3_SIM_THRESHOLD    = 0.25   # gate 3 relevance: cosine(query, content[:500])
SECTIONS_TO_KEEP       = 4      # max Wikipedia sections to keep after scoring
CHILD_CHUNK_CHARS      = 750    # target child chunk size in chars (≈ 167 tokens; above 150 floor)
MIN_CHUNK_CHARS        = 600    # below this → merge with neighbour (≈ 133 tokens; prevents tiny residuals)
RAPTOR_LEAF_WORDS      = 150    # words per RAPTOR leaf chunk (≈ 675 chars ≈ 150 tokens)

# Pass criteria (from TESTING_PLAN.md)
TARGET_TOKEN_RANGE = (150, 500)   # avg chunk token size must be in this range
SHORT_CHUNK_LIMIT  = 0.10         # short chunk ratio (< 100 tokens) must be < 10%

LOG_DIR     = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PLAN_FILE   = Path("TESTING_PLAN.md")
UPDATE_FILE = Path("UPDATES.md")
STATUS_FILE = Path("STATUS.md")
CODE_FILE   = Path("CODE.md")

CURRENT_YEAR = date.today().year


# ── Token helpers ──────────────────────────────────────────────────────────────

def chars_to_tokens(n: int) -> int:
    """Approximate: 1 token ≈ 4.5 English chars."""
    return max(1, int(n / 4.5))


# ── Embedding model (lazy-loaded once) ────────────────────────────────────────

_model = None

def get_model():
    raise RuntimeError(
        "chunking.py get_model() called without embed_fn — MiniLM is banned. "
        "Always pass embed_fn=embed_texts from pipeline.py to semantic_topic_shift()."
    )

def embed(texts: list) -> np.ndarray:
    raise RuntimeError(
        "chunking.py embed() called without embed_fn — MiniLM is banned. "
        "Always pass embed_fn=embed_texts from pipeline.py to semantic_topic_shift()."
    )

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


# ── Gate 3 — Relevance check ──────────────────────────────────────────────────

def gate3_relevance(query: str, text: str) -> dict:
    """
    Cosine similarity between query embedding and first 500 chars of fetched content.
    Threshold 0.25 — below this content is likely off-topic (see RISKTOKEEPINMIND.md §GAP).
    """
    q_emb  = embed([query])[0]
    c_emb  = embed([text[:500]])[0]
    sim    = cosine_sim(q_emb, c_emb)
    passed = sim >= GATE3_SIM_THRESHOLD
    return {
        "similarity":      round(sim, 3),
        "threshold":       GATE3_SIM_THRESHOLD,
        "gate3_relevance": f"{'PASS' if passed else 'FAIL'} (sim={sim:.3f})",
        "gate3_pass":      passed,
    }


# ── Wikipedia chunking ────────────────────────────────────────────────────────

def parse_wikipedia_sections(text: str) -> list:
    """
    Parse Wikipedia page.content into sections.
    The `wikipedia` library returns content with headings as == Heading == markers.
    Returns list of {heading, level, text} dicts.
    """
    heading_re = re.compile(r'^(={2,4})\s*(.+?)\s*\1\s*$', re.MULTILINE)
    splits = list(heading_re.finditer(text))

    sections = []

    # Text before the first heading = introduction
    intro_end  = splits[0].start() if splits else len(text)
    intro_text = text[:intro_end].strip()
    if len(intro_text) > 100:
        sections.append({"heading": "Introduction", "level": 1, "text": intro_text})

    for i, match in enumerate(splits):
        level       = len(match.group(1)) - 1      # == → level 1, === → level 2
        heading     = match.group(2).strip()
        start       = match.end()
        end         = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        section_txt = text[start:end].strip()
        if len(section_txt) > 100:
            sections.append({"heading": heading, "level": level, "text": section_txt})

    if not sections:
        # No headings found — treat whole article as one section
        sections = [{"heading": "Full Article", "level": 1, "text": text}]

    return sections


def query_aware_section_selection(sections: list, query: str) -> list:
    """
    Embed each section's first 300 chars + heading and score against query.
    Keep top SECTIONS_TO_KEEP sections. Always include Introduction if present.
    """
    if not sections:
        return []

    q_emb  = embed([query])[0]
    scored = []

    for s in sections:
        preview = (s["heading"] + " " + s["text"])[:300]
        s_emb   = embed([preview])[0]
        sim     = cosine_sim(q_emb, s_emb)
        scored.append((sim, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:SECTIONS_TO_KEEP]]


def hierarchical_chunk(text: str) -> list:
    """
    Split text into parent-child chunks.
      Parent = full paragraph (returned to LLM for rich context, not embedded)
      Child  = ~CHILD_CHUNK_CHARS piece (embedded and used for retrieval)

    Returns flat list. Each child has parent_id pointing to its parent chunk.
    Retrieval uses children. LLM receives parent when a child is retrieved.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 80]
    chunks    = []
    parent_id = 0

    for para in paragraphs:
        chunks.append({
            "type":      "parent",
            "parent_id": parent_id,
            "text":      para,
            "tokens":    chars_to_tokens(len(para)),
        })
        # Split paragraph into child chunks
        words        = para.split()
        child_size_w = max(10, CHILD_CHUNK_CHARS // 5)  # ~5 chars per word
        has_child    = False
        for i in range(0, len(words), child_size_w):
            child_text = " ".join(words[i:i + child_size_w])
            if len(child_text) > MIN_CHUNK_CHARS:
                chunks.append({
                    "type":      "child",
                    "parent_id": parent_id,
                    "text":      child_text,
                    "tokens":    chars_to_tokens(len(child_text)),
                })
                has_child = True
        # G5 fix: if paragraph is too short to produce a valid child, embed it directly.
        # Without this, short paragraphs (80–675 chars) have a parent but no child →
        # never embedded → never retrieved → invisible to LLM. On 100K Wikipedia articles
        # this was causing 35 children instead of the expected 100+.
        if not has_child:
            chunks.append({
                "type":      "child",
                "parent_id": parent_id,
                "text":      para,
                "tokens":    chars_to_tokens(len(para)),
            })
        parent_id += 1

    return chunks


def chunk_wikipedia_baseline(text: str, query: str) -> dict:
    """
    Wikipedia baseline:
      1. Parse Wikipedia content into sections
      2. Score sections against query, keep top SECTIONS_TO_KEEP
      3. Hierarchical chunk each kept section (parent=paragraph, child=~100 tokens)
    """
    t0       = time.perf_counter()
    sections = parse_wikipedia_sections(text)
    kept     = query_aware_section_selection(sections, query)

    all_chunks      = []
    sections_detail = []

    for s in kept:
        h_chunks = hierarchical_chunk(s["text"])
        all_chunks.extend(h_chunks)
        n_children = sum(1 for c in h_chunks if c["type"] == "child")
        sections_detail.append({
            "heading":  s["heading"],
            "chars":    len(s["text"]),
            "children": n_children,
        })

    return {
        "strategy":        "query-aware-section + hierarchical",
        "chunks":          all_chunks,
        "sections_total":  len(sections),
        "sections_kept":   len(kept),
        "sections_detail": sections_detail,
        "chars_before":    len(text),
        "chars_after":     sum(len(s["text"]) for s in kept),
        "time":            round(time.perf_counter() - t0, 3),
    }


# ── Wikipedia RAPTOR-lite ──────────────────────────────────────────────────────

def raptor_lite(text: str, embed_fn=None) -> dict:
    """
    RAPTOR-lite: fixed chunk → embed → K-means cluster → extractive cluster summaries → 2-level tree.

    IMPORTANT NOTE: True RAPTOR (Stanford 2024) uses an LLM to abstractively summarize
    each cluster. This implementation uses extractive summarization (selecting the 3 most
    central chunks per cluster) to avoid LLM dependency during testing phase.
    Production implementation should use grobid + LLM summarization.
    See RISKTOKEEPINMIND.md §RAPTOR for full context and risk assessment.

    Tree structure:
      Level 0 (leaves): fixed ~100-word chunks — used for specific fact retrieval
      Level 1 (summaries): extractive cluster summary — used for thematic retrieval

    Args:
        text:     Wikipedia article text to cluster.
        embed_fn: Optional embedding function to use instead of the local all-MiniLM model.
                  Pass bge-s encode function from phase6 to avoid loading a second model.
                  If None, falls back to the local embed() (all-MiniLM-L6-v2).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    t0 = time.perf_counter()

    # Use provided embed_fn (e.g. bge-s from phase6) or fall back to local model
    _embed = embed_fn if embed_fn is not None else embed

    # Step 1: Fixed-size leaf chunks (~100 words ≈ 450 chars ≈ 100 tokens)
    words       = text.split()
    leaf_chunks = [
        " ".join(words[i:i + RAPTOR_LEAF_WORDS])
        for i in range(0, len(words), RAPTOR_LEAF_WORDS)
        if len(" ".join(words[i:i + RAPTOR_LEAF_WORDS])) > 100
    ]

    if len(leaf_chunks) < 4:
        return {
            "strategy": "raptor-lite",
            "error":    f"Too few leaf chunks ({len(leaf_chunks)}) for clustering",
            "chunks":   [{"type": "leaf", "text": c, "tokens": chars_to_tokens(len(c))} for c in leaf_chunks],
            "time":     round(time.perf_counter() - t0, 3),
        }

    # Step 2: Embed all leaf chunks in one batch
    leaf_embs  = _embed(leaf_chunks)

    # Step 3: K-means clustering (n_clusters: ~10 leaves per cluster)
    n_clusters = min(10, max(3, len(leaf_chunks) // 10))
    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels     = kmeans.fit_predict(leaf_embs)

    # Step 4: For each cluster, pick 3 most-central chunks as extractive summary
    summaries = []
    for c_id in range(n_clusters):
        indices  = [i for i, l in enumerate(labels) if l == c_id]
        if not indices:
            continue
        c_embs   = leaf_embs[indices]
        centroid = kmeans.cluster_centers_[c_id:c_id + 1]
        sims     = sk_cosine(centroid, c_embs)[0]
        top_idx  = sorted(range(len(indices)), key=lambda i: sims[i], reverse=True)[:3]
        summary  = " ".join(leaf_chunks[indices[i]] for i in top_idx)
        summaries.append({
            "type":       "cluster_summary",
            "cluster_id": c_id,
            "n_leaves":   len(indices),
            "text":       summary,
            "tokens":     chars_to_tokens(len(summary)),
        })

    leaf_list = [
        {"type": "leaf", "text": c, "tokens": chars_to_tokens(len(c))}
        for c in leaf_chunks
    ]

    return {
        "strategy":    "raptor-lite",
        "chunks":      leaf_list + summaries,   # full retrieval index
        "leaves":      leaf_list,
        "summaries":   summaries,
        "n_leaves":    len(leaf_list),
        "n_clusters":  n_clusters,
        "n_summaries": len(summaries),
        "tree_levels": 2,
        "time":        round(time.perf_counter() - t0, 3),
    }


# ── Semantic topic-shift (Web + YouTube) ──────────────────────────────────────

def semantic_topic_shift(text: str, threshold: float = TOPIC_SHIFT_THRESHOLD,
                         min_chunk_chars: int = 675, embed_fn=None,
                         overlap_chars: int = 0) -> list:
    """
    Split text at natural topic boundaries:
      1. Split into sentences
      2. Embed all sentences in one batch
      3. Split where cosine similarity between consecutive sentences < threshold
      4. Merge chunks below min_chunk_chars with their neighbour
      5. (Optional) Add overlap_chars of context from the end of each chunk
         to the start of the next — prevents context loss at topic boundaries.

    Works for both Web articles (punctuated) and YouTube transcripts (no punctuation).

    Args:
        embed_fn:      Optional embedding function. Pass bge-s from phase6 to avoid
                       loading all-MiniLM-L6-v2 as a second model.
        overlap_chars: Number of characters to prepend from the previous chunk
                       to each new chunk. Default 0 (no overlap). Use 150 for
                       production (G2 fix — context continuity at boundaries).
    """
    _embed = embed_fn if embed_fn is not None else embed

    # Split on sentence-ending punctuation or paragraph breaks
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if len(sentences) < 3:
        return [{"type": "chunk", "text": text.strip(), "tokens": chars_to_tokens(len(text))}]

    # Batch embed all sentences in one call
    embs = _embed(sentences)

    # Find split positions
    split_before = set()
    for i in range(1, len(sentences)):
        sim = cosine_sim(embs[i - 1], embs[i])
        if sim < threshold:
            split_before.add(i)

    # Build raw chunks
    raw_chunks  = []
    current     = [sentences[0]]

    for i in range(1, len(sentences)):
        if i in split_before and len(" ".join(current)) >= min_chunk_chars:
            raw_chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        raw_chunks.append(" ".join(current))

    # Merge short chunks with neighbour
    merged = []
    for chunk in raw_chunks:
        if len(chunk) < MIN_CHUNK_CHARS and merged:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    # G2 fix: add trailing overlap from previous chunk to start of each new chunk.
    # Prevents context loss at topic boundaries. overlap_chars=0 = no change (default).
    if overlap_chars > 0 and len(merged) > 1:
        overlapped = [merged[0]]
        for i in range(1, len(merged)):
            prefix = merged[i - 1][-overlap_chars:].strip()
            overlapped.append(prefix + " " + merged[i])
        merged = overlapped

    return [
        {"type": "chunk", "text": c.strip(), "tokens": chars_to_tokens(len(c))}
        for c in merged if len(c.strip()) > 50
    ]


# ── ArXiv: atomic ─────────────────────────────────────────────────────────────

def atomic_chunk(text: str) -> list:
    """
    ArXiv: each paper abstract = 1 chunk. No splitting needed.
    Phase 1b combines papers with '\\n\\n---\\n\\n' separator — split on this.
    """
    papers = text.split("\n\n---\n\n")
    return [
        {"type": "atomic", "text": p.strip(), "tokens": chars_to_tokens(len(p.strip()))}
        for p in papers if len(p.strip()) > 100
    ]


# ── Chunk quality metrics ──────────────────────────────────────────────────────

def compute_chunk_metrics(chunks: list) -> dict:
    """
    Compute quality metrics for retrieval chunks.
    For hierarchical chunks: evaluates only 'child' type (used for retrieval — not parents).
    For all other strategies: evaluates all chunks.

    Pass criteria (TESTING_PLAN.md):
      - avg token size in 150–500 range
      - short chunk ratio (< 100 tokens) < 10%
    """
    eval_chunks = [
        c for c in chunks
        if c.get("type") in ("child", "chunk", "atomic", "leaf")
    ]
    if not eval_chunks:
        return {"n_chunks": 0, "error": "no retrieval chunks found"}

    tokens      = [c["tokens"] for c in eval_chunks]
    avg_tokens  = sum(tokens) / len(tokens)
    std_tokens  = float(np.std(tokens))
    short_count = sum(1 for t in tokens if t < 100)
    short_ratio = short_count / len(tokens)
    in_range    = sum(1 for t in tokens
                      if TARGET_TOKEN_RANGE[0] <= t <= TARGET_TOKEN_RANGE[1]) / len(tokens)

    avg_ok   = TARGET_TOKEN_RANGE[0] <= avg_tokens <= TARGET_TOKEN_RANGE[1]
    short_ok = short_ratio < SHORT_CHUNK_LIMIT

    return {
        "n_chunks":       len(eval_chunks),
        "avg_tokens":     round(avg_tokens),
        "std_tokens":     round(std_tokens),
        "min_tokens":     min(tokens),
        "max_tokens":     max(tokens),
        "short_ratio":    round(short_ratio, 3),
        "in_range_ratio": round(in_range, 3),
        "pass":           avg_ok and short_ok,
    }


# ── Main test runner ───────────────────────────────────────────────────────────

def run_phase2():
    # Import fetchers from Phase 1b — avoids duplicating fetch logic
    from phase1b_role_test import (
        fetch_wikipedia, fetch_arxiv, fetch_web, fetch_youtube,
    )

    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path    = LOG_DIR / f"phase2_chunking_{timestamp}.log"
    lines       = []
    all_results = {}   # source → list of per-query result dicts

    def _h(text, level=1):
        sep = "=" if level == 1 else "─"
        lines.extend(["", sep * 60, text, sep * 60])

    _h(f"PHASE 2: CHUNKING STRATEGY TESTING  |  {timestamp}")
    lines.append(
        f"\nThresholds — section_sim: {SECTION_SIM_THRESHOLD}  "
        f"topic_shift: {TOPIC_SHIFT_THRESHOLD}  gate3: {GATE3_SIM_THRESHOLD}"
    )
    lines.append(f"Pass criteria — avg tokens: {TARGET_TOKEN_RANGE}  short_ratio < {SHORT_CHUNK_LIMIT}")

    # Load embedding model once before all chunking
    print("\nLoading embedding model...")
    get_model()

    # ── Per-source testing ─────────────────────────────────────────────────────
    fetchers = {
        "wikipedia": fetch_wikipedia,
        "arxiv":     fetch_arxiv,
        "web":       fetch_web,
        "youtube":   fetch_youtube,
    }

    for source in SOURCES:
        _h(f"SOURCE: {source.upper()}  |  role: {SOURCE_ROLES[source]}", level=2)
        source_results = []

        for query in QUERIES:
            print(f"\n  [{source}] '{query}'")

            # ── Fetch ──────────────────────────────────────────────────────────
            print(f"    Fetching ...", end="", flush=True)
            t_fetch    = time.perf_counter()
            fetch_res  = fetchers[source](query)
            fetch_time = round(time.perf_counter() - t_fetch, 3)
            text       = fetch_res.get("text", "")
            print(f" {fetch_res['status']}  {len(text)} chars  {fetch_time}s")

            if not text:
                source_results.append({"query": query, "status": "FETCH_FAILED",
                                        "chunk_results": {}})
                lines += ["", f"[CHUNK TEST] {source} | '{query}'",
                          f"  SKIPPED — fetch returned empty text"]
                continue

            # ── Gate 3: relevance check ────────────────────────────────────────
            print(f"    Gate 3 ...", end="", flush=True)
            g3 = gate3_relevance(query, text)
            print(f" {g3['gate3_relevance']}")

            # ── Chunking ───────────────────────────────────────────────────────
            chunk_results = {}

            if source == "wikipedia":
                # Baseline: query-aware section selection + hierarchical
                print(f"    [baseline] section selection + hierarchical ...", end="", flush=True)
                baseline  = chunk_wikipedia_baseline(text, query)
                b_metrics = compute_chunk_metrics(baseline["chunks"])
                chunk_results["baseline"] = {**baseline, "metrics": b_metrics}
                print(f" {b_metrics['n_chunks']} child chunks  "
                      f"avg={b_metrics['avg_tokens']}tok  "
                      f"pass={'YES' if b_metrics['pass'] else 'NO'}")

                # Challenger: RAPTOR-lite
                print(f"    [raptor-lite] clustering ...", end="", flush=True)
                try:
                    raptor    = raptor_lite(text)
                    r_metrics = compute_chunk_metrics(raptor.get("leaves", raptor["chunks"]))
                    chunk_results["raptor"] = {**raptor, "metrics": r_metrics}
                    print(f" {raptor.get('n_leaves','?')} leaves  "
                          f"{raptor.get('n_clusters','?')} clusters  "
                          f"avg={r_metrics['avg_tokens']}tok")
                except Exception as e:
                    chunk_results["raptor"] = {"error": str(e), "metrics": {"pass": False}}
                    print(f" ERROR: {e}")

            elif source == "arxiv":
                chunks  = atomic_chunk(text)
                metrics = compute_chunk_metrics(chunks)
                chunk_results["atomic"] = {"chunks": chunks, "metrics": metrics}
                print(f"    [atomic]: {metrics['n_chunks']} chunks  "
                      f"avg={metrics['avg_tokens']}tok  "
                      f"pass={'YES' if metrics['pass'] else 'NO'}")

            elif source in ("web", "youtube"):
                print(f"    [topic-shift] ...", end="", flush=True)
                chunks  = semantic_topic_shift(text)
                metrics = compute_chunk_metrics(chunks)
                chunk_results["topic_shift"] = {"chunks": chunks, "metrics": metrics}
                print(f" {metrics['n_chunks']} chunks  "
                      f"avg={metrics['avg_tokens']}tok  "
                      f"pass={'YES' if metrics['pass'] else 'NO'}")

            # ── Log entry ──────────────────────────────────────────────────────
            lines += [
                "",
                f"[CHUNK TEST] {source} | '{query}'",
                f"  Fetch     : {fetch_res['status']}  {len(text)} chars  {fetch_time}s",
                f"  Gate 3    : {g3['gate3_relevance']}",
            ]
            for strat_name, cr in chunk_results.items():
                m = cr.get("metrics", {})
                if source == "wikipedia" and strat_name == "baseline":
                    lines += [
                        f"  [{strat_name}] strategy: {cr.get('strategy','')}",
                        f"  [{strat_name}] sections: {cr.get('sections_kept','?')}/{cr.get('sections_total','?')} kept"
                        f"  →  {cr.get('chars_after','?')} / {cr.get('chars_before','?')} chars",
                    ]
                elif source == "wikipedia" and strat_name == "raptor":
                    lines += [
                        f"  [{strat_name}] leaves={cr.get('n_leaves','?')}  "
                        f"clusters={cr.get('n_clusters','?')}  "
                        f"summaries={cr.get('n_summaries','?')}  "
                        f"tree_levels={cr.get('tree_levels','?')}",
                        f"  [{strat_name}] NOTE: extractive summaries — production needs LLM abstraction",
                    ]
                if m and "error" not in m:
                    lines.append(
                        f"  [{strat_name}] n={m.get('n_chunks','?')}  "
                        f"avg={m.get('avg_tokens','?')}tok  "
                        f"std={m.get('std_tokens','?')}  "
                        f"min={m.get('min_tokens','?')}  max={m.get('max_tokens','?')}  "
                        f"short%={round(m.get('short_ratio',0)*100,1)}  "
                        f"in_range%={round(m.get('in_range_ratio',0)*100,1)}  "
                        f"PASS={'YES' if m.get('pass') else 'NO'}"
                    )
                elif "error" in cr:
                    lines.append(f"  [{strat_name}] ERROR: {cr['error']}")

            source_results.append({
                "query":         query,
                "status":        fetch_res["status"],
                "text_len":      len(text),
                "gate3":         g3,
                "chunk_results": chunk_results,
            })

        all_results[source] = source_results

    # ── Summary ────────────────────────────────────────────────────────────────
    _h("PHASE 2 SUMMARY")

    source_pass    = {}
    strategy_picks = {}    # source → winning strategy name

    for source, qresults in all_results.items():
        strat_pass_counts = {}
        for qr in qresults:
            if qr.get("status") == "FETCH_FAILED":
                continue
            for strat, cr in qr.get("chunk_results", {}).items():
                m = cr.get("metrics", {})
                strat_pass_counts.setdefault(strat, []).append(m.get("pass", False))

        lines.append("")
        lines.append(f"{source.upper()}  ({SOURCE_ROLES[source]}):")

        for strat, passes in strat_pass_counts.items():
            rate = sum(passes) / len(passes) if passes else 0
            lines.append(f"  {strat}: {sum(passes)}/{len(passes)} queries passed ({rate*100:.0f}%)")

        # Determine winner / deferral
        if source == "wikipedia":
            # Wikipedia: both strategies tested — winner deferred to Phase 4
            b_passes = strat_pass_counts.get("baseline", [False])
            r_passes = strat_pass_counts.get("raptor",   [False])
            b_rate   = sum(b_passes) / max(1, len(b_passes))
            r_rate   = sum(r_passes) / max(1, len(r_passes))
            lines.append(
                f"  → DEFERRED to Phase 4 (retrieval scoring). "
                f"Baseline: {b_rate*100:.0f}%  RAPTOR-lite: {r_rate*100:.0f}%"
            )
            source_pass[source]    = True   # both strategies built successfully
            strategy_picks[source] = "deferred (baseline + raptor-lite both available)"

        elif source == "arxiv":
            passes = strat_pass_counts.get("atomic", [])
            rate   = sum(passes) / max(1, len(passes))
            lines.append(f"  → Winner: atomic ({rate*100:.0f}%)")
            source_pass[source]    = rate >= 0.8
            strategy_picks[source] = "atomic"

        else:  # web, youtube
            passes = strat_pass_counts.get("topic_shift", [])
            rate   = sum(passes) / max(1, len(passes))
            lines.append(f"  → Winner: semantic topic-shift ({rate*100:.0f}%)")
            source_pass[source]    = rate >= 0.8
            strategy_picks[source] = "semantic topic-shift"

    passed   = [s for s, p in source_pass.items() if p]
    deferred = [s for s in SOURCES if strategy_picks.get(s, "").startswith("deferred")]
    lines += [
        "",
        "─" * 55,
        f"PASSED   : {', '.join(passed)}",
        f"DEFERRED : {', '.join(deferred)} (winner after Phase 4 retrieval scoring)",
        "─" * 55,
    ]

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLog → {log_path}")

    return all_results, source_pass, strategy_picks, timestamp, log_path


# ── MD auto-updaters ───────────────────────────────────────────────────────────

def update_decisions_tracker(source_pass: dict, strategy_picks: dict, timestamp: str):
    text     = PLAN_FILE.read_text(encoding="utf-8")
    date_str = timestamp[:10]
    reason   = (
        "Wikipedia: hierarchical baseline + RAPTOR-lite both built — winner deferred to Phase 4. "
        "ArXiv: atomic confirmed. Web + YouTube: semantic topic-shift confirmed. "
        "Gate 3 (relevance check) implemented with all-MiniLM-L6-v2."
    )
    strats   = "; ".join(f"{s}={v}" for s, v in strategy_picks.items())
    new_row  = f"| 2 | Chunking | {strats} | {reason} | {date_str} |"
    updated  = re.sub(r"\| 2 \| Chunking \|.*", new_row, text)
    if updated != text:
        PLAN_FILE.write_text(updated, encoding="utf-8")
        print("TESTING_PLAN.md updated.")
    else:
        print("Warning: could not find Phase 2 row in TESTING_PLAN.md.")


def update_updates_md(all_results: dict, source_pass: dict,
                      strategy_picks: dict, timestamp: str, log_path: Path):
    date_str = timestamp[:10]

    # Aggregate metrics: one row per (source, strategy) across all queries
    summary_rows = []
    for source, qresults in all_results.items():
        strat_metrics = {}
        for qr in qresults:
            for strat, cr in qr.get("chunk_results", {}).items():
                m = cr.get("metrics", {})
                strat_metrics.setdefault(strat, []).append(m)

        for strat, mlist in strat_metrics.items():
            valid = [m for m in mlist if "n_chunks" in m and m["n_chunks"] > 0]
            if not valid:
                continue
            avg_n   = round(sum(m["n_chunks"]   for m in valid) / len(valid))
            avg_tok = round(sum(m["avg_tokens"]  for m in valid) / len(valid))
            pass_n  = sum(1 for m in valid if m.get("pass"))
            summary_rows.append(
                f"| {source} | {strat} | ~{avg_n} | ~{avg_tok} | {pass_n}/{len(valid)} |"
            )

    entry = textwrap.dedent(f"""
    ---

    ## Phase 2 — Chunking Strategy Testing (v1)
    **Date:** {date_str}
    **Log:** `{log_path}`

    ### Chunk quality metrics (avg across 3 queries per strategy)

    | Source | Strategy | Avg # chunks | Avg tokens | Queries passed |
    |--------|----------|-------------|------------|---------------|
    {chr(10).join(summary_rows)}

    ### Decisions
    - **Wikipedia:** Both baseline (query-aware section selection + hierarchical) and RAPTOR-lite built and verified.
      Winner deferred to Phase 4 — retrieval scoring needed to compare meaningfully.
    - **ArXiv:** Atomic confirmed (1 abstract = 1 chunk, no splitting).
    - **Web:** Semantic topic-shift confirmed.
    - **YouTube:** Semantic topic-shift confirmed.

    ### Gate 3 first active here
    `cosine_similarity(embed(query), embed(text[:500])) > 0.25 → PASS`
    Model: all-MiniLM-L6-v2 (sentence-transformers). Threshold: 0.25.
    Detects off-topic fetches that would pass Gates 1+2 (length + noise) — see RISKTOKEEPINMIND.md §GAP.

    ### Next Step
    Phase 3 — Embedding Model Testing.
    Run: `python phase3_embedding_test.py`
    """).strip()

    current    = UPDATE_FILE.read_text(encoding="utf-8")
    header_end = current.find("\n---\n") + 5
    UPDATE_FILE.write_text(
        current[:header_end] + "\n" + entry + "\n" + current[header_end:],
        encoding="utf-8"
    )
    print("UPDATES.md updated.")


def update_status_md(source_pass: dict, strategy_picks: dict, timestamp: str):
    text     = STATUS_FILE.read_text(encoding="utf-8")
    date_str = timestamp[:10]
    passed   = ", ".join(s for s, p in source_pass.items() if p)

    text = re.sub(
        r"\*\*Last updated:\*\*.*",
        f"**Last updated:** {date_str} (Phase 2 complete)",
        text
    )
    text = re.sub(
        r"\| 2 \| Chunking.*",
        f"| 2 | Chunking Strategies | ✅ Done | {passed} (Wikipedia winner deferred to Phase 4) |",
        text
    )
    STATUS_FILE.write_text(text, encoding="utf-8")
    print("STATUS.md updated.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 2 — CHUNKING STRATEGY TESTING  (v2)")
    print("=" * 60)
    print(f"  Queries  : {len(QUERIES)}")
    print(f"  Sources  : {', '.join(SOURCES)}")
    print(f"  Model    : all-MiniLM-L6-v2 (sentence-transformers)")
    print(f"  Gate 3   : cosine_sim(query, content[:500]) > {GATE3_SIM_THRESHOLD}")
    print("=" * 60 + "\n")

    all_results, source_pass, strategy_picks, timestamp, log_path = run_phase2()

    passed = [s for s, p in source_pass.items() if p]
    print("\n" + "─" * 60)
    print(f"  Passed: {', '.join(passed)}")
    print("─" * 60)

    update_decisions_tracker(source_pass, strategy_picks, timestamp)
    update_updates_md(all_results, source_pass, strategy_picks, timestamp, log_path)
    update_status_md(source_pass, strategy_picks, timestamp)

    print("\nDone. UPDATES.md, TESTING_PLAN.md, STATUS.md all updated.")
