"""
profile.md — Buddy's long-term memory of her.

Plain Markdown, five fixed sections. Read in full and injected into the chat
system prompt on every turn. After each exchange, llm.py extracts a small
set of new facts and merge_facts() folds them into the right section here —
this is what makes Buddy "evolve" instead of forgetting everything on reload.
"""

import re
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PROFILE_PATH = DATA_DIR / "profile.md"

# machine key -> heading text, in the order they render
SECTIONS = {
    "about":     "About Her",
    "interests": "Interests & Favorites",
    "goals":     "Current Learning Goals",
    "wins":      "Recent Wins & Notes",
    "remember":  "Things Buddy Should Remember",
}


def ensure_profile(her_name: str = "her") -> None:
    """Create profile.md from a blank template if it doesn't exist yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PROFILE_PATH.exists():
        return
    sections = {key: [] for key in SECTIONS}
    PROFILE_PATH.write_text(render_profile(her_name, sections), encoding="utf-8")


def read_profile_text() -> str:
    """Raw markdown, for injecting into the LLM prompt and for the Memory tab."""
    if not PROFILE_PATH.exists():
        ensure_profile()
    return PROFILE_PATH.read_text(encoding="utf-8")


def write_profile_text(text: str) -> None:
    """Manual overwrite — used when she edits the Memory tab by hand."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(text, encoding="utf-8")


def parse_sections(text: str) -> dict:
    """Markdown -> {machine_key: [bullet, bullet, ...]}."""
    sections = {key: [] for key in SECTIONS}
    heading_to_key = {v: k for k, v in SECTIONS.items()}
    current = None
    for line in text.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current = heading_to_key.get(heading.group(1))
            continue
        bullet = re.match(r"^-\s+(.+?)\s*$", line)
        if bullet and current:
            sections[current].append(bullet.group(1))
    return sections


def render_profile(her_name: str, sections: dict) -> str:
    lines = [f"# {her_name}'s Profile", f"_Last updated: {date.today().isoformat()}_", ""]
    for key, heading in SECTIONS.items():
        lines.append(f"## {heading}")
        bullets = sections.get(key) or []
        if bullets:
            lines.extend(f"- {b}" for b in bullets)
        else:
            lines.append("_Nothing yet — this fills in as Buddy gets to know her._")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def set_profile_name(her_name: str) -> None:
    """Rename the '# X's Profile' title line in place, keeping all sections intact.
    Used when onboarding completes — startup may have already created the file
    with a generic placeholder name before she'd entered her real one."""
    text = read_profile_text()
    sections = parse_sections(text)
    write_profile_text(render_profile(her_name, sections))


def _is_duplicate(new_fact: str, existing: list) -> bool:
    new_norm = new_fact.strip().lower()
    for item in existing:
        existing_norm = item.strip().lower()
        if new_norm == existing_norm or new_norm in existing_norm or existing_norm in new_norm:
            return True
    return False


def merge_facts(her_name: str, facts: dict, max_per_section: int = 12) -> bool:
    """
    Fold newly-extracted facts (machine keys from SECTIONS) into profile.md.
    Skips near-duplicates, caps each section so the prompt doesn't grow forever
    (oldest bullets drop off first). Returns True if anything changed.
    """
    text = read_profile_text()
    sections = parse_sections(text)
    changed = False

    for key, new_items in facts.items():
        if key not in SECTIONS or not new_items:
            continue
        bucket = sections.setdefault(key, [])
        for fact in new_items:
            fact = (fact or "").strip().strip("-").strip()
            if not fact or _is_duplicate(fact, bucket):
                continue
            bucket.append(fact)
            changed = True
        if len(bucket) > max_per_section:
            sections[key] = bucket[-max_per_section:]

    if changed:
        write_profile_text(render_profile(her_name, sections))
    return changed
