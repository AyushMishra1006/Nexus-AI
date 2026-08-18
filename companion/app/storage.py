"""
Plain-file storage for everything except profile.md (see memory.py).
Single-user personal app — JSON/JSONL files under data/, no DB needed.
"""

import json
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CONFIG_PATH = DATA_DIR / "config.json"
TASKS_PATH = DATA_DIR / "tasks.json"
MOOD_LOG_PATH = DATA_DIR / "mood_log.jsonl"
CHAT_HISTORY_PATH = DATA_DIR / "chat_history.json"
AFFIRMATION_CACHE_PATH = DATA_DIR / "affirmation_cache.json"

MAX_CHAT_HISTORY = 60   # stored/displayed
CHAT_CONTEXT_TURNS = 8  # fed into the prompt


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, data) -> None:
    _ensure_dir()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Config / onboarding ──────────────────────────────────────────────────────

def load_config() -> dict:
    return _read_json(CONFIG_PATH, {})


def save_config(her_name: str, buddy_name: str) -> dict:
    config = {
        "her_name": her_name.strip(),
        "buddy_name": buddy_name.strip() or "Buddy",
        "onboarded": True,
        "created_at": datetime.now().isoformat(),
    }
    _write_json(CONFIG_PATH, config)
    return config


def is_onboarded() -> bool:
    return bool(load_config().get("onboarded"))


# ── Tasks ─────────────────────────────────────────────────────────────────────

def load_tasks() -> list:
    return _read_json(TASKS_PATH, [])


def save_tasks(tasks: list) -> None:
    _write_json(TASKS_PATH, tasks)


def add_task(text: str, source: str = "manual") -> dict:
    tasks = load_tasks()
    task = {
        "id": uuid.uuid4().hex[:8],
        "text": text.strip(),
        "done": False,
        "source": source,
        "created_at": datetime.now().isoformat(),
    }
    tasks.append(task)
    save_tasks(tasks)
    return task


def add_tasks_bulk(texts: list, source: str = "ai") -> list:
    tasks = load_tasks()
    created = []
    for text in texts:
        text = (text or "").strip().strip("-").strip()
        if not text:
            continue
        task = {
            "id": uuid.uuid4().hex[:8],
            "text": text,
            "done": False,
            "source": source,
            "created_at": datetime.now().isoformat(),
        }
        tasks.append(task)
        created.append(task)
    if created:
        save_tasks(tasks)
    return created


def update_task(task_id: str, **fields) -> dict | None:
    tasks = load_tasks()
    for task in tasks:
        if task["id"] == task_id:
            task.update(fields)
            save_tasks(tasks)
            return task
    return None


def delete_task(task_id: str) -> bool:
    tasks = load_tasks()
    remaining = [t for t in tasks if t["id"] != task_id]
    if len(remaining) == len(tasks):
        return False
    save_tasks(remaining)
    return True


# ── Mood / check-ins ──────────────────────────────────────────────────────────

def load_mood_log() -> list:
    if not MOOD_LOG_PATH.exists():
        return []
    entries = []
    for line in MOOD_LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def append_mood(mood: str, note: str = "") -> dict:
    _ensure_dir()
    entry = {
        "date": date.today().isoformat(),
        "mood": mood,
        "note": note.strip(),
        "ts": datetime.now().isoformat(),
    }
    with MOOD_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def compute_streak() -> int:
    """Consecutive check-in days ending today or yesterday (grace of 1 day)."""
    days = {e["date"] for e in load_mood_log()}
    if not days:
        return 0
    today = date.today()
    if today.isoformat() not in days and (today - timedelta(days=1)).isoformat() not in days:
        return 0
    streak = 0
    cursor = today if today.isoformat() in days else today - timedelta(days=1)
    while cursor.isoformat() in days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


# ── Chat history ──────────────────────────────────────────────────────────────

def load_chat_history() -> list:
    return _read_json(CHAT_HISTORY_PATH, [])


def append_chat_turn(role: str, text: str) -> None:
    history = load_chat_history()
    history.append({"role": role, "text": text, "ts": datetime.now().isoformat()})
    if len(history) > MAX_CHAT_HISTORY:
        history = history[-MAX_CHAT_HISTORY:]
    _write_json(CHAT_HISTORY_PATH, history)


def recent_chat_context(turns: int = CHAT_CONTEXT_TURNS) -> list:
    return load_chat_history()[-turns:]


# ── Daily affirmation cache ────────────────────────────────────────────────────

def get_cached_affirmation() -> str | None:
    cache = _read_json(AFFIRMATION_CACHE_PATH, {})
    if cache.get("date") == date.today().isoformat():
        return cache.get("text")
    return None


def set_cached_affirmation(text: str) -> None:
    _write_json(AFFIRMATION_CACHE_PATH, {"date": date.today().isoformat(), "text": text})
