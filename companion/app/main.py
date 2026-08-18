"""
Buddy — a pink AI companion app.

Standalone FastAPI app, sibling to the main Nexus-AI research engine (`app/`).
Reuses the same Gemini setup (GEMINI_API_KEY, google-generativeai, gemini-2.5-flash)
and the same fetch()+ReadableStream SSE chat pattern as app/main.py, but is a
completely separate product — run it as its own service (see render.yaml).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent          # companion/app
REPO_ROOT = APP_DIR.parent.parent        # Nexus-AI/
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env", override=True)

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "[startup] GEMINI_API_KEY not set. Add it to the repo-root .env file."
    )

from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import llm
import memory
import storage

STATIC_DIR = APP_DIR / "static"
PASSCODE = os.getenv("COMPANION_PASSCODE", "").strip()

_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="Buddy", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_PUBLIC_PATHS = {"/api/lock-status", "/api/unlock", "/api/health"}


@app.middleware("http")
async def passcode_gate(request: Request, call_next):
    """Optional lock screen for public deploys. No-op if COMPANION_PASSCODE unset."""
    if PASSCODE and request.url.path.startswith("/api") and request.url.path not in _PUBLIC_PATHS:
        if request.headers.get("x-companion-passcode", "") != PASSCODE:
            return JSONResponse({"error": "locked"}, status_code=401)
    return await call_next(request)


@app.on_event("startup")
async def startup():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, llm.load_llm)
    memory.ensure_profile(storage.load_config().get("her_name", "her"))


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/lock-status")
async def lock_status():
    return {"locked": bool(PASSCODE)}


@app.post("/api/unlock")
async def unlock(body: dict):
    ok = bool(PASSCODE) and body.get("passcode", "") == PASSCODE
    return JSONResponse({"ok": ok}, status_code=200 if ok else 401)


# ── Onboarding ──────────────────────────────────────────────────────────────────

@app.get("/api/onboarding")
async def get_onboarding():
    config = storage.load_config()
    return {
        "onboarded": bool(config.get("onboarded")),
        "her_name": config.get("her_name", ""),
        "buddy_name": config.get("buddy_name", ""),
    }


class OnboardingRequest(BaseModel):
    her_name: str
    buddy_name: str = "Buddy"


@app.post("/api/onboarding")
async def post_onboarding(body: OnboardingRequest):
    if not body.her_name.strip():
        return JSONResponse({"error": "her_name is required"}, status_code=400)
    config = storage.save_config(body.her_name, body.buddy_name)
    memory.ensure_profile(config["her_name"])
    memory.set_profile_name(config["her_name"])
    return config


# ── Chat (SSE) ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


def _tasks_summary() -> str:
    open_tasks = [t["text"] for t in storage.load_tasks() if not t.get("done")]
    return "\n".join(f"- {t}" for t in open_tasks[:8])


def _mood_summary() -> str:
    entries = storage.load_mood_log()[-3:]
    return "\n".join(f"- {e['date']}: {e['mood']} {e.get('note', '')}".strip() for e in entries)


@app.post("/api/chat")
async def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    if len(message) > 2000:
        return JSONResponse({"error": "Message too long"}, status_code=413)

    config = storage.load_config()
    if not config.get("onboarded"):
        return JSONResponse({"error": "Not onboarded yet"}, status_code=400)

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def emit(event_type: str, **kwargs):
        asyncio.run_coroutine_threadsafe(q.put({"type": event_type, **kwargs}), loop)

    def run_chat():
        try:
            profile_text = memory.read_profile_text()
            history = storage.recent_chat_context()
            prompt = llm.build_chat_prompt(
                config, profile_text, history, _tasks_summary(), _mood_summary(), message
            )

            full_reply = ""
            for event in llm.stream_reply(prompt):
                if event["type"] == "token":
                    full_reply += event["text"]
                    emit("token", text=event["text"])
                elif event["type"] == "retry":
                    full_reply = ""
                    emit("retry", wait=event["wait"])
                elif event["type"] == "error":
                    emit("error", message=event["message"])
                    return

            if not full_reply.strip():
                emit("error", message="Buddy didn't say anything back — try again?")
                return

            storage.append_chat_turn("user", message)
            storage.append_chat_turn("buddy", full_reply)

            facts = llm.extract_memory_facts(config, message, full_reply)
            memory.merge_facts(config.get("her_name", "her"), facts)

            emit("done", full_text=full_reply)
        except Exception as exc:
            emit("error", message=str(exc))
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    _executor.submit(run_chat)

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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/api/chat/history")
async def chat_history():
    return storage.load_chat_history()


# ── Profile / memory ──────────────────────────────────────────────────────────

@app.get("/api/profile")
async def get_profile():
    return {"text": memory.read_profile_text()}


class ProfileUpdate(BaseModel):
    text: str


@app.put("/api/profile")
async def put_profile(body: ProfileUpdate):
    memory.write_profile_text(body.text)
    return {"ok": True}


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.get("/api/tasks")
async def get_tasks():
    return storage.load_tasks()


class TaskCreateRequest(BaseModel):
    text: str


@app.post("/api/tasks")
async def create_task(body: TaskCreateRequest):
    if not body.text.strip():
        return JSONResponse({"error": "text is required"}, status_code=400)
    return storage.add_task(body.text)


class TaskUpdateRequest(BaseModel):
    text: str | None = None
    done: bool | None = None


@app.patch("/api/tasks/{task_id}")
async def patch_task(task_id: str, body: TaskUpdateRequest):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    task = storage.update_task(task_id, **fields)
    if not task:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return task


@app.delete("/api/tasks/{task_id}")
async def remove_task(task_id: str):
    ok = storage.delete_task(task_id)
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return {"deleted": task_id}


class BreakdownRequest(BaseModel):
    goal: str


@app.post("/api/tasks/breakdown")
async def breakdown(body: BreakdownRequest):
    if not body.goal.strip():
        return JSONResponse({"error": "goal is required"}, status_code=400)
    her_name = storage.load_config().get("her_name", "her")
    loop = asyncio.get_running_loop()
    subtasks = await loop.run_in_executor(None, llm.breakdown_goal, her_name, body.goal)
    if not subtasks:
        return JSONResponse({"error": "Couldn't break that down — try rephrasing the goal"}, status_code=502)
    created = storage.add_tasks_bulk(subtasks, source="ai")
    return {"tasks": created}


# ── Mood / check-ins ────────────────────────────────────────────────────────────

@app.get("/api/mood")
async def get_mood():
    return {"entries": storage.load_mood_log()[-30:], "streak": storage.compute_streak()}


class MoodRequest(BaseModel):
    mood: str
    note: str = ""


@app.post("/api/mood")
async def post_mood(body: MoodRequest):
    if not body.mood.strip():
        return JSONResponse({"error": "mood is required"}, status_code=400)
    entry = storage.append_mood(body.mood, body.note)
    return {"entry": entry, "streak": storage.compute_streak()}


# ── Daily affirmation ────────────────────────────────────────────────────────────

@app.get("/api/affirmation")
async def get_affirmation():
    cached = storage.get_cached_affirmation()
    if cached:
        return {"text": cached, "cached": True}
    config = storage.load_config()
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(
        None, llm.generate_affirmation,
        config.get("her_name", "her"), config.get("buddy_name", "Buddy"),
    )
    storage.set_cached_affirmation(text)
    return {"text": text, "cached": False}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8090))
    print(f"\n  Buddy running at http://localhost:{port}\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
