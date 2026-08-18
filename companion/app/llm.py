"""
Gemini 2.5 Flash client — same model/library/env var as the main Nexus-AI
research engine (`google-generativeai`, `GEMINI_API_KEY`, gemini-2.5-flash),
just used here for a warm best-friend persona instead of research synthesis.
"""

import json
import os
import re
import time

LLM_MODEL = "gemini-2.5-flash"
RETRY_DELAYS = [5, 15, 45]  # same backoff schedule as app/pipeline.py

_llm = None


def load_llm():
    global _llm
    if _llm is None:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Add it to the repo-root .env file."
            )
        genai.configure(api_key=api_key)
        _llm = genai.GenerativeModel(LLM_MODEL)
    return _llm


def _retryable(err: Exception) -> bool:
    s = str(err)
    return "429" in s or "503" in s or "UNAVAILABLE" in s


# ── Persona + prompt building ──────────────────────────────────────────────────

PERSONA = """You are {buddy_name}, {her_name}'s AI best friend living inside her personal companion app.

Personality: warm, articulate, genuinely curious about her life, endlessly supportive — like her closest friend, not a customer-service assistant. Casual but well-composed tone, contractions, no emoji, short paragraphs. You celebrate her wins, gently check in when she seems stressed, and never lecture.

Your job day-to-day: help her learn things, plan and break down tasks, explain concepts simply, and just be there to talk to. When she asks something you don't know for certain, say so honestly rather than making it up.

Here is what you remember about her so far (from profile.md — keep using and building on this, don't ask her to repeat things you already know):
{profile_text}
{extra_context}
Stay in character as {buddy_name}. Reply directly to her latest message below — no meta-commentary about being an AI, no "As an AI language model" disclaimers."""


def build_chat_prompt(config: dict, profile_text: str, chat_history: list,
                       tasks_summary: str, mood_summary: str, message: str) -> str:
    her_name = config.get("her_name", "her")
    buddy_name = config.get("buddy_name", "Buddy")

    extras = []
    if tasks_summary:
        extras.append(f"What she's currently working on:\n{tasks_summary}")
    if mood_summary:
        extras.append(f"Recent mood check-ins:\n{mood_summary}")
    extra_context = ("\n" + "\n\n".join(extras) + "\n") if extras else ""

    persona = PERSONA.format(
        buddy_name=buddy_name, her_name=her_name,
        profile_text=profile_text.strip(), extra_context=extra_context,
    )

    convo = []
    for turn in chat_history:
        speaker = her_name if turn["role"] == "user" else buddy_name
        convo.append(f"{speaker}: {turn['text']}")
    convo.append(f"{her_name}: {message}")
    convo.append(f"{buddy_name}:")

    return persona + "\n\n" + "\n".join(convo)


def stream_reply(prompt: str):
    """Yields text tokens as they arrive. Retries on transient Gemini errors
    the same way app/main.py does — the caller resets its buffer on retry."""
    model = load_llm()
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                token = getattr(chunk, "text", None)
                if token:
                    yield {"type": "token", "text": token}
            return
        except Exception as e:
            if _retryable(e) and attempt < len(RETRY_DELAYS):
                wait_s = RETRY_DELAYS[attempt]
                yield {"type": "retry", "wait": wait_s}
                time.sleep(wait_s)
            else:
                yield {"type": "error", "message": str(e)}
                return


def _call_json(prompt: str, fallback: dict) -> dict:
    """Shared helper for the small structured (non-streaming) Gemini calls below."""
    try:
        model = load_llm()
        response = model.generate_content(
            prompt, generation_config={"response_mime_type": "application/json"}
        )
        raw = (response.text or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return fallback
        return json.loads(match.group())
    except Exception:
        return fallback


# ── Memory extraction (the "evolving buddy" step) ──────────────────────────────

MEMORY_EXTRACT_PROMPT = """You just had this exchange with {her_name} inside her companion app:

{her_name}: {user_message}
{buddy_name}: {buddy_reply}

Extract only genuinely new, durable facts worth remembering long-term about her — not small talk, not things already obvious from context. Each list should contain short bullet-style phrases (no more than ~10 words each). Leave a list empty if nothing new fits it.

Return ONLY this JSON, nothing else:
{{
  "about": [],
  "interests": [],
  "goals": [],
  "wins": [],
  "remember": []
}}

Field meaning:
- about: durable facts about who she is (job, school, family, etc.)
- interests: things she likes/enjoys
- goals: things she's learning or working toward
- wins: things she accomplished or is proud of
- remember: anything else worth Buddy recalling later (preferences, important dates, sensitivities)"""


def extract_memory_facts(config: dict, user_message: str, buddy_reply: str) -> dict:
    fallback = {"about": [], "interests": [], "goals": [], "wins": [], "remember": []}
    prompt = MEMORY_EXTRACT_PROMPT.format(
        her_name=config.get("her_name", "her"),
        buddy_name=config.get("buddy_name", "Buddy"),
        user_message=user_message, buddy_reply=buddy_reply,
    )
    return _call_json(prompt, fallback)


# ── Task breakdown ──────────────────────────────────────────────────────────────

def breakdown_goal(her_name: str, goal_text: str) -> list:
    prompt = f"""{her_name} wants help breaking a goal into small, concrete day-to-day learning tasks.

Goal: {goal_text}

Break it into 3-6 small, specific, doable tasks (each a short actionable phrase, no numbering). Order them sensibly.

Return ONLY this JSON: {{"tasks": ["...", "..."]}}"""
    result = _call_json(prompt, {"tasks": []})
    return [t for t in result.get("tasks", []) if isinstance(t, str) and t.strip()]


# ── Daily affirmation ────────────────────────────────────────────────────────────

def generate_affirmation(her_name: str, buddy_name: str) -> str:
    prompt = (
        f"Write one short, warm, specific note from {buddy_name} to {her_name} to start her day. "
        f"1-2 sentences, genuine and well-composed, not cheesy greeting-card language, no emoji, "
        f"no quotation marks around it. Plain text only, no JSON."
    )
    try:
        model = load_llm()
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        return text or f"Whatever today brings, {her_name}, I'm rooting for you."
    except Exception:
        return f"Whatever today brings, {her_name}, I'm rooting for you."
