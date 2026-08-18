# Buddy 💗

A pink AI companion app — a little gift project. A chatty best-friend buddy
(Gemini 2.5 Flash) that helps with day-to-day learning and tasks, and slowly
gets to know her by writing what it learns into `companion/app/data/profile.md`
as you chat.

Reuses the exact Gemini setup already in this repo's research engine
(`google-generativeai`, `GEMINI_API_KEY`, `gemini-2.5-flash`) — no extra
dependencies beyond the root `requirements.txt`.

## Features

- **Chat** — a warm, casual AI best friend, streamed live as it types.
- **Tasks** — a to-do list for small day-to-day learning goals, with an
  "help me break this down 💡" button that asks Buddy to split a bigger goal
  into small steps.
- **Check-in** — a daily mood check-in with a streak counter.
- **Memory jar** — everything Buddy has picked up about her over time,
  rendered from `profile.md`, editable by hand.
- **Home** — a fresh daily affirmation, generated once per day.

Buddy re-reads `profile.md` (plus recent tasks and mood) on every chat turn,
and after each exchange asks Gemini to extract any new durable facts worth
remembering — those get folded straight into `profile.md`. That's what makes
it "evolve" instead of forgetting everything on reload.

## Run locally

From the repo root, with `GEMINI_API_KEY` set in the root `.env`:

```bash
pip install -r requirements.txt
uvicorn companion.app.main:app --host 0.0.0.0 --port 8090
```

Open `http://localhost:8090`. First visit asks for her name and a name for
Buddy, then you're in.

All personal data (profile, tasks, mood log, chat history) lives in
`companion/app/data/` and is gitignored — it never gets committed.

## Optional lock screen

If this gets deployed somewhere public (e.g. Render), set `COMPANION_PASSCODE`
in the environment to gate the app behind a simple passphrase screen ("our
secret door"). Leave it unset for local/private use — the lock screen is
skipped entirely.

## Deployment

`render.yaml` at the repo root defines this as its own service
(`companion-buddy`), separate from the research engine — both can be deployed
from the same repo. Set `GEMINI_API_KEY` (and optionally `COMPANION_PASSCODE`)
in the Render dashboard for that service.
