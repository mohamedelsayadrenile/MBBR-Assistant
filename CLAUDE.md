# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Voice assistant for MBBR wastewater-plant operators speaking Egyptian Arabic:
`audio → ASR → agent (LLM + tools) → MBBR API → TTS → audio`. README.md has the full API contract and design notes.

## Commands

```bash
uv sync
docker run -d --name mbbr-redis -p 6379:6379 redis:7-alpine   # Redis is required at startup
uv run uvicorn main:app --reload          # run from repo root; module is main:app, not src.main:app
uv run streamlit run streamlit_app.py     # manual voice/text tester, talks to the HTTP API only
docker compose up -d --build              # shipped setup: Redis + API on one NVIDIA GPU (needs NVIDIA Container Toolkit)

uv run pytest                                          # full suite
uv run pytest tests/test_agent.py                      # one file
uv run pytest tests/test_agent.py::test_name           # one test
uv run python -m compileall src                        # syntax check; no linter is configured
```

`pyproject.toml` sets `pythonpath = ["src", "."]` and `asyncio_mode = "auto"`, so imports are `from services.x import ...` (no `src.` prefix) and async tests need no marker.

## Architecture

- `src/main.py` lifespan builds everything once and stores it on `app.state`: Redis (pinged), `RedisMemory`, ASR and TTS providers (models loaded **sequentially** at startup to avoid stacking GPU peak memory), the voice list, and `ChatService`. Endpoints read from `app.state`; there is no DI container.
- `services/chat_service.py` runs one turn: load memory → `MBBRAgent.run` → append user + assistant text → optional TTS. `LLMError`, `MBBRAPIError`, `RedisError` become a spoken Arabic apology with HTTP 200; `TTSError` degrades to a text-only reply.
- `agent/agent.py` is a plain tool-calling loop bounded by `MAX_STEPS` (LangChain `ChatOpenAI` from `agent/llm.py`, any OpenAI-compatible endpoint). `agent/tools.py` exposes exactly three tools — `get_figures`, `get_current_readings`, `get_historical_readings` — each wrapping one function in `services/{figures,readings,history}.py`, which all go through `services/mbbr_api.py` (shared GET, Bearer auth, envelope unwrapping).
- `interface.py + factory.py + providers/` exists only for ASR and TTS. Everything else is plain modules — don't add abstraction layers elsewhere.
- `core/helpers.py` holds endpoint-level helpers (WAV validation/transcription, voice-name resolution).

## Invariants

- **Understanding the operator is the LLM's job.** No intent classifier, figure resolver, fuzzy matcher, or alias table in Python. Tool results go to the model untouched. The only Python-side validation is the historical period: two real `YYYY-MM-DD` dates, ordered, not in the future, ≤ 31 days apart — a bad period is returned as a sentence for the model, not an exception.
- **The JWT never appears in a prompt, tool argument, Redis, or log.** It lives in the closure `build_tools` creates and only becomes an `Authorization` header in `mbbr_api.py`.
- Redis memory (`conversation:{id}`) stores only user/assistant text; append + trim to `MEMORY_MAX_MESSAGES` + TTL refresh happen in one transaction.
- The system prompt lives in `agent/prompts.py`; the model must never invent figures, ids, or readings.

## Configuration

`src/core/config.py` has **no in-code defaults** — `.env` is the single source of truth and a missing key fails at startup. A new setting must be added to `.env`, `.env.example`, `Settings`, and `BASE_ENV` in `tests/settings_factory.py` together. `LLM_TOP_K` / `LLM_ENABLE_THINKING` are vLLM extensions: leave blank for hosted APIs (they are then dropped from the request).

## Voices

TTS built-in speakers are read from the loaded model. Cloned voices (none ship today) are declared in `CUSTOM_VOICES` in `src/services/tts/voices.py` and need `assets/voices/<name>.wav` + a word-for-word matching `.txt` transcript, both required at startup; the `Dockerfile` would then also need `COPY assets/ assets/`. `TTS_DEFAULT_VOICE` must name an existing voice or startup fails.

## Docker

The image (`Dockerfile`) installs the locked deps with `uv sync --frozen --no-dev --no-install-project` and runs `uvicorn main:app --app-dir src` with a single worker (each worker would load its own models onto the GPU). `.env` is excluded from the image and passed at runtime via compose `env_file`; HF models are cached in the `hf-cache` volume, not in the image.

## Tests

Tests use hand-written fakes only — no live Redis, LLM, MBBR API, model downloads, or GPU. Build settings with `tests/settings_factory.py` and fake HTTP with `install_fake_client` in `tests/http_fake.py`.
