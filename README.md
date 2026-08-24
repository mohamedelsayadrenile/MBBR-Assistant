# MBBR Assistant

A voice chatbot for MBBR wastewater-treatment plant operators. An operator asks a
question in Egyptian Arabic, the assistant reads the live values from the MBBR
platform, and answers by voice. Naming a figure narrows the answer to it; asking
for a measurement on its own lists it across every figure that reports it. Only a
question about a past period needs a figure, so there the assistant asks which one.

```
Voice → ASR → agent (LLM + tools) → MBBR API → TTS → Voice
```

The MBBR API is the only source of truth: the assistant never invents a figure
name, a figure id, or a reading.

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI |
| Agent | One LLM with three tools, in a small tool-calling loop |
| ASR | `CohereLabs/cohere-transcribe-arabic-07-2026`, local via `transformers` |
| LLM | Any OpenAI-compatible endpoint — Qwen API in dev, self-hosted vLLM in prod |
| TTS | `mohammedaly22/VoiceTut-TTS`, local |
| Memory | Redis, latest N messages on a 1800 s sliding TTL |

## Project structure

```text
src/
├── agent/
│   ├── agent.py        # the tool-calling loop
│   ├── llm.py          # ChatOpenAI from settings, LLMError, <think> stripping
│   ├── prompts.py      # the system prompt
│   └── tools.py        # the three tools the model can call
├── services/
│   ├── asr/            # interface + factory + providers/cohere.py
│   ├── tts/            # interface + factory + providers/voicetut.py + voices.py
│   ├── mbbr_api.py     # shared GET + Bearer auth + envelope unwrapping
│   ├── figures.py      # get_figures(jwt, settings)
│   ├── readings.py     # get_current_readings(jwt, settings) — the whole plant
│   ├── history.py      # get_historical_readings(jwt, figure_id, start, end, settings)
│   ├── memory.py       # RedisMemory
│   └── chat_service.py # one turn: memory → agent → memory → speech
├── api/v1/endpoints/{chat.py,voices.py}
├── core/{config.py,logging.py}
└── main.py

assets/voices/          # elsayad.wav + elsayad.txt, the cloned voice
```

`interface + factory + providers/` is used only for the external-provider
services. Everything else is plain modules and functions.

There is no `services/llm/`: `agent/llm.py` owns the shared LangChain
`ChatOpenAI` handle and `LLMError` for a failed turn.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in LLM_API_KEY
```

`.env` is the single source of truth — `src/core/config.py` declares no in-code
defaults, so a missing key fails at startup rather than defaulting silently. Add a
new setting to `.env`, `.env.example`, and `Settings` in the same change.

## Running locally

```bash
docker run -d --name mbbr-redis -p 6379:6379 redis:7-alpine
uv run uvicorn main:app --reload
curl localhost:8000/health          # {"status":"ok"}
```

The app is started as `main:app`, not `src.main:app` — `src/` is the package root.
Both the ASR and the TTS model load at startup, one after the other, so the first
request pays no load cost. Expect `/health` to stay unreachable until they are in.

## Manual tester (Streamlit)

A small UI for exercising the assistant by voice. It drives the HTTP endpoint, so
it tests the real pipeline rather than importing anything from `src/`.

```bash
uv run uvicorn main:app --reload          # terminal 1
uv run streamlit run streamlit_app.py     # terminal 2
```

Put a JWT in the sidebar, pick a **Voice** if you want one other than the default,
then either record a question with the mic (or upload a
WAV) and press **Send**, or type one into the chat box at the bottom. A spoken
question is shown as text and played back as audio; a typed one comes back as text
only. **New conversation** starts a fresh `conversation_id` so you can test the
figure follow-up flow from scratch.

Point it at a non-default API with `CHAT_API_BASE_URL`, or edit the field in the
sidebar.

## API

### `POST /api/v1/chat`

`multipart/form-data`:

| field | type | notes |
|---|---|---|
| `conversation_id` | text | scopes the Redis memory |
| `jwt` | text | forwarded to the MBBR APIs as `Authorization: Bearer <jwt>` |
| `audio` | file | WAV — send this **or** `text`, not both |
| `text` | text | the message as typed — send this **or** `audio`, not both |
| `voice` | text | optional; who reads the reply. Case-insensitive, blank means the default, ignored on a `text` turn. `GET /api/v1/voices` lists the valid names |

Voice in, voice out; text in, text out. An `audio` turn is transcribed and the
reply comes back spoken; a `text` turn skips the ASR and the TTS, so the response
carries no audio fields.

```bash
curl -F conversation_id=c1 -F jwt="$JWT" -F audio=@sample.wav \
     http://localhost:8000/api/v1/chat

curl -F conversation_id=c1 -F jwt="$JWT" -F text="عايز درجة حرارة الماية دلوقتي" \
     http://localhost:8000/api/v1/chat

curl -F conversation_id=c1 -F jwt="$JWT" -F audio=@sample.wav -F voice=Elsayad \
     http://localhost:8000/api/v1/chat
```

```json
{
  "conversation_id": "c1",
  "transcript": "عايز درجة حرارة الماية دلوقتي",
  "reply": "درجة حرارة الماية في خزان A 24 درجة",
  "audio_base64": "...",
  "audio_content_type": "audio/wav"
}
```

For a `text` turn the reply is `{conversation_id, transcript, reply}` — the
`transcript` echoes what was sent. The audio fields are also omitted when speech
synthesis fails; the text reply still comes back.

| Status | Cause |
|---|---|
| 422 | blank `conversation_id`/`jwt`, neither or both of `audio`/`text`, non-WAV or empty audio, nothing recognised in the audio, unknown `voice` |
| 413 | audio over `ASR_MAX_AUDIO_BYTES` |
| 503 | transcription failed |

A `200` does not always mean everything worked: if the LLM, Redis, or the MBBR API
is down, the reply is an Arabic apology, spoken as usual.

### `GET /api/v1/voices`

```json
{ "voices": ["Abdelrahman", "…", "Elsayad"], "default": "Asmaa" }
```

Everything `POST /api/v1/chat` accepts as `voice`, and which one it falls back to.

### `GET /health`

Returns `{"status": "ok"}`. Does not check dependencies.

## Voices

Two kinds sit behind one name:

- The **built-in speakers** ship inside the VoiceTut model repo. The list is read
  off the loaded model rather than hardcoded, so a checkpoint that adds a speaker
  needs no code change.
- **`Elsayad`** is cloned zero-shot from `assets/voices/elsayad.wav` and its
  transcript in `assets/voices/elsayad.txt`. The model is given both, and **the
  transcript must match the clip word for word** — a wrong one does not error, it
  quietly degrades every reply in that voice. Both files are required at startup;
  the app refuses to start without them.

Adding another cloned voice means dropping a clip and a transcript into
`assets/voices/` and naming the pair in `CUSTOM_VOICES` in
[src/services/tts/voices.py](src/services/tts/voices.py).

`TTS_DEFAULT_VOICE` picks what an unspecified request gets; it may name a built-in
or a cloned voice, and startup fails if it names neither.

## Agent

The whole agent is a tool-calling loop:

```text
user message → LLM → tool call if it needs data → LLM → reply
```

[agent.py](src/agent/agent.py) builds the message list — the system prompt, the
conversation so far as real human/assistant messages, then this turn — and loops:
ask the model, run whatever tools it called, ask again, until it answers with
text. `MAX_STEPS` bounds the loop.

The three tools in [tools.py](src/agent/tools.py) each call one API and hand the
data back untouched:

| Tool | API | Returns |
|---|---|---|
| `get_figures()` | `/api/telemetry/figures` | every figure with its id and name |
| `get_current_readings()` | `/api/figures/readings/latest/` | the whole plant: every figure, every sensor it reports, and its current value |
| `get_historical_readings(figure_id, from_date, to_date)` | `/api/telemetry/figures/daily-averages` | daily averages for one figure over a period |

Understanding the operator is the model's job, not Python's. There is no intent
classifier, no figure resolver, no sensor matcher, no fuzzy matching and no alias
table. The model reads the operator's wording — typos, Arabic-Indic digits
(«جهاز ١»), Arabic written in latin letters («gehaz 1»), filler words — against
the real data a tool returned, and decides what answers the question. A follow-up
like «ومتوسطها امبارح؟» works because the conversation history is in the prompt,
not because anything is tracked in Python.

Python does four things and nothing else:

1. Provides the tools.
2. Keeps the JWT out of the model's messages. It lives in the closure
   `build_tools` creates, is never a tool argument, and only ever becomes an
   `Authorization` header in [mbbr_api.py](src/services/mbbr_api.py).
3. Runs the tool calls, turning an `MBBRAPIError` into a tool result the model
   can explain rather than a crash.
4. Checks the one thing that cannot safely be left to the model: that a
   historical period is two real YYYY-MM-DD dates, in order, not in the future,
   and at most 31 days apart. A bad period comes back as a sentence the model
   reads, so it asks the operator again instead of hitting the API.

**The JWT is never in a prompt, a tool argument, Redis, or a log line.**

Redis is the only cross-turn memory. Every value the assistant says comes from a
tool result on that turn — the prompt forbids inventing a figure, a reading, a
count, or an id.

## Memory

Key `conversation:{conversation_id}`. Each write is one transaction: append, trim
to `MEMORY_MAX_MESSAGES`, refresh the TTL — so an active conversation never expires
mid-flow. Only user and assistant text is stored; tool results and the JWT are not.

## Development

```bash
uv run pytest                        # full suite
uv run pytest tests/test_agent.py    # one file
uv run python -m compileall src      # syntax check (no linter configured)
uv run python -m py_compile streamlit_app.py
```

Tests use hand-written fakes — no live Redis, LLM, or MBBR API, no model downloads,
no GPU.

## Notes

- **Both readings endpoints name a sensor `name` / `unit`.** Latest carries
  `{name, value, unit}` (the `value` of a valve or pump is a state, and its unit is
  null); daily-averages adds `name_ar` and a `daily` series. Both payloads reach the
  model exactly as the API returned them.
- **`figures/readings/latest/` answers for the whole plant in one call**, figure ids
  and names included — so a current-reading question is one tool call: the model
  gets the plant and picks the figure and the measurement out of it itself.
- **`LLM_TOP_K` and `LLM_ENABLE_THINKING` are vLLM-server extensions.** Leave them
  blank on the Qwen API; when unset they are dropped from the request entirely, so
  a hosted endpoint never sees an unknown field.
- **`APIs.ipynb` is gitignored** — its saved output contains a live admin JWT.
