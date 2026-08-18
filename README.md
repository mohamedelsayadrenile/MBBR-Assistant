# MBBR Assistant

A voice chatbot for MBBR wastewater-treatment plant operators. An operator asks a
question in Egyptian Arabic, the assistant asks which device they mean, reads the
live values for that device from the MBBR platform, and answers by voice.

```
Voice → ASR → LangGraph agent → MBBR API → TTS → Voice
```

The MBBR API is the only source of truth: the assistant never invents a device
name, a device id, or a reading.

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI |
| Agent | LangGraph — bounded interpret → execute → compose workflow |
| ASR | `CohereLabs/cohere-transcribe-arabic-07-2026`, local via `transformers` |
| LLM | Any OpenAI-compatible endpoint — Qwen API in dev, self-hosted vLLM in prod |
| TTS | `mohammedaly22/VoiceTut-TTS`, local |
| Memory | Redis, latest N messages on a 1800 s sliding TTL |

## Project structure

```text
src/
├── agent/
│   ├── agent.py        # three-node graph, execution, and safety checks
│   ├── llm.py          # ChatOpenAI from settings, LLMError, <think> stripping
│   └── prompts.py      # shared interpretation and response system prompt
├── services/
│   ├── asr/            # interface + factory + providers/cohere.py
│   ├── tts/            # interface + factory + providers/voicetut.py + voices.py
│   ├── mbbr_api.py     # shared GET + Bearer auth + envelope unwrapping
│   ├── devices.py      # get_devices(jwt, settings)
│   ├── readings.py     # get_current_readings(jwt, device_id, settings)
│   ├── history.py      # get_historical_readings(jwt, device_id, start, end, settings)
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
device follow-up flow from scratch.

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
  "reply": "أنهي جهاز؟",
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

## Agent graph

The agent is one compiled, bounded graph:

```text
interpret → execute → compose → END
```

`interpret` uses LangChain structured output to identify the requested sensor and
carry conversation context forward. If the operator named a device, it fetches
the current list and asks the same model to resolve that wording to an id. If no
device was named, it asks for one without fetching the list. `execute` applies
date and id safety checks and awaits the current/history service. `compose`
answers only the requested sensor from the trusted result.

There are no model tools, agent loops, retries, worker threads, sync wrappers,
checkpoints, or custom reducers. Redis remains the only cross-turn memory.

**The JWT is never in graph state, a prompt, Redis, or a log line.** It is passed
to API nodes through LangGraph runtime context.

### Device selection

The operator picks the device; the assistant never infers it from the measurement.
A question with no device — and none already established — is routed to a short
clarification instead of guessing from the requested measurement.

The operator's answer ("جهاز 2") arrives on a later turn and Redis holds only
user/assistant text. The interpretation model recovers the pending request from
that history, then resolves it against a fresh device list before execution.

### The device the conversation is about

A conversation is about one device until the operator names another. They say it
once — «مستوى المياه في جهاز 2 كام؟» — and every follow-up («والضغط كام؟») is about
جهاز 2 without being asked again.

That is the interpretation model's to work out from real LangChain human and
assistant messages. Nothing about the device is stored separately: Redis holds
the words, while a stored id could go stale against a device list that can change.

### Reading the name the operator actually said

The operator is speaking, and a transcriber writes it down, so the name that
arrives is rarely the name the API holds: Arabic-Indic digits («جهاز ١»), a missing
space («جهاز١»), Arabic in latin letters («gehaz 1», «jihaz 1»), number words and
ordinals («الجهاز التاني»), the definite article, the filler words around the name
(«رقم»، «من فضلك»), and ordinary typos.

Reading through those variants and selecting from the live list is the
interpretation model's job. There is no Python fuzzy matcher.

### When the device does not exist

Before execution, Python only verifies that the id selected by the model exists
in the freshly fetched list. No readings API receives an unverified id.

### Measurement support

The requested measurement is checked against what the chosen device actually returns,
for every measurement type — water temperature, flow rate, pH, humidity, pressure, or
anything else in the payload. If the reading is there, the assistant reads out its
value; if the device returned readings but not that one, it says so plainly
(«جهاز 1 مش بيقيس درجة حرارة الميه.») without listing the device's other sensors or
steering the operator to a different device. A device with no readings at all is a
separate case, and keeps its own wording.

This lives in the prompt rather than in code because [readings.py](src/services/readings.py)
passes the payload through untouched — the live API returns `count: 0` for every device
today, so the sensor field names are unverified and pinning them down in Python would
be guesswork.

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

- **Readings shape is unverified upstream.** The live API returns `count: 0` for
  every device, so `get_current_readings` passes the `data` object to the model
  untouched rather than mapping fields that might not exist. A change in the sensor
  payload is a prompt concern, not a code change.
- **`LLM_TOP_K` and `LLM_ENABLE_THINKING` are vLLM-server extensions.** Leave them
  blank on the Qwen API; when unset they are dropped from the request entirely, so
  a hosted endpoint never sees an unknown field.
- **`APIs.ipynb` is gitignored** — its saved output contains a live admin JWT.
