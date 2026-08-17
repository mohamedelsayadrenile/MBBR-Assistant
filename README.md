# MBBR Assistant

A voice chatbot for MBBR wastewater-treatment plant operators. An operator asks a
question in Egyptian Arabic, the assistant asks which device they mean, reads the
live values for that device from the MBBR platform, and answers by voice.

```
Voice → ASR → Agent/LLM → MBBR API tools → TTS → Voice
```

The MBBR API is the only source of truth: the assistant never invents a device
name, a device id, or a reading.

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI |
| Agent | CrewAI — one agent, native tool calling |
| ASR | `CohereLabs/cohere-transcribe-arabic-07-2026`, local via `transformers` |
| LLM | Any OpenAI-compatible endpoint — Qwen API in dev, self-hosted vLLM in prod |
| TTS | `mohammedaly22/VoiceTut-TTS`, local |
| Memory | Redis, latest N messages on a 1800 s sliding TTL |

## Project structure

```text
src/
├── agent/
│   ├── agent.py        # one CrewAI agent, built per request
│   ├── llm.py          # crewai.LLM from settings, LLMError, <think> stripping
│   ├── tools.py        # the two tools, per-request context, device matching
│   └── prompts.py      # Egyptian Arabic system prompt + turn rendering
├── services/
│   ├── asr/            # interface + factory + providers/cohere.py
│   ├── tts/            # interface + factory + providers/voicetut.py
│   ├── mbbr_api.py     # shared GET + Bearer auth + envelope unwrapping
│   ├── devices.py      # get_devices(jwt, settings)
│   ├── readings.py     # get_current_readings(jwt, device_id, settings)
│   ├── memory.py       # RedisMemory
│   └── chat_service.py # one turn: memory → agent → memory → speech
├── api/v1/endpoints/chat.py
├── core/{config.py,logging.py}
└── main.py
```

`interface + factory + providers/` is used only for the external-provider
services. Everything else is plain modules and functions.

There is no `services/llm/`: CrewAI owns the model call, so `agent/llm.py` is the
only LLM code — the `crewai.LLM` handle and `LLMError` for a failed turn.

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
docker compose -f docker/compose.yml up -d redis
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

Put a JWT in the sidebar, then either record a question with the mic (or upload a
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

Voice in, voice out; text in, text out. An `audio` turn is transcribed and the
reply comes back spoken; a `text` turn skips the ASR and the TTS, so the response
carries no audio fields.

```bash
curl -F conversation_id=c1 -F jwt="$JWT" -F audio=@sample.wav \
     http://localhost:8000/api/v1/chat

curl -F conversation_id=c1 -F jwt="$JWT" -F text="عايز درجة حرارة الماية دلوقتي" \
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
| 422 | blank `conversation_id`/`jwt`, neither or both of `audio`/`text`, non-WAV or empty audio, nothing recognised in the audio |
| 413 | audio over `ASR_MAX_AUDIO_BYTES` |
| 503 | transcription failed |

A `200` does not always mean everything worked: if the LLM, Redis, or the MBBR API
is down, the reply is an Arabic apology, spoken as usual.

### `GET /health`

Returns `{"status": "ok"}`. Does not check dependencies.

## Agent tools

Two tools are exposed to the model:

- `get_devices()` — the plant's real devices, as `{id, name}`.
- `get_current_readings(device_id)` — the latest readings for one device.

**The JWT is never in a tool schema, a prompt, Redis, or a log line.** The tools
are constructed fresh for each request with the JWT on a private attribute, which
CrewAI does not read when it derives the tool schema.

### Device selection

The operator picks the device; the assistant never infers it from the measurement.
A question with no device gets exactly one reply — «أنهي جهاز؟» — and nothing else.
The prompt forbids the phrasings that would leak which device carries which sensor
(«إيه اسم الجهاز اللي بيسجل درجة حرارة الميه؟» and friends), because the operator
already knows the device they mean.

The operator's answer ("جهاز 2") arrives on a later turn than the device list, and
Redis holds only user/assistant text. Rather than a second store, the prompt
requires `get_devices` before every `get_current_readings`, which puts a fresh list
in the current turn.

### When the device does not exist

If the operator answers «أنهي جهاز؟» with a device the plant does not have, the
reply is «الجهاز ده مش موجود.» — and that is decided in code, before the model is
called at all. `MBBRAgent._answered_with_an_unknown_device` matches the
operator's own words against the live device list, allowing Arabic number words
and a name said inside a longer sentence.

This cannot be left to the prompt. Measured over the live model, the rule alone
held in 5 of 16 attempts; the rest of the time the assistant quietly picked a
device that does exist and reported its readings. The check runs only on the turn
straight after «أنهي جهاز؟», where the whole message is the operator's answer, and
it defers if the device list cannot be fetched, so an outage is never reported as
a bad device name.

Matching the model's `device_id` is a separate, weaker guard: `resolve_device_id`
rejects an id that is in no device's entry, which stops an invented UUID reaching
the API. It cannot catch a *mis-mapped* device, because a model that decides
«جهاز النفخ» means «جهاز 1» passes a perfectly valid id — which is why the
operator's words are checked separately.

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
