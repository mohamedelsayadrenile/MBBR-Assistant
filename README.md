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
│   ├── agent.py        # three-node graph, orchestration, and safety checks
│   ├── llm.py          # ChatOpenAI from settings, LLMError, <think> stripping
│   └── prompts.py      # interpret, device-resolver, and compose prompts
├── services/
│   ├── asr/            # interface + factory + providers/cohere.py
│   ├── tts/            # interface + factory + providers/voicetut.py + voices.py
│   ├── mbbr_api.py     # shared GET + Bearer auth + envelope unwrapping
│   ├── devices.py      # get_devices(jwt, settings)
│   ├── readings.py     # get_current_readings(jwt, settings) — the whole plant
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

`interpret` uses LangChain structured output to read the operator's message and
extract `intent`, `sensor`, `device_name`, and (for historical questions) the
date range — a pure LLM step that calls no APIs and resolves nothing.
`execute` is deterministic Python that routes the turn: `reply` and
conversational turns are answered or asked about without any API call; a
`station` question fetches the live list and answers only from it. `station` is
the plant as a whole — how many devices, what they are called — and never a
question about one device: «جهاز كذا بيقيس إيه؟» names no measurement but is
still a readings question, interpreted as `current` with `sensor` = `"all"`.

The two reading paths are deliberately different in shape:

- **Current** is one call and no logic. `/api/readings/latest/all` answers with
  the whole plant — every device, its id and name, and every sensor it is wired
  for — so `execute` fetches it and hands the response to `compose` untouched.
  Identifying the device, finding the measurement, spotting a null value and
  deciding what to say are all the model's reasoning, steered by the compose
  prompt. Nothing is resolved, filtered or reshaped on the way.
- **Historical** keeps the resolver machinery, because `daily-averages` answers
  for one device over one period and needs a verified id before it is called: a
  dedicated LLM device resolver (`matched` / `ambiguous` / `not_found`) runs
  against a fresh `get_devices` list, Python verifies the returned id, and a
  second small resolver matches the measurement against what the payload reports
  — see [Measurement support](#measurement-support).

`compose` emits a fixed clarification when one is still pending, otherwise
answers from the trusted result. It is given the conversation so far for one
reason only — so a follow-up («وبيقيس إيه تاني؟») can answer with what has not
been said yet instead of repeating itself. Every value it states still comes from
this turn's result; the prompt forbids taking a reading, a device or a count from
the history.

There are no model tools, agent loops, retries, worker threads, sync wrappers,
checkpoints, or custom reducers. There is no Python fuzzy matcher: matching the
operator's wording to a real device — or to a real measurement — is an LLM
resolver step that sees the live values, floored by an exact check in code. The assistant never invents data — it answers only from the device list it
fetched and the readings it retrieved. Redis remains the only cross-turn memory.

**The JWT is never in graph state, a prompt, Redis, or a log line.** It is passed
to API nodes through LangGraph runtime context.

### Device selection

The operator picks the device; the assistant never infers it from the measurement.
A question with no device — and none already established — is routed to the fixed
clarification «تقصد أي جهاز؟» without fetching the list.

The operator's answer ("جهاز 2") arrives on a later turn and Redis holds only
user/assistant text. The interpretation model recovers the pending request from
that history, and `execute` then fetches a fresh device list and resolves the
wording through the device resolver before reading any data.

### The device the conversation is about

A conversation is about one device until the operator names another. They say it
once — «مستوى المياه في جهاز 2 كام؟» — and every follow-up («والضغط كام؟»,
«الجهاز بيقيس إيه تاني؟») is about جهاز 2 without being asked again.

That is the interpretation model's to work out from real LangChain human and
assistant messages: when a turn points at the device already under discussion
(«الجهاز», «هو», or a bare follow-up) it copies forward the wording used earlier.
A bare «الجهاز» with nothing before it names no device, so that turn is asked
«تقصد أي جهاز؟» instead. Nothing about the device is stored separately: Redis
holds the words, while a stored id could go stale against a device list that can
change.

### Reading the name the operator actually said

The operator is speaking, and a transcriber writes it down, so the name that
arrives is rarely the name the API holds: Arabic-Indic digits («جهاز ١»), a missing
space («جهاز١»), Arabic in latin letters («gehaz 1», «jihaz 1»), number words and
ordinals («الجهاز التاني»), the definite article, the filler words around the name
(«رقم»، «من فضلك»), and ordinary typos.

On the historical path, reading through those variants and selecting from the
live list is a dedicated LLM device-resolver step: after interpretation and a
fresh `get_devices` fetch it returns `matched`, `ambiguous`, or `not_found`.
`ambiguous` names the top two real devices so the assistant can ask «تقصد ...
ولا ...؟»; only a verified `matched` id ever reaches the daily-averages API.

On the current path there is no id to verify — the snapshot already holds every
device — so the same reading-through-variants job belongs to the composer, which
sees the real names and answers only from them.

### When the device does not exist

Before a daily-averages call, Python only verifies that the id returned by the
resolver exists in the freshly fetched list. A resolver result that names an id
outside the list is treated as `not_found`, so that API never receives an
unverified id. A current question needs no such check: the request carries no id,
and the composer answers from the device names in the payload itself.

### Measurement support

What a device reports **is** the payload it returns: a measurement present in the
response is available, one absent from it is not, and a measurement present with no
value has simply not been read. That rule holds on both paths; what differs is who
applies it. On the historical path it is deterministic Python, described below. On
the current path the whole snapshot is in front of the composer, which applies the
same rule as a prompt instruction — the reply may name only a device and a sensor
that are in the payload it was given.

On the historical path, the operator's wording is matched to a reported measurement the same way a device
name is matched to the live list — a small resolver step sees only that device's
reported measurements (`type`, the Arabic `type_ar`, and the unit) and copies one
`type` back. `match_sensor` in [utils.py](src/agent/utils.py) then floors the answer
against the payload, so a name the payload never carried cannot reach the reply. The
interpretation model passes the operator's wording through verbatim and never invents
an English sensor name; the naming is not consistent between devices (`PH` and `ph`,
`Flow` and `flow_rate`, `LEVEL3` and `water_level`), so no vocabulary is pinned in
code, and no alias table or fuzzy matcher exists.

Three outcomes, kept distinct:

- **The measurement is reported.** `narrow_daily` reduces the payload to that one
  series before compose sees it, so the reply reads a value that is provably there
  rather than picking one out of a list.
- **The payload has measurements but not that one** → `sensor_not_supported`, which
  carries the device name and the Arabic names of what it does report. The reply says
  both («جهاز 2 مش بيقيس الحرارة، بيقيس الحموضة والتدفق.»).
- **The payload is empty, or the series holds no averages** → `no_readings`. A device
  that sent nothing has a missing reading, not a missing sensor, so this is never
  phrased as the device not measuring the thing.

The current path reaches the same three answers from the prompt instead: the endpoint
lists every sensor a device is wired for, with a null value when there is none, so a
missing sensor and a missing reading are genuinely distinguishable in the payload.
For an `"all"` question the composer reports the sensors that have values and says
the rest are unavailable.

The trade is explicit: the current path sends the composer the whole plant — 51
devices, about 10 KB of JSON — and trusts the model to read it, in exchange for no
resolver round trips, no filtering code, and one HTTP call per turn.

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
  null); daily-averages adds `name_ar` and a `daily` series. Only daily-averages has
  the Arabic name, so `payload_sensors` in [utils.py](src/agent/utils.py) leaves
  `type_ar` empty for current readings and the sensor resolver matches on `type`
  there. The services still pass `data` through untouched.
- **`readings/latest/all` answers for the whole plant in one call**, device ids and
  names included — which is why the current path needs no device list, no resolver
  step and no narrowing: it fetches once and hands the response to the composer.
- **`LLM_TOP_K` and `LLM_ENABLE_THINKING` are vLLM-server extensions.** Leave them
  blank on the Qwen API; when unset they are dropped from the request entirely, so
  a hosted endpoint never sees an unknown field.
- **`APIs.ipynb` is gitignored** — its saved output contains a live admin JWT.
