# MBBR Assistant — Deployment Guide

This guide takes you from a fresh server to a running MBBR Assistant: host
preparation, model access, choosing the language model, configuration, first
start, using the API, and day-to-day operations.

## 1. Overview

The MBBR Assistant is a voice assistant for MBBR wastewater-plant operators. An
operator asks a question in Egyptian Arabic; the assistant reads live and
historical values from the MBBR platform and answers by voice.

```
Operator voice → Speech-to-text → Language model + MBBR API → Text-to-speech → Voice reply
```

| Component | What it is | Where it runs |
|---|---|---|
| `api` | The assistant (FastAPI) with the speech-to-text and text-to-speech models | Docker container, GPU 0 |
| `redis` | Short-term conversation memory | Docker container |
| Language model (LLM) | Qwen3.6-35B-A3B — **either** the Qwen cloud API **or** a local vLLM server | Alibaba Cloud, or Docker container on GPU 1 |

Models used:

| Role | Model | Size | Access |
|---|---|---|---|
| Speech-to-text | `CohereLabs/cohere-transcribe-arabic-07-2026` | ~4 GB | Gated — requires accepting terms on Hugging Face |
| Text-to-speech | `mohammedaly22/VoiceTut-TTS` | ~7.5 GB | Public |
| LLM (local option) | `Qwen/Qwen3.6-35B-A3B-FP8` | ~37.5 GB | Public |

## 2. Requirements

### Hardware

Choose the column that matches the LLM option you will use (see section 6).

| | Option A — Qwen API | Option B — Local LLM |
|---|---|---|
| GPU for speech models | 1 × NVIDIA GPU, ≥ 24 GB VRAM (e.g. RTX 3090 / 4090, L4, A10) | same |
| GPU for the LLM | — | 1 additional NVIDIA GPU, ≥ 48 GB VRAM (e.g. L40S, RTX 6000 Ada, A6000) |
| CPU / RAM | 8+ cores, 32 GB RAM | 16+ cores, 64 GB RAM |
| Free disk | 60 GB | 120 GB |

The local LLM must have its own GPU. Qwen3.6-35B-A3B does not fit on a 24 GB card
(even 4-bit builds are ~25 GB), and sharing a GPU with the speech models causes
out-of-memory errors.

### Software

- Ubuntu 22.04 or 24.04 (x86_64)
- NVIDIA driver 570 or newer
- Docker Engine with Docker Compose v2
- NVIDIA Container Toolkit

### Network

- Internet access on the **first start** (models are downloaded from Hugging Face).
  Later starts use the local cache.
- Network access from the server to the MBBR platform API.
- Option A only: outbound HTTPS to Alibaba Cloud Model Studio.

### Accounts

- **Hugging Face** account — required for the speech-to-text model.
- **Alibaba Cloud Model Studio** account — only for Option A.

## 3. Prepare the host

1. **NVIDIA driver.** Install it and confirm all GPUs are visible:
   ```bash
   nvidia-smi
   ```
2. **Docker Engine + Compose v2.** Follow
   <https://docs.docker.com/engine/install/ubuntu/>, then confirm:
   ```bash
   docker compose version
   ```
3. **NVIDIA Container Toolkit.** Follow
   <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>,
   then register it with Docker:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
4. **Check that containers can see the GPU:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
   ```
   You should see the same GPU table as in step 1.

## 4. Get the project

Copy or clone the project to the server and create your configuration file:

```bash
cd MBBR-Assistant
cp .env.example .env
```

All commands in the rest of this guide are run from this folder.

## 5. Hugging Face access (speech-to-text model)

The speech-to-text model is gated: Hugging Face only serves it to accounts that
have accepted its terms.

1. Sign in at <https://huggingface.co>.
2. Open <https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026> and
   accept the access conditions.
3. Create a **Read** token at <https://huggingface.co/settings/tokens>.
4. Put it in `.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```

Without this, the first start fails with a `401` error and the `api` container
keeps restarting.

## 6. Choose the language model

The assistant works with any OpenAI-compatible endpoint. Pick **one** option.

### Option A — Qwen cloud API (simplest)

No extra GPU is needed. Operator questions and plant readings are sent to Alibaba
Cloud for processing.

1. Sign up for Alibaba Cloud Model Studio and create an API key:
   <https://www.alibabacloud.com/help/en/model-studio/get-api-key>.
2. Set in `.env`:
   ```
   LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
   LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
   LLM_MODEL=qwen3.6-35b-a3b
   LLM_TOP_K=
   LLM_ENABLE_THINKING=
   ```
   Accounts in the mainland-China region use
   `https://dashscope.aliyuncs.com/compatible-mode/v1` instead.
   `LLM_TOP_K` and `LLM_ENABLE_THINKING` must stay **blank** for the cloud API.

### Option B — Local LLM with vLLM (data stays on-premises)

Runs Qwen3.6-35B-A3B (FP8) on the second GPU. In the commands below, GPU `1` is
the LLM GPU and GPU `0` is used by the assistant — check the numbering with
`nvidia-smi`.

1. Start the vLLM server:
   ```bash
   docker run -d --name qwen-vllm \
     --gpus '"device=1"' \
     --ipc=host \
     --restart unless-stopped \
     -p 8001:8000 \
     -v vllm-hf-cache:/root/.cache/huggingface \
     vllm/vllm-openai:latest \
     --model Qwen/Qwen3.6-35B-A3B-FP8 \
     --served-model-name qwen3.6-35b-a3b \
     --max-model-len 32768 \
     --gpu-memory-utilization 0.90 \
     --enable-auto-tool-choice \
     --tool-call-parser qwen3_coder \
     --reasoning-parser qwen3 \
     --language-model-only
   ```
   - Port `8001` on the host is used because the assistant itself uses `8000`.
   - The tool-calling and reasoning flags are **required**: the assistant reads
     plant data through tool calls.
   - The first start downloads ~37.5 GB. Follow progress with
     `docker logs -f qwen-vllm`.

2. Check the server answers:
   ```bash
   curl http://localhost:8001/v1/models
   ```
   The response must list `qwen3.6-35b-a3b`.

3. Set in `.env`:
   ```
   LLM_BASE_URL=http://host.docker.internal:8001/v1
   LLM_API_KEY=EMPTY
   LLM_MODEL=qwen3.6-35b-a3b
   LLM_TOP_K=20
   LLM_ENABLE_THINKING=false
   ```
   - Use `host.docker.internal`, **not** `localhost`: the assistant runs inside a
     container, where `localhost` is the container itself.
   - `LLM_API_KEY` must not be empty; vLLM ignores its value.
   - `LLM_ENABLE_THINKING=false` keeps voice replies fast. Set it to `true` for
     more deliberate (slower) answers.

## 7. Configure `.env`

Every key in `.env.example` must be present in `.env` — the assistant refuses to
start if one is missing, and the error names the key. These are the keys you may
need to change; leave everything else as shipped.

| Key | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token (section 5) |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TOP_K`, `LLM_ENABLE_THINKING` | Language model (section 6) |
| `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TOP_P` | LLM sampling; the shipped values are tuned, change only if needed |
| `MBBR_API_BASE_URL` | Base URL of your MBBR platform API |
| `PLANT_TIMEZONE` | Plant time zone, used for "today" / "yesterday" (e.g. `Africa/Cairo`) |
| `TTS_DEFAULT_VOICE` | Voice used when a request names none (see section 9.3) |
| `TTS_SPEED` | Speaking rate (`1.0` = normal) |
| `TTS_NUM_STEP` | Speech quality vs. speed; lower is faster, `48` is the shipped balance |
| `REDIS_TTL_SECONDS` | How long an idle conversation is remembered (default 30 minutes) |
| `MEMORY_MAX_MESSAGES` | How many recent messages the assistant remembers per conversation |
| `ASR_MAX_AUDIO_BYTES` | Maximum uploaded audio size (default 5 MB) |
| `LOG_LEVEL` | `INFO` normally; `DEBUG` when troubleshooting |

`REDIS_URL` is overridden by Docker Compose automatically; do not change it.

## 8. Start and verify

1. **Build and start:**
   ```bash
   docker compose up -d --build
   ```
2. **Follow the startup logs:**
   ```bash
   docker compose logs -f api
   ```
   The first start downloads both speech models (~11.5 GB) and loads them onto the
   GPU — allow up to 15 minutes. The service is ready when Uvicorn logs
   `Application startup complete`. Later starts take about a minute.
3. **Health check:**
   ```bash
   curl http://localhost:8000/health
   # {"status":"ok"}
   ```
   `docker compose ps` should show `api` as `healthy`.
4. **List voices:**
   ```bash
   curl http://localhost:8000/api/v1/voices
   ```
5. **Text test** (replace `$JWT` with a valid MBBR platform token):
   ```bash
   curl -F conversation_id=test1 -F jwt="$JWT" \
        -F text="عايز درجة حرارة الماية دلوقتي" \
        http://localhost:8000/api/v1/chat
   ```
   The `reply` should contain a real reading from your plant.
6. **Voice test** with a recorded WAV question:
   ```bash
   curl -F conversation_id=test1 -F jwt="$JWT" -F audio=@question.wav \
        http://localhost:8000/api/v1/chat
   ```
   The response includes `transcript` (what was heard) and `audio_base64` (the
   spoken reply).

## 9. Using the API

Interactive API docs are available at `http://<server>:8000/docs`.

### 9.1 `POST /api/v1/chat`

Request: `multipart/form-data`.

| Field | Required | Notes |
|---|---|---|
| `conversation_id` | yes | Any string. Messages with the same id share memory. Use one id per operator session. |
| `jwt` | yes | The operator's MBBR platform token. Used only to call the MBBR API; never stored or logged. |
| `audio` | one of `audio` / `text` | WAV file (`.wav` or `audio/wav`), up to `ASR_MAX_AUDIO_BYTES` |
| `text` | one of `audio` / `text` | The question as text |
| `voice` | no | Voice for the spoken reply; case-insensitive. Blank = default voice. |

**Voice in → voice out; text in → text out.** An `audio` request is transcribed and
answered with speech. A `text` request returns text only.

Response:

```json
{
  "conversation_id": "test1",
  "transcript": "عايز درجة حرارة الماية دلوقتي",
  "reply": "درجة حرارة الماية في خزان A 24 درجة",
  "audio_base64": "UklGR...",
  "audio_content_type": "audio/wav"
}
```

`audio_base64` and `audio_content_type` are present only for `audio` requests. If
speech synthesis fails, the text reply is still returned without them.

Save the spoken reply to a file:

```bash
curl -s -F conversation_id=test1 -F jwt="$JWT" -F audio=@question.wav \
     http://localhost:8000/api/v1/chat \
  | python3 -c "import sys,json,base64; open('reply.wav','wb').write(base64.b64decode(json.load(sys.stdin)['audio_base64']))"
```

### 9.2 Errors

| Status | Meaning |
|---|---|
| `200` | Answered. If the LLM, the MBBR API, or Redis is unavailable, the reply is the apology «معلش، حصلت مشكلة مؤقتة. جرّب تاني بعد شوية.» |
| `422` | Missing `conversation_id` / `jwt`; both or neither of `audio` / `text`; audio is not WAV or is empty; no speech recognised; unknown `voice` |
| `413` | Audio larger than `ASR_MAX_AUDIO_BYTES` |
| `503` | Speech-to-text failed — ask the operator to repeat |

### 9.3 `GET /api/v1/voices`

```json
{ "voices": ["Abdelrahman", "Abdullah", "...", "Omnia"], "default": "Asmaa" }
```

Built-in voices: Abdelrahman, Abdullah, Kamal, Hossam, Mohamed, Omar, Sayed, Zaki,
Aly, Essam, Ahmed, Asmaa, Esraa, Hanan, Sarah, Yasmin, Omnia.
`TTS_DEFAULT_VOICE` must be one of them, or startup fails.

### 9.4 `GET /health`

Returns `{"status":"ok"}` once both speech models are loaded. It does not check the
LLM or the MBBR API — use a text test (section 8, step 5) for that.

### 9.5 What the assistant can answer

- **Current readings** — "What is the dissolved oxygen in tank A now?"; a
  measurement without a figure lists it across all figures.
- **Historical readings** — daily averages for one figure over a period of up to
  31 days, not in the future. If the figure is not named, the assistant asks.
- **Figures** — which figures (tanks, units) exist.

All values come from the MBBR API at the moment of the question; the assistant does
not invent readings.

## 10. Operations

| Task | Command |
|---|---|
| Status | `docker compose ps` |
| Logs | `docker compose logs -f api` |
| Restart | `docker compose restart api` |
| Stop | `docker compose down` |
| Start again | `docker compose up -d` |
| Apply `.env` changes | `docker compose up -d` (recreates `api` with the new values) |
| Update to a new release | replace the project files (or `git pull`), then `docker compose up -d --build` |
| Local LLM logs / restart | `docker logs -f qwen-vllm` / `docker restart qwen-vllm` |

All containers restart automatically after a reboot or crash.

**Data volumes:**

| Volume | Contents |
|---|---|
| `hf-cache` | Speech models (~11.5 GB) |
| `redis-data` | Conversation memory (short-lived) |
| `vllm-hf-cache` | Local LLM weights (Option B) |

`docker compose down` keeps the volumes. **`docker compose down -v` deletes them**,
and the next start downloads the models again.

**Servers without internet:** start once on a connected machine, then copy the
`hf-cache` volume (and `vllm-hf-cache` for Option B) to the offline server. After
that, `HF_TOKEN` and internet access are no longer needed.

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `api` restarts in a loop; logs show `401` / `GatedRepoError` | `HF_TOKEN` missing, or model terms not accepted | Section 5, then `docker compose up -d` |
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit missing or not configured | Section 3, step 3 |
| `CUDA out of memory` | Another process uses the assistant's GPU, or vLLM runs on the same GPU | Check `nvidia-smi`; run vLLM on a separate GPU |
| `api` never becomes `healthy` | Models still downloading, or a startup error | `docker compose logs api` |
| Startup error naming a setting (e.g. `Field required`) | Key missing in `.env` | Copy the key from `.env.example` |
| Startup error about the default voice | `TTS_DEFAULT_VOICE` is not a known voice | Use a name from section 9.3 |
| Every reply is the apology message | LLM unreachable or misconfigured | Check `LLM_*` keys; for Option B use `host.docker.internal`, and confirm `curl localhost:8001/v1/models` |
| Small talk works but data questions get the apology | MBBR API unreachable or JWT invalid/expired | Check `MBBR_API_BASE_URL` and the token |
| `422 audio must be a WAV file.` | Uploaded file is not WAV | Record or convert to WAV |
| Answers come back slowly | LLM thinking enabled or GPU busy | Set `LLM_ENABLE_THINKING=false` (Option B); lower `TTS_NUM_STEP` |

For more detail, set `LOG_LEVEL=DEBUG`, run `docker compose up -d`, and reproduce
the issue.

## 12. Security

- **The API on port 8000 has no authentication of its own.** Expose it only on the
  plant's internal network. For wider access, put it behind a reverse proxy with
  HTTPS and access control, and publish it only on `127.0.0.1` by changing the
  `ports` entry in `docker-compose.yml` to `"127.0.0.1:8000:8000"`.
- Operator tokens (`jwt`) are only forwarded to the MBBR API; the assistant never
  stores or logs them. Use HTTPS for `MBBR_API_BASE_URL` when the platform supports
  it, so tokens are not sent in clear text.
- `.env` holds your Hugging Face token and LLM key. Keep it readable only by
  administrators (`chmod 600 .env`) and never share it.
- Option A sends questions and readings to Alibaba Cloud. Use Option B if plant data
  must not leave the premises.
