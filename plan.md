# Add a historical-readings tool to the MBBR agent

## Context

Today the agent can only answer "what is it *now*". [tools.py](src/agent/tools.py) exposes exactly
two tools — `get_devices` and `get_current_readings` — and the whole system prompt is written
around that single readings path. An operator asking «الضغط كان عامل ايه امبارح؟» or «متوسط التدفق
الأسبوع اللي فات» has no way to be answered.

The MBBR platform already serves this: `GET /api/telemetry/daily-averages` (exercised in
[test.ipynb](test.ipynb)). This change adds a third tool over that endpoint, keeping the existing
device-selection flow exactly as it is — ask «أنهي جهاز؟» when no device is established, resolve the
name against the live `get_devices` list, then call the tool with the real id.

The outcome: the agent picks `get_current_readings` for "now" questions and
`get_historical_readings` for past-period questions, and never confuses the two.

### Verified API behaviour

Probed live against `http://41.32.195.157:8083` during planning:

| Fact | Detail |
|---|---|
| Params | `device_id`, `from`, `to` — **all three required** |
| Date format | `YYYY-MM-DD`; anything else → HTTP 400 `VALIDATION_INVALID_DATE` |
| `to` | **inclusive** (`from=to=2026-08-16` returns the 16th) |
| `to < from` | HTTP 400 `VALIDATION_FAILED` |
| Response `data` | `{device_name, sensors: [{name, name_ar, unit, daily: [{day, avg}]}]}` |
| Missing data | the sensor is still listed, with `daily: []` |

That last row matters: **"sensor absent from `sensors`" and "sensor present with an empty `daily`"
are different situations** — the first means the device does not measure it, the second means it
measures it but has no data for that period. The prompt must say both.

## Decisions taken

1. **The model computes the dates.** Tool args are `from_date` / `to_date` as `YYYY-MM-DD`;
   today's date is injected into every turn so the model has something to compute from. Python
   validates strictly afterwards, in the spirit of `resolve_device_id`.
2. **Spoken reply = period average + min–max**, one sentence.
3. **Maximum span is one month.** `MAX_RANGE_DAYS = 31` — not 30 — so that «الشهر اللي فات» for a
   31-day month (July, August…) is not rejected on an off-by-one. Change the constant to `30` if a
   hard 30 is wanted; it is used in exactly one place.

## Implementation

### 1. Config — `src/core/config.py`, `.env`, `.env.example`

Two new required settings (README rule: add to all three in the same change):

```
HISTORICAL_READINGS_API_PATH=/api/telemetry/daily-averages
PLANT_TIMEZONE=Africa/Cairo
```

```python
historical_readings_api_path: str = Field(alias="HISTORICAL_READINGS_API_PATH")
plant_timezone: str = Field(alias="PLANT_TIMEZONE")
```

`PLANT_TIMEZONE` exists so "yesterday" means the operator's yesterday, not the server's — resolved
with stdlib `zoneinfo.ZoneInfo`, no new dependency. Add both keys to
[tests/settings_factory.py](tests/settings_factory.py) `BASE_ENV`.

### 2. Service — new `src/services/history.py`

Mirror [readings.py](src/services/readings.py) exactly, including its pass-the-payload-through
rationale. Reuse `get_json_data` from [mbbr_api.py](src/services/mbbr_api.py) — no new HTTP code.

```python
async def get_historical_readings(
    jwt: str, device_id: str, start: date, end: date, settings: Settings
) -> dict[str, Any]:
    data = await get_json_data(
        settings.historical_readings_api_path,
        jwt,
        {"device_id": device_id, "from": start.isoformat(), "to": end.isoformat()},
        settings,
    )
    if not isinstance(data, dict):
        raise MBBRAPIError("Historical readings API did not return an object")
    return data
```

Log `history_fetch_started/completed` with `device_id`, `from`, `to`, and the sensor count, matching
the existing `key=value` style.

### 3. Tool — `src/agent/tools.py`

**A pure date gate, tested on its own** (sibling of `resolve_device_id`, same strict-on-purpose
posture):

```python
MAX_RANGE_DAYS = 31

def parse_date_range(raw_from: str, raw_to: str, today: date) -> tuple[date, date] | None:
    """Parse the model's two dates, or None if they are not a usable past window."""
```

- Parse with `datetime.strptime(value, "%Y-%m-%d").date()` — **not** `date.fromisoformat`, which
  also accepts `20260816` and would let a shape the API rejects through.
- `None` on: unparseable input, `start > end`, or `start` in the future.
- Clamp `end` to `today` when the model overshoots (e.g. "this month" → end of month). Lossless —
  there is no future data — and avoids failing the operator over an off-by-one.

Span is checked in `_run`, not here, so "malformed" and "too long" stay distinguishable:

```python
GET_HISTORICAL_READINGS = "get_historical_readings"
INVALID_RANGE_RESULT = "Invalid date range."
RANGE_TOO_LONG_RESULT = "Range too long."
```

**`ToolContext` gains `today: date`** so the gate is deterministic and testable rather than reading
the clock inside the tool.

**`GetHistoricalReadingsTool(_ContextBoundTool)`** — same body shape as `GetCurrentReadingsTool`:
fetch/reuse `context.devices` → `resolve_device_id` → `parse_date_range` → span check → service call
→ `json.dumps(..., ensure_ascii=False)`. Every failure is a sentinel *string*, never an exception —
`_call` already enforces this and the docstring at [tools.py:94-100](src/agent/tools.py#L94-L100)
explains why.

Args schema, with descriptions the model actually reads:

```python
class _HistoricalReadingsArguments(BaseModel):
    device_id: str   # "The real device id copied from the get_devices result."
    from_date: str   # "First day of the period, YYYY-MM-DD. Inclusive."
    to_date: str     # "Last day of the period, YYYY-MM-DD. Inclusive."
```

Tool description must draw the line against the existing tool: *"Daily average sensor readings for
one device over a past period — yesterday, a past week, a month, or a named date range. Never use
this for the current or latest value; use get_current_readings for that."*

`build_tools` returns all three.

### 4. Prompt — `src/agent/prompts.py`

New fixed Arabic literals (same contract as the existing five — the operator hears them verbatim):

```python
NO_HISTORY = "مفيش قراءات مسجلة للجهاز ده في الفترة دي."
PERIOD_TOO_LONG = "أقدر أجيب بيانات آخر شهر بحد أقصى، قولّي فترة أقصر من كده."
ASK_WHICH_PERIOD = "أنهي فترة بالظبط؟"
```

**`build_input` gains a `today: date` argument** and prepends a `# Today` block:

```
# Today
Today is 2026-08-17 (Monday). Yesterday was 2026-08-16.
```

It goes in the *turn input*, not `SYSTEM_PROMPT`, deliberately: `SYSTEM_PROMPT` is a module-level
constant evaluated at import, so a date baked in there would freeze at process start and be wrong
by the next morning. `MBBRAgent.run` computes it as
`datetime.now(ZoneInfo(settings.plant_timezone)).date()` and passes it to both `build_input` and
`ToolContext`, so the prompt and the gate always agree on what "today" is.

New `SYSTEM_PROMPT` sections:

- **`# Now versus the past`** — the routing rule, stated before anything else about the two tools.
  «دلوقتي، حالياً، آخر قراءة، كام؟» with no time word → `get_current_readings`. «امبارح، الأسبوع
  اللي فات، الشهر اللي فات، يوم 5 أغسطس، من ... لـ ...» → `get_historical_readings`. Critical
  negative rule: **a question with no time word at all is a current-readings question** — answer it,
  do not ask `ASK_WHICH_PERIOD`. Only ask that when the operator clearly means the past but the
  period is genuinely unreadable.
- **`# Working out the period`** — turn Arabic period words into `from_date`/`to_date` off the
  `# Today` line, with worked examples: امبارح → both = yesterday; الأسبوع اللي فات → `today-7` →
  yesterday; الشهر اللي فات → the previous calendar month, 1st to last day; آخر شهر → `today-30` →
  yesterday; a named date → both = that date. Never invent a date, never guess today.
- **`# Reporting a past period`** — the one-sentence summary shape. Find the sensor by `name` or
  `name_ar`, average the `avg` values across the returned days, and say the average with its `unit`
  plus the min–max, e.g. «متوسط التدفق الأسبوع اللي فات كان 160 لتر في الدقيقة، وتراوح بين 121
  و171». Never read the days out one by one. Then the three-way split the API shape forces:
  - sensor present with values → the sentence above;
  - sensor present, `daily` empty → `NO_HISTORY`;
  - sensor **not in `sensors` at all** → the existing "does not measure it" sentence.
- **Sentinel mapping** — extend the existing tool-result rules: `"Invalid date range."` →
  `ASK_WHICH_PERIOD`; `"Range too long."` → `PERIOD_TOO_LONG`; `"Device not found."` and
  `"Tool failed temporarily."` already map correctly and apply unchanged.

**Generalise the existing sections.** `# Tool order`, `# Choosing the device`, `# Carrying the
device from one question to the next`, and `# Identifying the device` all name
`get_current_readings` literally ("before you call get_current_readings", "Call get_devices before
every single get_current_readings call"). Reword each to cover **either readings tool**, so
device-selection, carry-over and name-matching behave identically on the new path. This is the
highest-risk edit in the change — the device rules are load-bearing and heavily tuned.

### 5. `src/agent/agent.py`

- Compute `today` once per turn from `settings.plant_timezone`; pass to `build_input` and
  `ToolContext`.
- **Raise `MAX_ITER` from 3 to 4.** With three tools the model has a real chance of reaching for the
  wrong one first; at 3 iterations `get_devices` → wrong tool → correct tool leaves nothing for the
  answer and the turn dies as `LLMError`. The extra iteration costs one LLM round trip only when
  it is actually needed.

### 6. Tests

| File | Coverage |
|---|---|
| `tests/test_history.py` (new) | Bearer header forwarded; path + `{device_id, from, to}` params; `data` returned untouched; non-dict and HTTP error → `MBBRAPIError`. Mirror [test_readings.py](tests/test_readings.py), using the **real captured payload** from [test.ipynb](test.ipynb) as the fixture. |
| [tests/test_tools.py](tests/test_tools.py) | Three tools exposed (update `test_only_two_tools_are_exposed`); new schema is exactly `device_id/from_date/to_date`; JWT still absent from all three schemas. `parse_date_range`: valid range, single day, reversed → `None`, `2026-8-1` → `None`, garbage → `None`, future `start` → `None`, future `end` clamped, 31 days OK, 32 days → too long. |
| [tests/test_prompts.py](tests/test_prompts.py) | New literals appear verbatim and unquoted (add them to `FIXED_REPLIES`); `# Today` block rendered with the right date. **`test_build_input_returns_a_first_turn_unchanged` will now fail** — a first turn is no longer returned unchanged. Update it to assert the `# Today` block plus the bare message. |
| [tests/test_config.py](tests/test_config.py) | Both new settings required, no in-code default. |
| [tests/settings_factory.py](tests/settings_factory.py) | Add both keys to `BASE_ENV`. |

### 7. `README.md`

Update the project-structure block (`services/history.py`, "the three tools") and the design notes
that say the agent exposes two tools.

## Verification

```bash
uv run pytest -q                      # full suite, including the new date-gate tests
uv run ruff check . && uv run ruff format --check .
```

End-to-end against the live plant API:

1. `docker run -d --name mbbr-redis -p 6379:6379 redis:7-alpine`
2. `uv run uvicorn main:app --reload` (wait for `/health` — ASR and TTS load first)
3. `uv run streamlit run streamlit_app.py` and walk the flow by voice/text:
   - «الضغط عامل ايه دلوقتي؟» → «أنهي جهاز؟» → «جهاز 1» → **`get_current_readings`**, not the new tool
   - «وامبارح كان عامل ايه؟» → same device carried over, `get_historical_readings` with
     `from_date == to_date == yesterday`
   - «ومتوسط التدفق الأسبوع اللي فات؟» → one sentence with the average and the min–max
   - «وآخر سنة؟» → `PERIOD_TOO_LONG`
   - «جهاز الطرد المركزي الأسبوع اللي فات؟» → `DEVICE_NOT_FOUND`, no API call
4. Watch the logs to confirm the tool choice and the exact `from`/`to` sent upstream:
   `grep -E 'tool_call_started|history_fetch_started|mbbr_http_get_completed'`

Direct check that the plan's date semantics still hold, independent of the agent:

```bash
curl -s -H "Authorization: Bearer $JWT" \
  "http://41.32.195.157:8083/api/telemetry/daily-averages?device_id=<id>&from=2026-08-16&to=2026-08-16"
```

Note for manual testing: the seeded device `11111111-1111-4111-8111-111111111111` is the only one
carrying telemetry. The real devices (`جهاز 1`, `device 1`, …) are all `offline` with `last_read:
null`, so they will exercise the `NO_HISTORY` path rather than the summary path.
