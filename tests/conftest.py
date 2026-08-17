"""Undo CrewAI's import-time `.env` loading so tests stay isolated from it.

`crewai/llm.py` calls `load_dotenv()` at module scope, with no way to opt out.
That copies the developer's real `.env` into `os.environ`, and pydantic-settings
ranks the process environment above an `env_file` — so `ExampleSettings` would
read the live values instead of `.env.example`, and a key deliberately left out
of a test would be satisfied from the ambient environment instead of failing.

Importing crewai here forces that side effect to happen once, at collection,
where it can be cleaned up before any test builds a Settings object.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

import agent  # noqa: F401  -- switches CrewAI telemetry off before it is imported
import crewai  # noqa: F401  -- calls load_dotenv() at import time

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

for _key in dotenv_values(_PROJECT_ROOT / ".env"):
    os.environ.pop(_key, None)
