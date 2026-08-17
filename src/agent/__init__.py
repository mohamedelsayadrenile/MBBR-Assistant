import os

# CrewAI ships OTLP telemetry and trace collection switched on, and both read the
# real process environment at import time -- a key in .env is loaded too late to
# matter. setdefault so an operator can still turn them back on from outside.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
