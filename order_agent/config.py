import contextvars
import logging
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Any OpenAI-compatible endpoint: gpt-oss is open-weight, so controlled data can go to a self-hosted server instead.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:exacto")
ORDER_API_URL = os.getenv("ORDER_API_URL", "http://localhost:5001")
# gpt-oss reasoning effort: "low" | "medium" | "high" ("" = provider default). Copying fields needs little reasoning.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")
# OpenRouter providers to skip, comma-separated. CoreWeave returned finish_reason=error with no tool call on every
# tool-calling request (October 2026), while DeepInfra passed the same requests.
IGNORE_PROVIDERS = [p.strip() for p in os.getenv("OPENROUTER_IGNORE_PROVIDERS", "CoreWeave").split(",") if p.strip()]
UI_PORT = int(os.getenv("UI_PORT", "8000"))

# Context-window and cost guards. Every LLM call sees one query or one record, so input size is fixed;
# MAX_LLM_CALLS bounds how many records one query may send to the model (the rest are reported as skipped).
MAX_RECORD_CHARS = 2000
MAX_QUERY_CHARS = 1000
MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALLS", "200"))

LLM_TIMEOUT_S = 60
LLM_TRANSPORT_RETRIES = 2  # network/5xx/429 retries, done by the OpenAI client
LLM_MAX_CONCURRENCY = 8
HTTP_TIMEOUT_S = 10
HTTP_MAX_ATTEMPTS = 3

# Price model: flag an order when its total is more than this many robust std devs from the expected total.
ANOMALY_Z = 3.0

# Every log line carries the id of the query it belongs to, including lines from parallel LLM calls.
RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")


def new_run_id() -> str:
    run_id = uuid.uuid4().hex[:8]
    RUN_ID.set(run_id)
    return run_id


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = RUN_ID.get()
        return True


def setup_logging() -> None:
    # Logs go to stderr so stdout stays clean JSON.
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s [%(run_id)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_RunIdFilter())
    for noisy in ("httpx", "httpx2", "openai", "urllib3", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
