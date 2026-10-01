import json
import logging
import time

import requests

from order_agent import config

logger = logging.getLogger(__name__)

# The key the current API uses. Anything else goes through the record-list search below, which fails loudly
# when it can't tell which list holds the orders.
KNOWN_KEYS = ("raw_orders",)
OK_STATUSES = {"ok", "success"}


class APIError(Exception):
    pass


def fetch_orders(base_url: str | None = None) -> list[str]:
    """Return every raw order record as text. Lookups by ID happen in code: the API's /api/order/<id> matches by
    substring (/api/order/100 returns order 1001), and no model output ever becomes part of a URL."""
    return extract_records(_get(f"{base_url or config.ORDER_API_URL}/api/orders"))


def _get(url: str) -> dict | list:
    for attempt in range(1, config.HTTP_MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=config.HTTP_TIMEOUT_S)
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == config.HTTP_MAX_ATTEMPTS:
                raise APIError(f"Order API unreachable: {e}") from e
            logger.warning("order API attempt %d failed (%s), retrying", attempt, type(e).__name__)
            time.sleep(0.5 * attempt)
    if resp.status_code != 200:
        raise APIError(f"Order API returned HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError as e:
        raise APIError("Order API returned non-JSON response") from e


def extract_records(payload) -> list[str]:
    """Pull the list of order records out of an API payload, tolerating envelope changes but never guessing."""
    if isinstance(payload, dict):
        status = payload.get("status")
        if isinstance(status, str) and status.casefold() not in OK_STATUSES:
            raise APIError(f"Order API reported status {status!r}")
        for key in KNOWN_KEYS:
            if key in payload:
                return _records(payload[key], key)

    lists = list(_lists(payload, "$"))
    candidates = [(path, items) for path, items in lists if _looks_like_records(items)]
    if len(candidates) > 1:  # prefer the one whose key says "order", else refuse to pick
        named = [(p, items) for p, items in candidates if "order" in p.rsplit(".", 1)[-1].casefold()]
        if len(named) != 1:
            raise APIError(f"Ambiguous API response: several lists could hold orders ({', '.join(p for p, _ in candidates)})")
        candidates = named
    if candidates:
        path, items = candidates[0]
        logger.warning("API response did not use a known key; using the record list at %s", path)
        return [_to_text(v) for v in items]
    empty_orders = [p for p, items in lists if not items and "order" in p.rsplit(".", 1)[-1].casefold()]
    if empty_orders:
        logger.warning("API response did not use a known key; %s is empty", empty_orders[0])
        return []
    raise APIError("Could not find order records in API response")


def _records(value, key: str) -> list[str]:
    if isinstance(value, (str, dict)):
        return [_to_text(value)]
    if isinstance(value, list):
        return [_to_text(v) for v in value]
    raise APIError(f"Order API field {key!r} is {type(value).__name__}, expected a list or a record")


def _lists(node, path: str):
    """Every list in the payload, with a dotted path for error messages."""
    if isinstance(node, list):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _lists(value, f"{path}.{key}")


def _looks_like_records(items: list) -> bool:
    """A non-empty list of dicts, or of strings that could each hold an order (an ID means at least one digit)."""
    return bool(items) and all(
        isinstance(v, dict) or (isinstance(v, str) and len(v) >= 8 and any(c.isdigit() for c in v)) for v in items
    )


def _to_text(item) -> str:
    # ensure_ascii=False: "José" must stay "José", or grounding can't find the name the model returns.
    return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
