"""Usage:
    python main.py "Show me all orders where the buyer was located in Ohio and total value was over 500."
    python main.py --ui

Exit codes: 0 = complete answer, 2 = partial answer (see "skipped"), 1 = error.
"""

import argparse
import json
import logging
import sys
import threading
import time
from urllib.parse import urlparse

import requests

from order_agent import config
from order_agent.graph import build_graph, run
from order_agent.llm import get_llm
from order_agent.schemas import OrdersResponse

logger = logging.getLogger("main")

DEFAULT_QUERY = "Show me all orders where the buyer was located in Ohio and total value was over 500."
EXIT_OK, EXIT_ERROR, EXIT_PARTIAL = 0, 1, 2


def api_is_up() -> bool:
    try:
        return requests.get(f"{config.ORDER_API_URL}/api/orders?limit=1", timeout=1).ok
    except requests.RequestException:
        return False


def ensure_api() -> None:
    """Start the provided dummy API in a background thread if ORDER_API_URL is local and nothing is listening."""
    if api_is_up():
        return
    url = urlparse(config.ORDER_API_URL)
    if url.hostname not in ("localhost", "127.0.0.1"):
        raise RuntimeError(f"order API at {config.ORDER_API_URL} is not reachable")
    import flask.cli
    from dummy_customer_api import app

    flask.cli.show_server_banner = lambda *_, **__: None  # Flask prints its banner to stdout, which must stay pure JSON
    port = url.port or 80
    logger.info("starting dummy customer API on port %d", port)
    threading.Thread(target=app.run, kwargs={"port": port, "use_reloader": False}, daemon=True).start()
    for _ in range(50):
        if api_is_up():
            return
        time.sleep(0.1)
    raise RuntimeError("dummy customer API failed to start")


def serve_ui() -> int:
    from order_agent.api_client import fetch_orders
    from ui.server import create_app

    ensure_api()
    logger.info("UI running at http://localhost:%d", config.UI_PORT)
    create_app(get_llm(), fetch_orders).run(port=config.UI_PORT, use_reloader=False)
    return EXIT_OK


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natural-language order lookup that returns validated JSON.")
    parser.add_argument("query", nargs="*", help=f'the request (default: "{DEFAULT_QUERY}")')
    parser.add_argument("--ui", action="store_true", help="serve the web UI instead of answering one query")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    config.setup_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.ui:
        return serve_ui()

    query = " ".join(args.query) or DEFAULT_QUERY
    logger.info("query: %s", query)
    try:
        ensure_api()
        response = run(build_graph(get_llm()), query)
    except Exception as e:  # the CLI user is the operator, so the message is shown (the UI hides it)
        logger.exception("agent failed")
        response = OrdersResponse(orders=[], error=f"Internal error: {e}")

    # Exactly {"orders": [...]} when complete; "skipped" or "error" appear only when they apply.
    print(json.dumps(response.model_dump(exclude_none=True), indent=2, ensure_ascii=False))
    if response.error:
        return EXIT_ERROR
    return EXIT_OK if response.complete else EXIT_PARTIAL


if __name__ == "__main__":
    sys.exit(main())
