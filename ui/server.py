"""Web UI: `python main.py --ui`. One page plus one JSON endpoint; the agent itself is unchanged."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from langchain_core.language_models import BaseChatModel

from order_agent import config, model
from order_agent.graph import build_graph, response_from_state
from ui.samples import MESSY_RECORDS

logger = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).with_name("index.html")

# Which state key each step's "count" in the pipeline panel reports.
STEP_COUNT_KEY = {
    "fetch_orders": "raw_records",
    "screen_records": "raw_records",
    "extract_orders": "extracted",
    "validate": "validated",
    "score": "scores",
    "apply_filter": "orders",
}


def create_app(llm: BaseChatModel, fetch: Callable) -> Flask:
    app = Flask(__name__)
    graphs = {
        "api": build_graph(llm, fetch, score_all=True),
        "sample": build_graph(llm, fetch=lambda: list(MESSY_RECORDS), score_all=True),
    }

    @app.get("/")
    def index():
        return send_file(INDEX_HTML)

    @app.post("/api/query")
    def query():
        body = request.get_json(silent=True) or {}
        text = str(body.get("query") or "").strip()
        source = body.get("source", "api")
        if not text:
            return jsonify(orders=[], error="Query is required"), 400
        if source not in graphs:
            return jsonify(orders=[], error="Unknown source"), 400
        try:
            return jsonify(run_with_trace(graphs[source], text))
        except Exception:
            run_id = config.RUN_ID.get()
            logger.exception("UI query failed")
            # Exception text can carry internals (URLs, provider messages): log it, don't send it to the browser.
            return jsonify(orders=[], error=f"Internal error (run {run_id}; see server log)"), 500

    return app


def run_with_trace(graph, query: str) -> dict:
    """Run the graph step by step, recording timing and counts per step for the pipeline panel."""
    config.new_run_id()
    state, steps = {}, []
    started = last = time.perf_counter()
    for mode, chunk in graph.stream({"query": query}, stream_mode=["updates", "values"]):
        if mode == "values":  # the full state after each step, as the graph's reducers built it
            state = chunk
            continue
        for node, delta in chunk.items():
            now = time.perf_counter()
            steps.append({
                "name": node,
                "seconds": round(now - last, 2),
                "count": len(delta[STEP_COUNT_KEY[node]]) if STEP_COUNT_KEY.get(node) in delta else None,
                "error": delta.get("error"),
            })
            last = now

    response = response_from_state(state)
    return {
        **response.model_dump(mode="json", exclude_none=True),
        "insights": model.insights(response.orders, state.get("scores", {})),
        "trace": {
            "filter": state["filter"].model_dump(exclude_none=True) if "filter" in state else None,
            "steps": steps,
            "seconds": round(time.perf_counter() - started, 2),
        },
    }
