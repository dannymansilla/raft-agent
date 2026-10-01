"""The order agent as a linear LangGraph graph:
parse_query -> fetch_orders -> screen_records -> extract_orders -> validate -> score -> apply_filter."""

import logging
import operator
import time
from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from order_agent import api_client, config, prompts, validation
from order_agent.normalize import preview
from order_agent.schemas import (
    ExtractedOrder, Order, OrderFilter, OrdersResponse, RecordExtraction, Score, Skipped,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    query: str
    filter: OrderFilter
    raw_records: list[str]
    extracted: list[ExtractedOrder]
    validated: list[ExtractedOrder]
    scores: dict[str, Score]
    orders: list[Order]
    skipped: Annotated[list[Skipped], operator.add]  # every node may add; nothing is dropped silently
    error: str


def _structured(llm: BaseChatModel, system: str, schema: type) -> RunnableLambda:
    """Schema-bound call. If the output fails validation, retry once and tell the model what was wrong:
    re-sending the identical prompt at temperature 0 tends to fail the same way. Network errors are retried
    by the HTTP client, not here."""
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}{feedback}")])
    chain = prompt | llm.with_structured_output(schema, method="function_calling")

    def call(text: str, feedback: str = ""):
        result = chain.invoke({"input": text, "feedback": feedback})
        if result is None:
            raise OutputParserException(f"no {schema.__name__} tool call in the response")
        return result

    def invoke(text: str):
        try:
            return call(text)
        except (OutputParserException, ValidationError) as e:
            error = " ".join(str(e).split())[:400]
            logger.warning("%s rejected by schema, retrying with the error: %s", schema.__name__, error)
            return call(text, f"\n\n(Your previous answer was rejected: {error}. Answer again, following the schema exactly.)")

    return RunnableLambda(invoke)


def _tokens(usage: dict) -> str:
    """Summarize LLM token usage for a step's log line, e.g. 'tokens in=812 out=640 (reasoning=512) '."""
    if not usage:
        return ""
    models = usage.values()
    tokens_in = sum(u.get("input_tokens", 0) for u in models)
    tokens_out = sum(u.get("output_tokens", 0) for u in models)
    reasoning = sum(u.get("output_token_details", {}).get("reasoning", 0) for u in models)
    return f"tokens in={tokens_in} out={tokens_out} (reasoning={reasoning}) "


def _timed(name: str, fn: Callable[[AgentState], AgentState]) -> Callable[[AgentState], AgentState]:
    def node(state: AgentState) -> AgentState:
        start = time.perf_counter()
        with get_usage_metadata_callback() as usage:
            update = fn(state)
        logger.info("%-15s %.2fs %s%s", name, time.perf_counter() - start, _tokens(usage.usage_metadata),
                    update.get("error", ""))
        return update

    return node


def _skip(record: str, reason: str, detail: str | None = None, order_id: str | None = None) -> Skipped:
    return Skipped(record=preview(record), reason=reason, orderId=order_id, detail=detail)


def build_graph(
    llm: BaseChatModel,
    fetch: Callable[[], list[str]] = api_client.fetch_orders,
    score_all: bool = False,
):
    """`score_all` runs the price model on every query (the UI shows the scores); otherwise it runs only when the
    user asks for unusual orders."""
    parse_chain = _structured(llm, prompts.PARSE_QUERY_SYSTEM, OrderFilter)
    extract_chain = _structured(llm, prompts.EXTRACT_SYSTEM, RecordExtraction)

    def parse_query(state: AgentState) -> AgentState:
        query = state["query"].strip()
        if not query or len(query) > config.MAX_QUERY_CHARS:
            return {"error": f"Query must be 1-{config.MAX_QUERY_CHARS} characters"}
        reason = validation.unsupported_reason(query)
        if reason:
            return {"error": f"Unsupported request: {reason}"}
        f = parse_chain.invoke(query)
        logger.info("filter: %s", f.model_dump(exclude_none=True))
        if not f.supported:
            return {"filter": f, "error": "Unsupported request: only order lookups the filter can express are supported"}
        problems = validation.filter_problems(query, f)
        if problems:  # the model added a condition the user never stated
            return {"filter": f, "error": f"Could not verify the parsed filter against the request: {'; '.join(problems)}"}
        return {"filter": f}

    def fetch_orders(state: AgentState) -> AgentState:
        try:
            records = fetch()
        except api_client.APIError as e:
            return {"error": str(e)}
        logger.info("fetched %d raw records", len(records))
        return {"raw_records": sorted(records)}  # the API's order must never influence the answer

    def screen_records(state: AgentState) -> AgentState:
        # Deterministic: flagged records never reach the LLM, and are reported, never silently dropped.
        kept, skipped = [], []
        for record in state["raw_records"]:
            if validation.looks_like_injection(record):
                logger.warning("quarantined possible prompt injection: %r", preview(record))
                skipped.append(_skip(record, "quarantined", "matches a known prompt-injection pattern"))
            else:
                kept.append(record)
        return {"raw_records": kept, "skipped": skipped}

    def extract_orders(state: AgentState) -> AgentState:
        records = state["raw_records"]
        n = min(len(records), config.MAX_LLM_CALLS)
        skipped = [_skip(r, "over_budget", f"more than {config.MAX_LLM_CALLS} records needed the LLM")
                   for r in records[n:]]
        results = extract_chain.batch([validation.truncate(r) for r in records[:n]],
                                      config={"max_concurrency": config.LLM_MAX_CONCURRENCY}, return_exceptions=True)
        extracted, failures = [], 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures += 1
                logger.error("extraction failed for record %d: %s", i, result)
                skipped.append(_skip(records[i], "extraction_failed", f"{type(result).__name__}: {str(result)[:200]}"))
            elif (fields := result.fields()) is None:
                skipped.append(_skip(records[i], "no_order_found", "the model found no order in this record"))
            else:  # the index comes from our loop, never from the model; a missing ID fails grounding
                extracted.append(ExtractedOrder(**fields.model_dump(exclude={"orderId"}), orderId=fields.orderId or "",
                                                source_index=i))
        if n and failures == n:
            return {"error": f"Extraction failed for all {n} records: {results[0]}"}
        logger.info("extracted %d orders (%d LLM calls, %d failed, %d over budget)",
                    len(extracted), n, failures, len(records) - n)
        return {"extracted": extracted, "skipped": skipped}

    def validate(state: AgentState) -> AgentState:
        records, f = state["raw_records"], state["filter"]
        grounded, ungrounded = validation.ground(state["extracted"], records, f)
        valid, conflicts = validation.dedupe(grounded, records, f)
        for s in ungrounded + conflicts:
            logger.warning("skipped order %s: %s: %s", s.orderId, s.reason, s.detail)
        logger.info("%d of %d orders passed validation", len(valid), len(state["extracted"]))
        return {"validated": valid, "skipped": ungrounded + conflicts}

    def score(state: AgentState) -> AgentState:
        # Deterministic, no LLM: the price model scores validated orders only; it never decides which orders exist.
        if not (score_all or state["filter"].anomalous_only):
            return {"scores": {}}
        from order_agent import model  # scikit-learn loads only when scores are needed

        scores = model.score_orders(state["validated"])
        logger.info("%d orders scored, %d flagged unusual", len(scores), sum(s.anomalous for s in scores.values()))
        return {"scores": scores}

    def apply_filter(state: AgentState) -> AgentState:
        orders = validation.apply_filter(state["validated"], state["filter"], state["scores"])
        logger.info("%d orders matched filter", len(orders))
        return {"orders": orders}

    def stop_on_error(state: AgentState) -> str:
        return "end" if state.get("error") else "next"

    steps = [parse_query, fetch_orders, screen_records, extract_orders, validate, score, apply_filter]
    graph = StateGraph(AgentState)
    for step in steps:
        graph.add_node(step.__name__, _timed(step.__name__, step))
    graph.add_edge(START, steps[0].__name__)
    for current, nxt in zip(steps, steps[1:]):
        graph.add_conditional_edges(current.__name__, stop_on_error, {"next": nxt.__name__, "end": END})
    graph.add_edge(steps[-1].__name__, END)
    return graph.compile()


def response_from_state(state: AgentState) -> OrdersResponse:
    return OrdersResponse(orders=state.get("orders", []), skipped=state.get("skipped") or None, error=state.get("error"))


def run(graph, query: str) -> OrdersResponse:
    config.new_run_id()
    return response_from_state(graph.invoke({"query": query}))
