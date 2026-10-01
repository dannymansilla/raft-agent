"""Live eval: real model + real dummy API. Run with `pytest -m live` (about 20 LLM calls)."""

import pytest

from labeled_orders import labeled_orders

from order_agent import config
from order_agent.schemas import OrderFilter, TotalCondition

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not config.OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set"),
]

EVAL_SET = [
    ("Show me all orders where the buyer was located in Ohio and total value was over 500.", ["1001", "1003", "1005"]),
    ("orders from Texas", ["1002"]),
    ("orders under $100", ["1004"]),
    ("orders under 200", ["1002", "1004"]),
    ("Ohio orders of at least 512", ["1001", "1003", "1005"]),  # 1005 is exactly 512.00: ">=", not ">"
    ("orders between 100 and 800 dollars", ["1001", "1002", "1005"]),
    ("What did Mike Turner order?", ["1003"]),
    ("show me order 1003", ["1003"]),
    ("anyone who bought a monitor", ["1005"]),
    ("orders over 10,000", []),
    ("show me unusual orders", []),  # none of the real orders is priced far from its items
]


@pytest.fixture(scope="module")
def graph():
    from main import ensure_api
    from order_agent.graph import build_graph
    from order_agent.llm import get_llm

    ensure_api()
    return build_graph(get_llm())


@pytest.mark.parametrize("query,expected", EVAL_SET)
def test_eval(graph, query, expected):
    from order_agent.graph import run

    result = run(graph, query)
    assert result.error is None and result.complete, result
    assert [o.orderId for o in result.orders] == expected


@pytest.mark.parametrize("query", ["what's the weather in Columbus?", "orders not from Ohio",
                                   "orders from Ohio or over 1000 dollars"])
def test_unsupported_query(graph, query):
    from order_agent.graph import run

    result = run(graph, query)
    assert result.orders == [] and result.error


def test_city_is_not_turned_into_a_state(graph):
    from order_agent.graph import run

    result = run(graph, "orders shipped to Columbus")  # the filter can't express cities
    assert result.error or result.orders == []


def test_deterministic_across_runs(graph):
    from order_agent.graph import run

    outputs = {run(graph, EVAL_SET[0][0]).model_dump_json() for _ in range(3)}
    assert len(outputs) == 1


# Edge cases from the brief, against the real model, with controlled records.

OHIO_OVER_500 = EVAL_SET[0][0]


def run_with_records(records: list[str], query: str = OHIO_OVER_500):
    from order_agent.graph import build_graph, run
    from order_agent.llm import get_llm

    return run(build_graph(get_llm(), fetch=lambda: records), query)


def ids(result) -> list[str]:
    return [o.orderId for o in result.orders]


def test_changed_text_format():
    records = [
        "customer: John Davis | ship to: Columbus, Ohio | amount: 742.10 USD | ref #1001 | laptop",
        '{"id": "1002", "client": "Sarah Liu", "addr": "Austin, Texas", "sum": "$156.55"}',
        "#1003 -- Mike Turner (Cleveland OH) paid $1,299.99 for a gaming pc",
    ]
    result = run_with_records(records)
    assert ids(result) == ["1001", "1003"]
    assert result.orders[1].total == 1299.99


def test_missing_total_is_not_invented_and_is_reported():
    records = ["Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop",
               "Order 1007: Buyer=Ann Lee, Location=Dayton, OH, Items: monitor",
               "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc"]
    result = run_with_records(records, "orders from Ohio")
    assert ids(result) == ["1001", "1003"]
    # 1007 has no total: the model must return null (-> ungrounded) or no order, never a number from the text.
    assert [s.reason for s in result.skipped] in (["ungrounded"], ["no_order_found"])


def test_structured_records_with_renamed_keys():
    # The API switches to JSON records whose keys aren't the usual labels.
    records = [
        '{"order_id": "1001", "buyer": "John Davis", "city": "Columbus", "state": "OH", "order_total": 742.10}',
        '{"order_id": "1002", "buyer": "Sarah Liu", "city": "Austin", "state": "TX", "order_total": 156.55}',
        '{"orderId": "1003", "buyer": "Mike Turner", "city": "Cleveland", "state": "OH", "orderTotal": 1299.99}',
        '{"orderId": "1005", "buyer": "Chris Myers", "city": "Cincinnati", "state": "OH", "orderTotal": 512.00}',
    ]
    result = run_with_records(records)
    assert ids(result) == ["1001", "1003", "1005"] and result.complete, result


def test_list_price_is_not_taken_as_the_total():
    records = ["Order 1008: Buyer=Tom Hill, Location=Toledo, OH, Total=$450.00 (list price $650.00), Items: monitor"]
    result = run_with_records(records)
    assert result.orders == []


def test_messy_sample_end_to_end():
    # The UI's "Messy sample": format drift, a missing total (1004), an overpriced TX order (1006) and an injection (9999).
    from ui.samples import MESSY_RECORDS

    result = run_with_records(list(MESSY_RECORDS))
    assert ids(result) == ["1001", "1003", "1005"]
    assert [s.reason for s in result.skipped] == ["quarantined"]  # the injected "Ohio, $50000" record


def test_messy_sample_unusual_orders():
    # The price model flags 1006 ($2,450 for a mouse); the LLM only has to parse "unusual" into anomalous_only.
    from ui.samples import MESSY_RECORDS

    assert ids(run_with_records(list(MESSY_RECORDS), "show me unusual orders")) == ["1006"]


# The labeled set: 40 seeded records in five formats, 8 of them unreadable on purpose. The agent may skip a record,
# but it may never answer wrongly or leave out a matching order without reporting it.

LABELED_QUERIES = [
    (OHIO_OVER_500, OrderFilter(supported=True, states=["OH"], total=[TotalCondition(op=">", value=500)])),
    ("orders from Texas", OrderFilter(supported=True, states=["TX"])),
    ("orders under 300", OrderFilter(supported=True, total=[TotalCondition(op="<", value=300)])),
]


@pytest.mark.parametrize("query,f", LABELED_QUERIES)
def test_labeled_set(query, f):
    from order_agent.normalize import preview
    from order_agent.validation import failed_conditions

    cases = labeled_orders()
    result = run_with_records([c.record for c in cases], query)
    assert result.error is None, result.error
    truth = {c.truth.orderId: c.truth for c in cases}
    reported = {s.record for s in result.skipped or []}
    matching = [c for c in cases if not failed_conditions(c.truth, f)]

    wrong = [o.orderId for o in result.orders
             if o.orderId not in truth or failed_conditions(truth[o.orderId], f)
             or (o.buyer.casefold(), o.state, o.total) != (truth[o.orderId].buyer.casefold(), truth[o.orderId].state,
                                                           truth[o.orderId].total)]
    answered = [c for c in matching if c.truth.orderId in {o.orderId for o in result.orders}]
    missing = [c.truth.orderId for c in matching if c not in answered and preview(c.record) not in reported]
    readable_skipped = [c.truth.orderId for c in matching if c.readable and c not in answered]
    print(f"\n{query!r}: {len(matching)} matching, {len(answered)} answered, {len(matching) - len(answered)} reported "
          f"({len(readable_skipped)} of them readable), {len(missing)} missing, {len(wrong)} wrong")
    assert not wrong and not missing, (wrong, missing)
