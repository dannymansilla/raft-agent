from conftest import FakeLLM, found, regex_extract

from order_agent.schemas import OrderFields, OrderFilter, TotalCondition
from ui.server import create_app

OHIO_OVER_500 = OrderFilter(supported=True, states=["OH"], total=[TotalCondition(op=">", value=500)])
OHIO_QUERY = "Ohio orders over 500"
STEPS = ["parse_query", "fetch_orders", "screen_records", "extract_orders", "validate", "score", "apply_filter"]
MODEL_CARD_KEYS = {"n_train", "n_holdout", "n_planted", "r2", "mae", "precision", "recall", "prices"}


def client(llm, records):
    return create_app(llm, fetch=lambda: records).test_client()


def post(c, query=OHIO_QUERY, **body):
    return c.post("/api/query", json={"query": query, **body})


def test_query_returns_orders_and_full_trace(raw_orders):
    resp = post(client(FakeLLM(OHIO_OVER_500), raw_orders))
    body = resp.get_json()
    assert resp.status_code == 200
    assert [o["orderId"] for o in body["orders"]] == ["1001", "1003", "1005"]
    assert "skipped" not in body and "error" not in body
    assert [s["name"] for s in body["trace"]["steps"]] == STEPS
    assert [s["count"] for s in body["trace"]["steps"]] == [None, 5, 5, 5, 5, 5, 3]
    assert body["trace"]["filter"]["states"] == ["OH"]


def test_insights_has_model_card_scores_and_holdout(raw_orders):
    insights = post(client(FakeLLM(OHIO_OVER_500), raw_orders)).get_json()["insights"]
    assert MODEL_CARD_KEYS <= insights["model"].keys()
    assert sorted(insights["scores"]) == ["1001", "1003", "1005"]
    assert not any(s["anomalous"] for s in insights["scores"].values())
    assert all(len(p) == 3 for p in insights["holdout"])


def sample_extract(text):
    # regex_extract only reads the dummy format; stand in for the LLM on the sample's other formats.
    known = {
        "John Davis": OrderFields(orderId="ref #1001", buyer="John Davis", state="Ohio", total=742.10, items=["laptop", "hdmi cable"]),
        "Sarah Liu": OrderFields(orderId="1002", buyer="Sarah Liu", state="Texas", total=156.55, items=["headphones"]),
        "Rachel Kim": OrderFields(orderId="1004", buyer="Rachel Kim", state="WA", items=["coffee maker"]),  # no total
        "Mike Turner": OrderFields(orderId="#1003", buyer="Mike Turner", state="OH", total=1299.99, items=["gaming pc", "mouse"]),
        "CHRIS MYERS": OrderFields(orderId="1005", buyer="CHRIS MYERS", state="OH", total=512.00, items=["monitor", "desk lamp"]),
        "Dana Cole": OrderFields(orderId="1006", buyer="Dana Cole", state="TX", total=2450.0, items=["mouse"]),
    }
    for name, fields in known.items():
        if name in text:
            return found(fields)
    return regex_extract(text)


def test_sample_source_flags_the_overpriced_order():
    unusual = OrderFilter(supported=True, anomalous_only=True)
    body = post(client(FakeLLM(unusual, extract=sample_extract), records=[]), "show me unusual orders", source="sample").get_json()
    assert [o["orderId"] for o in body["orders"]] == ["1006"]
    assert body["insights"]["scores"]["1006"]["anomalous"] is True
    assert body["trace"]["filter"]["anomalous_only"] is True


def test_sample_source_reports_what_it_skipped():
    llm = FakeLLM(OHIO_OVER_500, extract=sample_extract)
    body = post(client(llm, records=["should not be used"]), source="sample").get_json()
    assert [o["orderId"] for o in body["orders"]] == ["1001", "1003", "1005"]
    assert not any("should not be used" in p or "SYSTEM NOTE" in p for p in llm.prompts)
    [skip] = body["skipped"]  # 1004 has no total, but it is in WA, so a grounded value rules it out of an Ohio answer
    assert skip["reason"] == "quarantined" and "SYSTEM NOTE" in skip["record"]


def test_unsupported_query_stops_after_parse(raw_orders):
    body = post(client(FakeLLM(OrderFilter(supported=False)), raw_orders), "weather?").get_json()
    assert body["orders"] == [] and "Unsupported" in body["error"]
    assert [s["name"] for s in body["trace"]["steps"]] == ["parse_query"]


def test_ungrounded_extraction_is_explained(raw_orders):
    def hallucinating(text):
        result = regex_extract(text)
        if "1001" in text:
            return found(result.fields().model_copy(update={"orderId": "9999", "buyer": "Evil Corp"}))
        return result

    body = post(client(FakeLLM(OHIO_OVER_500, extract=hallucinating), raw_orders)).get_json()
    [skip] = body["skipped"]
    assert (skip["reason"], skip["orderId"]) == ("ungrounded", "9999")
    assert "orderId" in skip["detail"] and "buyer" in skip["detail"]


def test_internal_errors_are_not_leaked():
    def boom():
        raise RuntimeError("secret upstream detail https://internal.example")

    resp = post(create_app(FakeLLM(OHIO_OVER_500), fetch=boom).test_client())
    assert resp.status_code == 500 and "secret" not in resp.get_json()["error"]


def test_bad_requests_rejected(raw_orders):
    c = client(FakeLLM(OHIO_OVER_500), raw_orders)
    assert c.post("/api/query", json={"query": "  "}).status_code == 400
    assert c.post("/api/query", json={}).status_code == 400
    assert post(c, source="prod-db").status_code == 400


def test_index_served(raw_orders):
    resp = client(FakeLLM(OHIO_OVER_500), raw_orders).get("/")
    assert resp.status_code == 200 and b"Order Agent" in resp.data
