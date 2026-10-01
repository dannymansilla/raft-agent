from unittest.mock import Mock, patch

import pytest
import requests

from order_agent.api_client import APIError, extract_records, fetch_orders

R1 = "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10"
R2 = "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55"


@pytest.mark.parametrize("payload,expected", [
    ({"status": "ok", "raw_orders": [R1, R2]}, [R1, R2]),
    ({"status": "ok", "raw_orders": []}, []),
    ({"status": "ok", "raw_orders": R1}, [R1]),                               # one record instead of a list
    ({"status": "ok", "data": [R1]}, [R1]),                                   # renamed key
    ({"result": {"page": 1, "orders": [R1]}}, [R1]),                          # nested envelope
    ([R1, R2], [R1, R2]),                                                     # bare list
    ({"orders": [{"id": 1, "total": 5}]}, ['{"id": 1, "total": 5}']),         # structured items
    ({"status": "ok", "errors": [], "orders": [R1]}, [R1]),                   # an empty list first must not win
    ({"meta": {"fields": ["id", "buyer"]}, "orders": [R1]}, [R1]),            # field names are not records
    ({"status": "success", "warnings": [], "orders": []}, []),                # genuinely no orders
    ({"tags": [R2], "orders": [R1]}, [R1]),                                   # two candidates: the "orders" key wins
])
def test_extract_records_tolerates_schema_changes(payload, expected):
    assert extract_records(payload) == expected


@pytest.mark.parametrize("payload,match", [
    ({"status": "ok", "message": "maintenance"}, "Could not find"),
    ({"status": "error", "raw_orders": []}, "status 'error'"),
    ({"status": "ok", "raw_orders": None}, "NoneType"),
    ({"status": "ok", "raw_orders": 5}, "int"),
    ({"a": [R1], "b": [R2]}, "Ambiguous"),
])
def test_extract_records_fails_loudly_instead_of_guessing(payload, match):
    with pytest.raises(APIError, match=match):
        extract_records(payload)


def test_structured_records_keep_non_ascii():
    [text] = extract_records({"orders": [{"id": "1007", "buyer": "José García"}]})
    assert "José García" in text


def _response(status=200, json=None, bad_json=False):
    resp = Mock(status_code=status)
    resp.json.side_effect = ValueError if bad_json else (lambda: json)
    return resp


def test_fetch_http_error():
    with patch("requests.get", return_value=_response(500)), pytest.raises(APIError, match="500"):
        fetch_orders()


def test_fetch_non_json():
    with patch("requests.get", return_value=_response(bad_json=True)), pytest.raises(APIError, match="non-JSON"):
        fetch_orders()


def test_fetch_retries_transient_connection_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    flaky = [requests.ConnectionError("refused"), _response(json={"raw_orders": [R1]})]
    with patch("requests.get", side_effect=flaky):
        assert fetch_orders() == [R1]


def test_fetch_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with patch("requests.get", side_effect=requests.ConnectionError("refused")), pytest.raises(APIError, match="unreachable"):
        fetch_orders()
