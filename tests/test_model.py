"""Price model: offline, trained on synthetic data in milliseconds."""

import pytest

from conftest import extract_all

from order_agent import model, synthetic
from order_agent.schemas import ExtractedOrder


@pytest.fixture(scope="module")
def trained():
    return model.get_model()


def order(items, total=100.0, order_id="1") -> ExtractedOrder:
    return ExtractedOrder(source_index=0, orderId=order_id, buyer="x", state="OH", total=total, items=items)


def test_learned_prices_match_catalog(trained):
    for item, price in trained.card["prices"].items():
        assert price["learned"] == pytest.approx(price["true"], rel=0.05), item


def test_holdout_fit_and_anomaly_detection(trained):
    card = trained.card
    assert card["n_holdout"] == 1000 and card["n_planted"] >= 10
    assert card["r2"] > 0.95
    assert card["recall"] >= 0.85 and card["precision"] >= 0.8
    assert card["recall_over"] >= 0.9  # overpricing is easy; underpricing a cheap item can hide in the noise


def test_detection_holds_across_seeds():
    cards = [model.train(seed).card for seed in range(5)]
    assert min(c["precision"] for c in cards) >= 0.7 and min(c["recall"] for c in cards) >= 0.8


def test_real_dummy_orders_are_not_flagged(raw_orders):
    # A sanity check, not a validation: the catalog was chosen so these orders fit it.
    scores = model.score_orders(extract_all(raw_orders))
    assert len(scores) == 5
    assert not any(s.anomalous for s in scores.values())
    assert all(s.expected is not None for s in scores.values())


def test_overpriced_order_is_flagged():
    s = model.score_orders([order(["Mouse "], total=2450.0, order_id="1006")])["1006"]
    assert s.anomalous and s.z > 3 and s.expected == pytest.approx(50, rel=0.05) and "above" in s.reason


def test_underpriced_order_is_flagged():
    s = model.score_orders([order(["gaming pc"], total=150.0)])["1"]
    assert s.anomalous and s.z < -3 and "below" in s.reason


@pytest.mark.parametrize("items", [["mouse", "mouse"], ["2x mouse"], ["mouse x2"], ["2 mouse"]])
def test_quantities_are_priced(items):
    # Seen in review: ["mouse", "mouse"] at $100 was priced as one mouse and flagged at z=3.4.
    s = model.score_orders([order(items, total=100.0)])["1"]
    assert s.expected == pytest.approx(100, rel=0.05) and not s.anomalous


@pytest.mark.parametrize("item,parsed", [("2x Mouse ", ("mouse", 2)), ("4k monitor", ("4k monitor", 1)),
                                         ("xbox 360", ("xbox 360", 1)), ("monitor", ("monitor", 1))])
def test_parse_item(item, parsed):
    assert model.parse_item(item) == parsed


@pytest.mark.parametrize("items,reason", [(["mouse", "yacht"], "unknown items: yacht"), ([], "no items")])
def test_unscorable_order_gets_no_score(items, reason):
    s = model.score_orders([order(items, total=99999.0)])["1"]
    assert (s.expected, s.residual, s.z, s.anomalous, s.reason) == (None, None, None, False, reason)


def test_training_is_deterministic():
    a, b = model.train(seed=0), model.train(seed=0)
    assert a.regression.coef_.tolist() == b.regression.coef_.tolist() and a.card == b.card
    assert synthetic.generate(seed=1) == synthetic.generate(seed=1)


def test_insights_only_scores_returned_orders(trained):
    scores = model.score_orders([order(["mouse"], total=50.0, order_id="1"), order(["mouse"], total=50.0, order_id="2")])
    out = model.insights([order(["mouse"], order_id="2")], scores)
    assert list(out["scores"]) == ["2"] and len(out["holdout"]) == 1000
    assert {"n_train", "n_holdout", "r2", "mae", "precision", "recall", "prices"} <= out["model"].keys()
