"""Price model: predicts an order's total from its items and flags totals far from that prediction.

Robust linear regression (Huber loss) on item-count features with no intercept, so each coefficient is the learned
price of one item. It is fit on ALL training orders: in production nobody labels anomalies for you, so the model
must tolerate them. Labels are used only to evaluate. It runs after grounding: it scores validated orders and
never decides which exist.
"""

import functools
import re
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split

from order_agent import config, synthetic
from order_agent.schemas import ExtractedOrder, Order, Score

TRAIN_SIZE = 5000
MAX_QUANTITY = 100
_QTY_BEFORE = re.compile(r"(\d+)\s*(?:x\s*|\s)(.+)")  # "2x mouse", "2 x mouse", "2 mouse" (not "4k monitor")
_QTY_AFTER = re.compile(r"(.+?)\s+x\s*(\d+)")  # "mouse x2", "mouse x 2"


@dataclass
class OrderModel:
    vocab: list[str]  # item names, casefolded; one feature column each
    regression: HuberRegressor
    residual_scale: float  # robust std of training residuals (1.4826 * MAD)
    card: dict  # evaluation on the holdout split, for the README and the UI
    holdout: list[list]  # [actual, expected, flagged] per holdout order, for the UI scatter


def parse_item(item: str) -> tuple[str, int]:
    """'2x Mouse ' -> ('mouse', 2); 'monitor' -> ('monitor', 1)."""
    text = " ".join(item.split()).casefold()
    if m := _QTY_BEFORE.fullmatch(text):
        return m[2].strip(), int(m[1])
    if m := _QTY_AFTER.fullmatch(text):
        return m[1].strip(), int(m[2])
    return text, 1


def _features(items: list[str], vocab: list[str]) -> list[float]:
    counts = dict.fromkeys(vocab, 0.0)
    for item in items:
        name, qty = parse_item(item)
        counts[name] = counts.get(name, 0.0) + qty
    return [counts[v] for v in vocab]


def _robust_scale(residuals: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))


def train(seed: int = 0) -> OrderModel:
    """Fit on all of an 80/20 split of synthetic orders (anomalies included, labels unused); evaluate on the 20%."""
    orders, labels = synthetic.generate(n=TRAIN_SIZE, seed=seed)
    train_split, holdout = train_test_split(list(zip(orders, labels)), test_size=0.2, random_state=seed)

    vocab = [k.casefold() for k in synthetic.CATALOG]
    x_train = np.array([_features(items, vocab) for (items, _), _ in train_split])
    y_train = np.array([total for (_, total), _ in train_split])
    regression = HuberRegressor(fit_intercept=False, alpha=0.0, max_iter=1000).fit(x_train, y_train)
    scale = _robust_scale(y_train - regression.predict(x_train))

    actual = np.array([total for (_, total), _ in holdout])
    expected = regression.predict(np.array([_features(items, vocab) for (items, _), _ in holdout]))
    label = np.array([lab or "" for _, lab in holdout])
    planted, clean = label != "", label == ""
    flagged = np.abs(actual - expected) / scale > config.ANOMALY_Z

    def recall_of(kind: str) -> float | None:
        mask = label == kind
        return round(float(flagged[mask].mean()), 3) if mask.any() else None

    card = {
        "n_train": len(train_split),
        "n_holdout": len(holdout),
        "n_planted": int(planted.sum()),  # planted anomalies in the holdout: precision/recall are measured on these
        "fit_uses_labels": False,
        "r2": round(float(r2_score(actual[clean], expected[clean])), 4),
        "mae": round(float(mean_absolute_error(actual[clean], expected[clean])), 2),
        "precision": round(float(precision_score(planted, flagged, zero_division=0)), 3),
        "recall": round(float(recall_score(planted, flagged, zero_division=0)), 3),
        "recall_over": recall_of("over"),
        "recall_under": recall_of("under"),
        "residual_std": round(scale, 2),
        "anomaly_z": config.ANOMALY_Z,
        "prices": {item: {"learned": round(float(coef), 2), "true": synthetic.CATALOG[item]}
                   for item, coef in zip(vocab, regression.coef_)},
    }
    points = [[float(a), round(float(e), 2), bool(f)] for a, e, f in zip(actual, expected, flagged)]
    return OrderModel(vocab, regression, scale, card, points)


@functools.lru_cache(maxsize=1)
def get_model() -> OrderModel:
    """Trained once per process (well under a second); there is no model file."""
    return train()


def score_order(order: ExtractedOrder, model: OrderModel) -> Score:
    parsed = [parse_item(i) for i in order.items]
    unknown = [i for i, (name, qty) in zip(order.items, parsed) if name not in model.vocab or qty > MAX_QUANTITY]
    if not order.items or unknown:
        # Never a guess: an order we can't price gets no score, like a field grounding can't find.
        reason = f"unknown items: {', '.join(unknown)}" if unknown else "no items"
        return Score(orderId=order.orderId, expected=None, residual=None, z=None, anomalous=False, reason=reason)
    expected = float(model.regression.predict(np.array([_features(order.items, model.vocab)]))[0])
    residual = order.total - expected
    z = residual / model.residual_scale
    anomalous = abs(z) > config.ANOMALY_Z
    reason = f"total is {abs(z):.1f} std devs {'above' if z > 0 else 'below'} expected" if anomalous else None
    return Score(orderId=order.orderId, expected=round(expected, 2), residual=round(residual, 2), z=round(z, 2),
                 anomalous=anomalous, reason=reason)


def score_orders(orders: list[ExtractedOrder]) -> dict[str, Score]:
    model = get_model()
    return {o.orderId: score_order(o, model) for o in orders}


def insights(orders: list[Order], scores: dict[str, Score]) -> dict:
    """For the UI: the model card, the scores of the returned orders and the holdout points for the scatter plot."""
    model = get_model()
    return {"model": model.card, "holdout": model.holdout,
            "scores": {o.orderId: scores[o.orderId].model_dump() for o in orders if o.orderId in scores}}
