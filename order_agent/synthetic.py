"""Synthetic training data for the price model. Five real records can't train anything, so we generate orders
from a catalog. This data never goes through the LLM.

Known limit: the catalog prices were chosen so the dummy API's 5 orders fit them (e.g. 1001 = laptop + hdmi cable
= 740, actual 742.10). "The real orders aren't flagged" is therefore a sanity check, not a validation."""

import random

CATALOG = {
    "laptop": 700.0, "hdmi cable": 40.0, "headphones": 155.0, "gaming pc": 1250.0,
    "mouse": 50.0, "coffee maker": 90.0, "monitor": 470.0, "desk lamp": 40.0,
}
NOISE_STD = 15.0  # tax, shipping, discounts
ANOMALY_RATE = 0.03  # planted anomalies, half overpriced (x2.5-5), half underpriced (x0.1-0.4)
QUANTITY_RATE = 0.15  # share of line items bought twice, written "2x item"


def generate(n: int = 1000, seed: int = 0) -> tuple[list[tuple[list[str], float]], list[str | None]]:
    """Return (items, total) pairs and each one's planted label: None, "over" or "under". Deterministic per seed."""
    rng = random.Random(seed)
    orders, labels = [], []
    for _ in range(n):
        items, total = [], 0.0
        for name in rng.sample(sorted(CATALOG), rng.randint(1, 3)):
            qty = 2 if rng.random() < QUANTITY_RATE else 1
            items.append(f"{qty}x {name}" if qty > 1 else name)
            total += qty * CATALOG[name]
        total += rng.gauss(0, NOISE_STD)
        label = None
        if rng.random() < ANOMALY_RATE:
            label = rng.choice(["over", "under"])
            total *= rng.uniform(2.5, 5) if label == "over" else rng.uniform(0.1, 0.4)
        orders.append((items, round(total, 2)))
        labels.append(label)
    return orders, labels
