"""A labeled eval set: seeded orders in the formats the agent has to read, each with the order it truly describes.

Most records are readable by design. Some are unreadable on purpose (no total, or two states): those must be
reported in "skipped", never answered, and never silently missing.
"""

import json
import random
from typing import NamedTuple

from order_agent.schemas import OrderFields
from order_agent.synthetic import CATALOG

# No buyer name contains a state name, a state code or a label word, so a record can only be misread by the model.
BUYERS = ["John Davis", "Sarah Liu", "Mike Turner", "Rachel Kim", "Chris Myers", "Priya Shah", "Tom Hill", "Ana Lopez",
          "Ben Carter", "Lena Novak", "Omar Haddad", "Grace Park", "Luis Ortega", "Emma Stone", "Noah Brooks",
          "Zoe Adams", "Ethan Cole", "Maya Patel", "Chloe Wu", "Sam Reyes"]
CITIES = [("Columbus", "OH", "Ohio"), ("Cleveland", "OH", "Ohio"), ("Dayton", "OH", "Ohio"), ("Austin", "TX", "Texas"),
          ("Dallas", "TX", "Texas"), ("Seattle", "WA", "Washington"), ("Denver", "CO", "Colorado"),
          ("Miami", "FL", "Florida"), ("Chicago", "IL", "Illinois"), ("Phoenix", "AZ", "Arizona")]


class Labeled(NamedTuple):
    record: str
    truth: OrderFields  # the order the record describes; state as a code
    readable: bool      # False: built so that grounding can't verify it, so it must be reported


def _format(kind: str, oid: str, buyer: str, city: str, code: str, name: str, total: float, items: list[str]) -> str:
    listed = ", ".join(items)
    if kind == "api":
        return f"Order {oid}: Buyer={buyer}, Location={city}, {code}, Total=${total:.2f}, Items: {listed}"
    if kind == "pipe":
        return f"customer: {buyer} | ship to: {city}, {name} | amount: {total:.2f} USD | ref #{oid} | {listed}"
    if kind == "json":
        return json.dumps({"order_id": oid, "buyer": buyer, "city": city, "state": code, "order_total": total,
                           "items": items})
    if kind == "prose":
        return f"#{oid} -- {buyer} ({city} {code}) paid ${total:,.2f} for {' and '.join(items)}"
    if kind == "caps":
        return f"ORDER {oid} / {buyer} / {city.upper()}, {code} / TOTAL {total:.2f} / {listed}"
    raise ValueError(kind)


def labeled_orders(n: int = 40, seed: int = 0) -> list[Labeled]:
    """Deterministic per seed. Every 10th record from the 4th has no total; every 10th from the 8th names two states."""
    rng = random.Random(seed)
    kinds = ["api", "pipe", "json", "prose", "caps"]
    out = []
    for k in range(n):
        oid = str(2001 + k)
        buyer = rng.choice(BUYERS)
        city, code, name = rng.choice(CITIES)
        items = rng.sample(sorted(CATALOG), rng.randint(1, 3))
        total = round(sum(CATALOG[i] for i in items) + rng.gauss(0, 15), 2)
        if k % 10 == 3:
            record = f"Order {oid}: Buyer={buyer}, Location={city}, {code}, Items: {', '.join(items)}"
            out.append(Labeled(record, OrderFields(orderId=oid, buyer=buyer, state=code, items=items), False))
            continue
        if k % 10 == 7:
            bill_city, bill_code, _ = next(c for c in CITIES if c[1] != code)
            record = (f"Order {oid}: Buyer={buyer}, bill to: {bill_city}, {bill_code}, ship to: {city}, {code}, "
                      f"Total=${total:.2f}, Items: {', '.join(items)}")
            out.append(Labeled(record, OrderFields(orderId=oid, buyer=buyer, state=code, total=total, items=items), False))
            continue
        kind = kinds[k % len(kinds)]
        if kind == "caps":
            buyer = buyer.upper()
        record = _format(kind, oid, buyer, city, code, name, total, items)
        out.append(Labeled(record, OrderFields(orderId=oid, buyer=buyer, state=code, total=total, items=items), True))
    return out
