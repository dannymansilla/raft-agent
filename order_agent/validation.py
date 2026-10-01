"""Deterministic checks and post-processing. Pure functions only: no I/O, no LLM."""

import operator
import re
from collections import defaultdict

from order_agent import config
from order_agent.normalize import US_STATES, order_id, preview, screen_text, state_code
from order_agent.schemas import ExtractedOrder, Order, OrderFilter, Score, Skipped

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}
TOTAL_TOLERANCE = 0.005

# ---------------------------------------------------------------- injection screen

# Known prompt-injection phrasings, matched on NFKC text with zero-width characters removed.
# Signature-based: catches common patterns, not every attack. Grounding is the backstop.
INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore (all |any )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
        r"\b(disregard|forget|override|bypass)\b.{0,20}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions|rules|guidance|prompts?)\b",
        r"\bdisregard (all |any )?(previous|prior|above)\b",
        r"\b(system|developer) (note|prompt|message|override)\b",
        r"\byou are now\b",
        r"\bnew instructions\b",
    )
]


def looks_like_injection(text: str) -> bool:
    canonical = screen_text(text)
    return any(p.search(canonical) for p in INJECTION_PATTERNS)


# ---------------------------------------------------------------- what a record mentions
# Each helper answers "where in this record could field X have come from?".

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"


def _labeled(words: str) -> re.Pattern:
    return re.compile(rf"\b(?:{words})\b[^\w\n]{{0,6}}(?:(?:USD|EUR|GBP)[^\w\n]{{0,4}})?({_NUMBER})", re.I)


# Where an order total can come from, most specific first. Only the first kind the record has counts, so
# "Total=$450.00 (list price $650.00)" grounds 450 and "Price=742.10" grounds 742.10.
_TOTAL_SOURCES = [
    _labeled("total|amount|sum|paid|charged"),
    _labeled("price|cost|value"),
    re.compile(rf"[$€£]\s?({_NUMBER})|({_NUMBER})\s?(?:USD|EUR|GBP|dollars?)\b", re.I),
]
# A word label needs a separator ("Order 1001", '"id": "1002"'); "#" doesn't ("#1003"). Case-insensitive label,
# case-sensitive ID, and the ID never touches a word character, so this is always stricter than mentions_id.
_ID_LABEL = r"(?i:\b(?:order|ref|id|po|no)\b[^\w\n]{1,4}|#[^\w\n]{0,3})"
# An ID after an order-type label only ("Order 1001", "Order #1001", "order_id": "1001"): a bare "#200" can be a suite.
_ORDER_REF = re.compile(r"\border\b(?:\s*(?:id|no|number))?[^\w\n]{0,4}(\w*\d[\w-]*)", re.I)


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _label_text(text: str) -> str:
    """Labels match inside keys: 'order_total' and 'orderTotal' both read as 'order total'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text.replace("_", " "))


def amounts(raw: str) -> list[float]:
    """The amounts that could be the order total: those of the most specific kind the record has."""
    text = _label_text(raw)
    for source in _TOTAL_SOURCES:
        found = [_num(next(g for g in m.groups() if g)) for m in source.finditer(text)]
        if found:
            return found
    return []


def total_grounded(raw: str, total: float) -> bool:
    """Never any number: 'Order 1004' is not a total, and a list price next to a total is not the total."""
    return any(abs(n - total) < TOTAL_TOLERANCE for n in amounts(raw))


def mentions_id(raw: str, oid: str) -> bool:
    """Loose: the ID appears as a standalone token anywhere. Used to check the parsed filter against the request."""
    return bool(oid) and re.search(rf"(?<![\w]){re.escape(oid)}(?![\w])", raw) is not None


def id_grounded(raw: str, oid: str) -> bool:
    """Strict: the ID appears right after an ID label ('Order 1001', 'ref #1001', '"order_id": "1002"', '#1003')."""
    pattern = rf"{_ID_LABEL}{re.escape(_label_text(oid))}(?!\w|\.\d)"
    return bool(oid) and re.search(pattern, _label_text(raw)) is not None


def order_count(raw: str) -> int:
    """How many different orders the record names by ID."""
    return len({m[1].casefold() for m in _ORDER_REF.finditer(_label_text(raw))})


_CODES = "|".join(US_STATES)
_END = r"(?=\s*(?:[,)/|;.\"']|\d{5}\b|$))"  # a state code ends a location: "Columbus, OH," "(Cleveland OH)"
_STATE_PATTERNS = [
    re.compile(r"\b(" + "|".join(sorted(US_STATES.values(), key=len, reverse=True)) + r")\b(?!(?<!york)\s+city\b)", re.I),  # "Kansas City" is a city
    re.compile(rf"\b({_CODES})\b{_END}"),                 # "Seattle, WA,"  "(Cleveland OH)"  "OH 43215"
    re.compile(rf"(?<=[a-z] )({_CODES})\b"),              # "Dayton OH Total" (but not "PAID IN FULL")
    re.compile(rf"(?<=,)\s*({_CODES})\b{_END}", re.I),    # "akron, oh"
]


def state_mentions(raw: str) -> list[tuple[str, int, int]]:
    """Every (code, start, end) where the record names a state in a location-like position. Longest name wins,
    so 'West Virginia' is WV (not VA) and 'Arkansas' is AR (not KS)."""
    found = []
    for pattern in _STATE_PATTERNS:
        for m in pattern.finditer(raw):
            found.append((state_code(m[1]), m.start(1), m.end(1)))
    return found


def state_grounded(raw: str, code: str | None, buyer: str | None) -> tuple[bool, str | None]:
    """The state must be named in the record and be the only state named (outside the buyer's name, so a buyer
    called 'Georgia Lee' doesn't count as a second state). Returns (ok, why_not)."""
    mentions = state_mentions(raw)
    if not code or not any(c == code for c, _, _ in mentions):
        return False, None
    buyer_spans = [m.span() for m in re.finditer(re.escape(buyer), raw, re.I)] if buyer else []
    others = {c for c, s, e in mentions if c != code and not any(bs <= s and e <= be for bs, be in buyer_spans)}
    return (False, "record names several states") if others else (True, None)


def mentions_text(raw: str, value: str) -> bool:
    """Case-insensitive, whitespace-normalized substring match."""
    value = " ".join(value.split()).casefold()
    return bool(value) and value in " ".join(raw.split()).casefold()


# ---------------------------------------------------------------- grounding

def ungrounded_fields(order: ExtractedOrder, raw: str) -> list[str]:
    """Fields of `order` that can't be traced to the right place in its source text. Empty list = grounded."""
    bad = []
    if not id_grounded(raw, order_id(order.orderId)):  # ground the ID we output, not the label the model copied
        bad.append("orderId")
    if not order.buyer or not mentions_text(raw, order.buyer):
        bad.append("buyer")
    if order.total is None or not total_grounded(raw, order.total):
        bad.append("total")
    ok, why = state_grounded(raw, state_code(order.state), order.buyer)
    if not ok:
        bad.append(f"state ({why})" if why else "state")
    return bad


def ground(extracted: list[ExtractedOrder], raw_records: list[str],
           f: OrderFilter) -> tuple[list[ExtractedOrder], list[Skipped]]:
    """Keep orders whose fields are grounded in their source record, normalized. Report the rest, unless a field
    that did ground already fails the filter: that order can't be in the answer whatever the others hold."""
    grounded, skipped = [], []
    for order in extracted:
        raw = raw_records[order.source_index]
        normalized = order.model_copy(update={"orderId": order_id(order.orderId), "state": state_code(order.state)})
        if order_count(raw) > 1:  # no value can be attributed to one of them, so nothing rules this record out
            skipped.append(Skipped(record=preview(raw), reason="ungrounded", orderId=order.orderId,
                                   detail="the record names several orders; each needs its own record"))
            continue
        bad = ungrounded_fields(order, raw)
        if not bad:
            grounded.append(normalized.model_copy(update={"items": [i for i in order.items if mentions_text(raw, i)]}))
        elif not ruled_out(normalized, f, unverified={b.split()[0] for b in bad}):
            note = f" (record truncated to {config.MAX_RECORD_CHARS} chars before extraction)" \
                if len(raw) > config.MAX_RECORD_CHARS else ""
            skipped.append(Skipped(record=preview(raw), reason="ungrounded", orderId=order.orderId,
                                   detail="not traceable to the source: " + ", ".join(bad) + note))
    return grounded, skipped


def dedupe(orders: list[ExtractedOrder], raw_records: list[str],
           f: OrderFilter) -> tuple[list[ExtractedOrder], list[Skipped]]:
    """One order per ID. Copies that agree are merged (deterministically); copies that disagree are reported (unless
    they fail the filter), never resolved by whichever record the API happened to return first."""
    by_id: dict[str, list[ExtractedOrder]] = defaultdict(list)
    for o in orders:
        by_id[o.orderId].append(o)
    kept, skipped = [], []
    for oid, copies in by_id.items():
        versions = {(" ".join(c.buyer.split()).casefold(), c.state, c.total) for c in copies}
        if len(versions) == 1:
            kept.append(min(copies, key=lambda c: raw_records[c.source_index]))
            continue
        skipped += [Skipped(record=preview(raw_records[c.source_index]), reason="conflicting_duplicate", orderId=oid,
                            detail=f"{len(copies)} records for order {oid} disagree")
                    for c in copies if not ruled_out(c, f)]
    return kept, skipped


# ---------------------------------------------------------------- filter

# Grounding checks these fields. Items can be missed by the model and unusualness needs a score, so a failed item or
# unusual-only condition never rules an order out: only a grounded value that fails the filter does.
GROUNDED_FIELDS = {"orderId", "buyer", "total", "state"}


def failed_conditions(o: ExtractedOrder, f: OrderFilter, scores: dict[str, Score] | None = None) -> set[str]:
    """The fields on which `o` fails the filter; empty means it matches. A missing value fails.

    `anomalous_only` keeps orders the price model flagged; an unscored order never matches it.
    """
    scores = scores or {}
    failed = set()
    if f.states and o.state not in f.states:
        failed.add("state")
    if f.total and (o.total is None or not all(OPS[c.op](o.total, c.value) for c in f.total)):
        failed.add("total")
    if f.buyer_name and not (o.buyer and mentions_text(o.buyer, f.buyer_name)):
        failed.add("buyer")
    if f.order_ids and o.orderId not in f.order_ids:
        failed.add("orderId")
    if f.item_keyword and not any(mentions_text(i, f.item_keyword) for i in o.items):
        failed.add("items")
    if f.anomalous_only and not (o.orderId in scores and scores[o.orderId].anomalous):
        failed.add("anomalous")
    return failed


def ruled_out(o: ExtractedOrder, f: OrderFilter, unverified: set[str] = frozenset()) -> bool:
    """True if a grounded value already fails the filter, so the order can't be in the answer and isn't reported."""
    return bool(failed_conditions(o, f) & (GROUNDED_FIELDS - unverified))


def apply_filter(orders: list[ExtractedOrder], f: OrderFilter, scores: dict[str, Score] | None = None) -> list[Order]:
    """Filter validated orders and project them to the public schema, sorted by orderId."""
    result = [Order(orderId=o.orderId, buyer=o.buyer, state=o.state, total=o.total)
              for o in orders if not failed_conditions(o, f, scores)]
    return sorted(result, key=lambda o: (len(o.orderId), o.orderId))


# ---------------------------------------------------------------- query checks

_UNSUPPORTED = [
    (re.compile(r"\b(not|except|excluding|other than|besides|without|isn'?t|aren'?t|non)\b", re.I), "negation"),
    # "at least 500" / "at most 500" are comparisons the filter expresses, not rankings.
    (re.compile(r"\b(top|bottom|first|last)\s+\d+\b|\b(largest|biggest|smallest|highest|lowest|cheapest)\b"
                r"|(?<!\bat )\b(most|least)\b", re.I), "ranking"),
    (re.compile(r"\b(sort|sorted|order by|ordered by|rank|ranked)\b", re.I), "sorting"),
    (re.compile(r"\b(how many|count|average|avg|mean|sum of)\b", re.I), "aggregation"),
]
_ANOMALY_WORDS = re.compile(r"\b(unusual|suspicious|anomal\w*|outliers?|odd|weird|strange|irregular|suspect)\b", re.I)
_QUERY_NUMBER = re.compile(rf"({_NUMBER})\s*(k|thousand)?\b", re.I)


def unsupported_reason(query: str) -> str | None:
    """Requests the filter can't express are refused up front instead of answered wrongly."""
    for pattern, kind in _UNSUPPORTED:
        m = pattern.search(query)
        if m:
            return f"{kind} ('{m[0]}') can't be expressed by the order filter"
    return None


def query_numbers(query: str) -> list[float]:
    return [_num(m[1]) * (1000 if m[2] else 1) for m in _QUERY_NUMBER.finditer(query)]


def filter_problems(query: str, f: OrderFilter) -> list[str]:
    """Every value in the parsed filter must come from the request: the model may not add conditions."""
    problems = []
    numbers = query_numbers(query)
    for c in f.total:
        if not any(abs(n - c.value) < TOTAL_TOLERANCE for n in numbers):
            problems.append(f"total {c.op} {c.value:g}")
    for code in f.states or []:  # codes case-sensitive ("in" is a word, "IN" is Indiana); names case-insensitive
        if not re.search(rf"\b{code}\b", query) and not re.search(rf"\b{US_STATES[code]}\b", query, re.I):
            problems.append(f"state {code}")
    if f.buyer_name and not mentions_text(query, f.buyer_name):
        problems.append(f"buyer {f.buyer_name!r}")
    for oid in f.order_ids or []:
        if not mentions_id(query, oid):
            problems.append(f"order {oid}")
    if f.item_keyword and not mentions_text(query, f.item_keyword):
        problems.append(f"item {f.item_keyword!r}")
    if f.anomalous_only and not _ANOMALY_WORDS.search(query):
        problems.append("unusual-only")
    return problems


def truncate(record: str) -> str:
    return record[: config.MAX_RECORD_CHARS]
