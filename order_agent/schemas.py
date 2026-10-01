from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from order_agent import normalize

# Field descriptions are part of the prompt: the model sees them in the tool schema.
# coerce_numbers_to_str: the model sometimes emits IDs as numbers (1003 instead of "1003").
LENIENT_IDS = ConfigDict(coerce_numbers_to_str=True)


def _null_to_empty(v):
    # Models often send null for an empty list (our prompts even say "null if missing"). Same meaning; accept it.
    return v or []


class TotalCondition(BaseModel):
    op: Literal[">", ">=", "<", "<=", "=="] = Field(
        description="'over'/'more than' -> '>', 'at least' -> '>=', 'under'/'less than' -> '<', 'at most' -> '<=', 'exactly' -> '=='"
    )
    value: float


class OrderFilter(BaseModel):
    """Structured filter parsed from the user's request."""

    model_config = LENIENT_IDS

    supported: bool = Field(
        description="False if the request is not about finding customer orders, or needs something this filter can't "
                    "express: a condition on a city, date or other field not listed here, negation ('not', 'except'), "
                    "OR across different fields, sorting, top-N, counts or averages."
    )
    states: list[str] | None = Field(None, description="US states as 2-letter codes, e.g. ['OH'].")
    total: list[TotalCondition] = Field(default_factory=list, description="Conditions on order total; all must hold.")
    buyer_name: str | None = Field(None, description="Buyer name or part of it, exactly as the user wrote it.")
    order_ids: list[str] | None = Field(None, description="Specific order IDs, e.g. ['1003'].")
    item_keyword: str | None = Field(None, description="A purchased item the user asked about, e.g. 'monitor'.")
    anomalous_only: bool = Field(
        False, description="True only if the user asks for unusual, suspicious, anomalous or outlier orders."
    )

    _total_null_to_empty = field_validator("total", mode="before")(_null_to_empty)

    # Constraints live in types, not only in descriptions: a violation is a validation error the model must fix.
    @field_validator("states")
    @classmethod
    def _state_codes(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        codes = [normalize.state_code(s) for s in v]
        if None in codes:
            raise ValueError(f"not a US state: {v[codes.index(None)]!r}; use 2-letter codes like 'OH'")
        return codes

    @field_validator("order_ids")
    @classmethod
    def _bare_ids(cls, v: list[str] | None) -> list[str] | None:
        return None if v is None else [normalize.order_id(i) for i in v]  # "#1003" -> "1003"


def _money(v):
    # The model sometimes copies a total with its formatting ("$1,299.99"). Parsing it loosens nothing:
    # grounding still requires the value to be the record's total.
    if isinstance(v, str):
        cleaned = v.replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return v  # let validation reject it
    return v


def _item_list(v):
    # "laptop, hdmi cable" as one string -> two items. Items are grounded against the record afterwards.
    if isinstance(v, str):
        return [i.strip() for i in v.split(",") if i.strip()]
    return v or []


class OrderFields(BaseModel):
    """One order's fields, as copied from a record."""

    model_config = LENIENT_IDS

    orderId: str | None = Field(None, description="Order ID exactly as written in the record, or null if absent.")
    buyer: str | None = Field(None, description="Buyer full name exactly as written, or null if absent.")
    state: str | None = Field(None, description="US state exactly as written in the record (code or name), or null if absent.")
    total: float | None = Field(None, description="Order total as a number without currency symbols, or null if absent.")
    items: list[str] = Field(default_factory=list, description="Purchased items exactly as written.")

    _total_money = field_validator("total", mode="before")(_money)
    _items_list = field_validator("items", mode="before")(_item_list)


class RecordExtraction(OrderFields):
    """The order described by one raw record. A flat object holds exactly one order, so a record can't yield two.
    Kept flat on purpose: nested inside anyOf-null, the model ignored the inner schema (seen live)."""

    is_order: bool = Field(description="False if the record describes no order; then leave every other field null.")

    def fields(self) -> OrderFields | None:
        return OrderFields(**self.model_dump(exclude={"is_order"})) if self.is_order else None


class ExtractedOrder(OrderFields):
    """An extracted order plus the index of its source record. The index is assigned by code, never by the model."""

    orderId: str
    source_index: int


class Order(BaseModel):
    orderId: str
    buyer: str
    state: str
    total: float


SkipReason = Literal[
    "quarantined", "extraction_failed", "no_order_found", "ungrounded", "conflicting_duplicate", "over_budget"
]


class Skipped(BaseModel):
    """A record left out of the answer that no grounded value rules out. Any skip makes the answer partial."""

    record: str  # escaped preview of the raw record
    reason: SkipReason
    orderId: str | None = None
    detail: str | None = None


class OrdersResponse(BaseModel):
    orders: list[Order]
    skipped: list[Skipped] | None = None  # omitted when the answer is complete
    error: str | None = None

    @property
    def complete(self) -> bool:
        return not self.error and not self.skipped


class Score(BaseModel):
    """The price model's view of one validated order. expected=None means it wasn't scored (never a guess)."""

    orderId: str
    expected: float | None
    residual: float | None
    z: float | None
    anomalous: bool
    reason: str | None
