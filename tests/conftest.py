import re
import sys
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dummy_customer_api import ORDERS  # noqa: E402
from order_agent.schemas import ExtractedOrder, OrderFields, OrderFilter, RecordExtraction  # noqa: E402

RECORD = re.compile(r"Order (\d+): Buyer=([^,]+), Location=[^,]+, (\w\w), Total=\$([\d.,]+), Items: (.*)")


def parse_record(text: str) -> OrderFields | None:
    """Stand-in for the LLM on the dummy API's format."""
    m = RECORD.search(text)
    if not m:
        return None
    oid, buyer, state, total, items = m.groups()
    return OrderFields(orderId=oid, buyer=buyer, state=state, total=float(total.replace(",", "")),
                       items=[x.strip() for x in items.split(",")])


def found(fields: OrderFields) -> RecordExtraction:
    return RecordExtraction(**fields.model_dump(), is_order=True)


NOT_FOUND = RecordExtraction(is_order=False)


def regex_extract(prompt_text: str) -> RecordExtraction:
    """Stand-in for the LLM extractor: one record in, at most one order out."""
    fields = parse_record(prompt_text)
    return found(fields) if fields else NOT_FOUND


def extract_all(records: list[str]) -> list[ExtractedOrder]:
    """What the extraction step produces for these records, with the fake extractor."""
    return [ExtractedOrder(**f.model_dump(), source_index=i)
            for i, r in enumerate(records) if (f := parse_record(r)) is not None]


class FakeLLM:
    """Implements the one method the graph uses: with_structured_output(schema) -> Runnable."""

    def __init__(self, filter: OrderFilter, extract=regex_extract):
        self.filter, self.extract = filter, extract
        self.extract_calls = 0
        self.prompts: list[str] = []

    def with_structured_output(self, schema, **_):
        if schema is OrderFilter:
            return RunnableLambda(lambda _: self.filter)
        return RunnableLambda(self._extract)

    def _extract(self, prompt_value):
        self.extract_calls += 1
        text = prompt_value.to_messages()[-1].content
        self.prompts.append(text)
        return self.extract(text)


@pytest.fixture
def raw_orders() -> list[str]:
    return list(ORDERS)
