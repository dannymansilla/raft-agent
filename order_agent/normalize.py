"""Normalizers shared by the schemas (to constrain what the model may return) and by grounding. Pure functions."""

import re
import unicodedata

US_STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "DC": "district of columbia",
    "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho", "IL": "illinois",
    "IN": "indiana", "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana",
    "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan", "MN": "minnesota",
    "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon",
    "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina", "SD": "south dakota",
    "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia",
    "WA": "washington", "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming",
}
_NAME_TO_CODE = {name: code for code, name in US_STATES.items()}

# Labels the model copies along with an ID: "#1001", "ref #1001", "Order 1001", "Order #: 1001", "PO 1001".
_ID_LABELS = re.compile(r"^(?:(?:order|ref|id|po|no|number)\b\.?[\s:=#-]*|#\s*)+", re.IGNORECASE)


def _exact_state(v: str) -> str | None:
    return v.upper() if v.upper() in US_STATES else _NAME_TO_CODE.get(v.casefold())


def state_code(value: str | None) -> str | None:
    """'OH', 'oh', 'Ohio', ' ohio ' -> 'OH'. A location copied whole ends with its state:
    'Cleveland OH', 'Columbus, Ohio', 'Kansas City, MO' -> the state. Anything else -> None."""
    if not value:
        return None
    v = " ".join(value.split())
    if code := _exact_state(v):
        return code
    parts = [p for p in re.split(r"[,\s]+", v) if p]
    for n in (3, 2, 1):  # longest tail first: "district of columbia", "new hampshire"
        if len(parts) > n and (code := _exact_state(" ".join(parts[-n:]))):
            return code
    return None


def order_id(value: str) -> str:
    """Strip copied labels and trailing notes: 'ref #1001', 'Order 1001', '#1001 (rush)' -> '1001'."""
    rest = _ID_LABELS.sub("", value.strip()).split()
    return rest[0].strip(".,:;()[]{}\"'") if rest else ""


def screen_text(text: str) -> str:
    """Canonical form for pattern matching: NFKC, and invisible format characters (zero-width etc.) removed."""
    return "".join(c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c) != "Cf")


def preview(text: str, limit: int = 120) -> str:
    """One-line, printable excerpt of untrusted text, safe for logs and output."""
    flat = " ".join(screen_text(text).split())
    flat = "".join(c if c.isprintable() else "?" for c in flat)
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
