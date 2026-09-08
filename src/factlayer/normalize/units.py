"""Unit parsing and canonicalisation.

Indian financial and statistical writing mixes scales freely: the same figure
appears as "Rs. 8,142 Cr" in a results deck and "81,415.38 million" in the
annual report. Those are the same number. Recognising that is the difference
between reporting a corroboration and reporting a contradiction.

Everything here is a pure function over strings -- no model calls -- so it is
cheap to test exhaustively, and when it cannot recognise a unit it says so
rather than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Multiplicative scale words. Indian scales (lakh, crore) sit alongside the
# international ones because filings use both, sometimes on the same page.
SCALES: dict[str, float] = {
    "hundred": 1e2,
    "thousand": 1e3,
    "k": 1e3,
    "lakh": 1e5,
    "lac": 1e5,
    "lakhs": 1e5,
    "million": 1e6,
    "mn": 1e6,
    "mln": 1e6,
    "m": 1e6,
    "crore": 1e7,
    "crores": 1e7,
    "cr": 1e7,
    "billion": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
}

CURRENCIES: dict[str, str] = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupees": "INR", "rupee": "INR",
    "$": "USD", "usd": "USD", "us$": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP",
}

_PERCENT = re.compile(r"(%|\bper\s?cent\b|\bpercent\b|\bpercentage\b)", re.IGNORECASE)
_BPS = re.compile(r"\b(bps|basis points?)\b", re.IGNORECASE)
_MASS = re.compile(r"\b(tons?|tonnes?|mt)\b", re.IGNORECASE)
_KG = re.compile(r"\bkg\b|\bkilograms?\b", re.IGNORECASE)
# Countable nouns that appear as units in these corpora.
_COUNT = re.compile(
    r"\b(shipments?|parcels?|units?|customers?|employees?|people|persons?|"
    r"consignments?|orders?|clients?|stores?|centres?|centers?|vehicles?)\b",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-z₹$€£]+\.?|%", re.IGNORECASE)


@dataclass(frozen=True)
class UnitSpec:
    """A parsed unit: what dimension it measures, and how to reach the base.

    `scale` converts the written number into `base`. "Rs. Cr" is
    (currency, INR, 1e7, "INR"), so 8,142 becomes 81,420,000,000 rupees.
    """

    dimension: str          # currency | percent | count | mass | ratio
    scale: float
    base: str               # INR | USD | percent | items | tonnes
    currency: str | None = None
    raw: str = ""

    def comparable_with(self, other: "UnitSpec") -> bool:
        """Same dimension and same base -- otherwise the values are not commensurable.

        Currencies are never converted. Exchange rates move, filings rarely state
        the rate they used, and inventing one would manufacture agreement or
        disagreement that is not in the documents.
        """
        return self.dimension == other.dimension and self.base == other.base


def parse_unit(raw: str | None) -> UnitSpec | None:
    """Parse a written unit into a UnitSpec, or None if unrecognised.

    Returning None matters: an unrecognised unit makes two claims
    non-comparable, which the reconciler surfaces as UNDERSPECIFIED rather than
    silently assuming they share a scale.
    """
    if not raw:
        return None
    text = raw.strip().lower()
    if not text:
        return None

    # Basis points are a percentage in disguise; 100 bps == 1 per cent.
    if _BPS.search(text):
        return UnitSpec("percent", 0.01, "percent", raw=raw)
    if _PERCENT.search(text):
        return UnitSpec("percent", 1.0, "percent", raw=raw)

    tokens = [t.strip(".") for t in _TOKEN.findall(text)]
    # A currency symbol may be glued to the number ("₹8,142"), so also scan raw.
    currency = next(
        (CURRENCIES[t] for t in tokens if t in CURRENCIES),
        next((c for sym, c in CURRENCIES.items() if sym in text and not sym.isalpha()), None),
    )
    scale = next((SCALES[t] for t in tokens if t in SCALES), 1.0)

    if currency:
        return UnitSpec("currency", scale, currency, currency=currency, raw=raw)
    if _MASS.search(text):
        return UnitSpec("mass", scale, "tonnes", raw=raw)
    if _KG.search(text):
        return UnitSpec("mass", scale * 0.001, "tonnes", raw=raw)
    if _COUNT.search(text):
        return UnitSpec("count", scale, "items", raw=raw)
    # A bare scale word with no noun ("Mn", "Cr") is a count of something the
    # measure name will have to explain.
    if any(t in SCALES for t in tokens):
        return UnitSpec("count", scale, "items", raw=raw)
    return None


def canonicalise(value: float, unit: UnitSpec | None) -> float | None:
    """Rescale a written value into its dimension's base unit."""
    if unit is None:
        return None
    return value * unit.scale
