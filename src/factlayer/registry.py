"""The measure registry: a canonical vocabulary that grows with the corpus.

Nothing here is seeded with a list of expected measures. The first document
defines the vocabulary, and every document after it either matches an existing
entry or extends the registry. That is what "the documents should guide what
counts as a fact" means in practice, and it is why a new corpus needs no code
changes.

Matching runs in three stages, cheapest first:

1. **Normalised string equality** -- "Revenue from Operations" and "revenue
   from operations" are obviously the same. Free.
2. **Character n-gram similarity** -- proposes a small candidate set from the
   existing registry. Free, and used only for recall.
3. **An LLM adjudication** -- decides whether a candidate really is the same
   measure. Cached, so each distinct question is asked once.

Stage 3 exists because this is exactly where embeddings fail. "Revenue from
operations", "revenue from services" and "total income" are near-identical in
any embedding space and are *genuinely different measures* in a set of
accounts; collapsing them would erase the very distinctions the system exists
to surface. Similarity is allowed to propose; it is not allowed to decide.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient, LLMError

log = logging.getLogger(__name__)

# Words that qualify a measure rather than name it. They are lifted into their
# own fields during extraction, but survive in some measure names anyway.
# NOTE: "total", "net" and "gross" are deliberately NOT in this set. They look
# like qualifiers but in accounting they name distinct line items -- total
# income is not income, and net profit is not gross profit. Stripping them
# merged "Total Income" into "income" and would have silently destroyed exactly
# the distinctions this system exists to surface.
_QUALIFIER_WORDS = {
    "adjusted", "adj", "reported", "restated", "audited", "unaudited",
    "provisional", "estimated", "projected", "normalised", "normalized",
    "consolidated", "standalone",
}
_PERIODISH = re.compile(r"\b(fy\s?\d{2,4}|q[1-4]|h[12]|\b(?:19|20)\d{2})\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SIM_THRESHOLD = 0.45      # candidate proposal only; never decides on its own
_AUTO_ACCEPT = 0.995       # effectively identical strings skip adjudication


def normalise_measure(name: str) -> str:
    """Fold a measure name to a comparable form, without changing its meaning."""
    s = _PERIODISH.sub(" ", (name or "").lower())
    s = _PUNCT.sub(" ", s)
    tokens = [t for t in s.split() if t not in _QUALIFIER_WORDS]
    # Dropping every token would lose the name entirely; keep the original then.
    return " ".join(tokens) or " ".join(_PUNCT.sub(" ", (name or "").lower()).split())


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    padded = f"  {text}  "
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def similarity(a: str, b: str) -> float:
    """Similarity used only to *propose* candidates for adjudication.

    Blends character trigrams (robust to spelling and inflection) with whole-word
    overlap (robust to long shared prefixes). Character n-grams alone scored
    "revenue from operations" against "revenue from services" at 0.41 -- below
    any usable threshold -- because the differing tails are long, so the pair was
    never even offered for a decision. Word overlap catches that case.

    Recall matters here and precision does not: a bad candidate costs one cached
    LLM call, while a missed candidate silently loses a cross-document link.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ga, gb = _char_ngrams(a), _char_ngrams(b)
    char_sim = len(ga & gb) / len(ga | gb) if (ga | gb) else 0.0
    ta, tb = set(a.split()), set(b.split())
    word_sim = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return max(char_sim, word_sim)


ADJUDICATE_SYSTEM = """\
You decide whether two measure names refer to the same underlying quantity. \
You are conservative: near-synonyms that are separate line items in a real \
report are DIFFERENT. Reply with JSON only."""

ADJUDICATE_USER = """\
Do these two names refer to the SAME underlying measured quantity?

  A: "{a}"
  B: "{b}"

Context: both appear in documents about "{domain}".

Return JSON: {{"same": true|false, "confidence": 0.0-1.0, "reason": "one short sentence"}}

Guidance:
- Same quantity worded differently is SAME:
  "turnover" and "revenue from operations"; "profit after tax" and "net profit".
- Quantities that appear as SEPARATE LINE ITEMS in a real report are DIFFERENT,
  even when the words are close:
  "revenue from operations" vs "total income" (total income adds other income);
  "revenue from operations" vs "revenue from services" (one may exclude traded
  goods); "EBITDA" vs "EBIT"; "GDP" vs "GVA"; "headline inflation" vs "core
  inflation".
- A measure and its rate of change are DIFFERENT: "revenue" vs "revenue growth".
- A measure and its ratio are DIFFERENT: "EBITDA" vs "EBITDA margin".
- When genuinely unsure, answer false. A missed link is recoverable; a wrong
  merge silently destroys a real distinction.
"""


@dataclass
class MeasureEntry:
    key: str                                  # canonical key
    canonical_name: str                       # first spelling we saw
    aliases: set[str] = field(default_factory=set)   # normalised forms that map here
    surface_forms: set[str] = field(default_factory=set)  # as written in documents


class MeasureRegistry:
    """A growing map from written measure names to canonical keys."""

    def __init__(self, domain: str = "business and economics") -> None:
        self.domain = domain
        self.entries: dict[str, MeasureEntry] = {}
        self._by_alias: dict[str, str] = {}
        # Adjudications already made, so a re-run never re-asks. Keyed by the
        # unordered pair of normalised names.
        self._decisions: dict[tuple[str, str], bool] = {}

    # -- lookup -----------------------------------------------------------

    def _candidates(self, norm: str, limit: int = 5) -> list[tuple[str, float]]:
        scored = [
            (key, similarity(norm, alias))
            for alias, key in self._by_alias.items()
        ]
        best: dict[str, float] = {}
        for key, score in scored:
            if score > best.get(key, 0.0):
                best[key] = score
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return [(k, s) for k, s in ranked[:limit] if s >= _SIM_THRESHOLD]

    def _add(self, norm: str, surface: str, key: str | None = None) -> str:
        if key is None:
            key = norm.replace(" ", "_")[:60] or "measure"
            suffix = 1
            while key in self.entries and norm not in self.entries[key].aliases:
                suffix += 1
                key = f"{key.rstrip('0123456789_')}_{suffix}"
            self.entries[key] = MeasureEntry(key=key, canonical_name=surface)
        entry = self.entries[key]
        entry.aliases.add(norm)
        entry.surface_forms.add(surface)
        self._by_alias[norm] = key
        return key

    async def resolve(self, name: str, llm: LLMClient | None = None) -> tuple[str, str]:
        """Map a written measure name to a canonical key.

        Returns (key, how) where `how` explains the decision for the trace:
        "exact", "similar", "adjudicated" or "new".
        """
        surface = (name or "").strip()
        norm = normalise_measure(surface)
        if not norm:
            return self._add("unknown measure", surface or "unknown"), "new"

        if (key := self._by_alias.get(norm)) is not None:
            return key, "exact"

        for candidate_key, score in self._candidates(norm):
            entry = self.entries[candidate_key]
            if score >= _AUTO_ACCEPT:
                return self._add(norm, surface, candidate_key), "similar"

            pair = tuple(sorted((norm, entry.canonical_name.lower())))
            if pair in self._decisions:
                if self._decisions[pair]:
                    return self._add(norm, surface, candidate_key), "adjudicated"
                continue

            if llm is None:
                continue
            try:
                data = await llm.json_call(
                    ADJUDICATE_SYSTEM,
                    ADJUDICATE_USER.format(
                        a=surface, b=entry.canonical_name, domain=self.domain
                    ),
                )
                same = bool(data.get("same"))
            except LLMError as exc:
                log.warning("measure adjudication failed (%s vs %s): %s",
                            surface, entry.canonical_name, exc)
                continue
            self._decisions[pair] = same
            if same:
                return self._add(norm, surface, candidate_key), "adjudicated"

        return self._add(norm, surface), "new"

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "entries": {
                k: {
                    "canonical_name": e.canonical_name,
                    "aliases": sorted(e.aliases),
                    "surface_forms": sorted(e.surface_forms),
                }
                for k, e in self.entries.items()
            },
            "decisions": [
                {"a": a, "b": b, "same": same}
                for (a, b), same in self._decisions.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MeasureRegistry":
        reg = cls(domain=data.get("domain", "business and economics"))
        for key, e in (data.get("entries") or {}).items():
            entry = MeasureEntry(
                key=key,
                canonical_name=e["canonical_name"],
                aliases=set(e.get("aliases", [])),
                surface_forms=set(e.get("surface_forms", [])),
            )
            reg.entries[key] = entry
            for alias in entry.aliases:
                reg._by_alias[alias] = key
        for d in data.get("decisions") or []:
            reg._decisions[(d["a"], d["b"])] = d["same"]
        return reg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "MeasureRegistry":
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))

    def stats(self) -> dict[str, int]:
        return {
            "measures": len(self.entries),
            "aliases": len(self._by_alias),
            "adjudications": len(self._decisions),
        }


# Subject (entity) canonicalisation reuses the same similarity machinery, but
# entities are far more forgiving than measures: "Delhivery Limited",
# "Delhivery Ltd" and "the Company" all denote one thing, and merging them
# wrongly is much less costly than merging two measures.
_COMPANY_SUFFIXES = re.compile(
    r"\b(limited|ltd|private|pvt|plc|inc|incorporated|corporation|corp|llp|company|co)\b",
    re.IGNORECASE,
)
_HONORIFICS = re.compile(r"^\s*(mr|mrs|ms|dr|shri|smt|prof)\.?\s+", re.IGNORECASE)


def normalise_subject(name: str) -> str:
    s = _HONORIFICS.sub("", (name or "").strip())
    s = _PUNCT.sub(" ", s.lower())
    s = _COMPANY_SUFFIXES.sub(" ", s)
    return " ".join(s.split()) or (name or "").strip().lower()
