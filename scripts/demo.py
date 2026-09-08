"""Show the four cases the assignment asks for, drawn from the live database.

Nothing here is hardcoded to a document or a figure. The script searches the
stored relations for the *shapes* the brief describes -- a cross-document
corroboration, a genuine contradiction, an apparent contradiction explained by
context, and a claim the grounding pass rejected -- and prints whatever it
finds, with the full derivation and the source evidence.

    python scripts/demo.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factlayer.config import settings   # noqa: E402
from factlayer.db import Store          # noqa: E402
from factlayer.models import Relation   # noqa: E402

MARK = {"pass": "OK ", "fail": "XX ", "info": " . ", "blocked": " ! "}
W = 78


def banner(n: int, title: str, asked: str) -> None:
    print(f"\n{'=' * W}\nCASE {n}: {title}")
    print(f"{'-' * W}\n{textwrap.fill(asked, W)}\n")


def show(rel: Relation, store: Store, docs: dict[str, str]) -> None:
    claims = store.claims_by_id([rel.claim_a_id, rel.claim_b_id])
    a, b = claims.get(rel.claim_a_id), claims.get(rel.claim_b_id)
    if not (a and b):
        print("  (claims missing)")
        return

    for label, c in (("A", a), ("B", b)):
        q = c.qualifiers
        quals = ", ".join(
            f"{k}={v}" for k, v in (
                ("period", q.period.label if q.period else None),
                ("scope", q.scope), ("basis", q.basis),
                ("as_of", q.as_of), ("type", q.estimate_type),
            ) if v
        ) or "no qualifiers stated"
        # Always show the PDF page: it is the one a reader can actually turn to.
        page = f"pdf p.{c.evidence.page_no}"
        if c.evidence.printed_page:
            page += f" (printed {c.evidence.printed_page})"
        print(f"  {label}  {c.measure_raw}  =  {c.display_value()}")
        print(f"     {quals}")
        print(f"     {docs.get(c.doc_id, c.doc_id)} · {page} · grounding={c.evidence.grounding.value}")
        for line in textwrap.wrap(f'"{c.evidence.quote}"', W - 8):
            print(f"       {line}")
        print()

    print("  How the verdict was reached:")
    indent = " " * 26
    for s in rel.trace:
        lines = textwrap.wrap(s.detail, W - 26) or [""]
        print(f"    {MARK[s.outcome]} {s.check:<18} {lines[0]}")
        for extra in lines[1:]:
            print(f"{indent}{extra}")
    print(f"\n  ==> {rel.verdict.value.upper()}   (confidence {rel.confidence})")
    for line in textwrap.wrap(rel.summary, W - 6):
        print(f"      {line}")


def pick(store: Store, verdict: str, *, cross_doc: bool, want_trace: str | None = None):
    """First relation of a verdict, preferring one whose trace shows `want_trace`."""
    pool = store.relations(verdict=verdict, cross_doc_only=cross_doc, limit=200)
    if want_trace:
        for r in pool:
            if any(s.check == want_trace and s.outcome in ("pass", "fail", "info") for s in r.trace):
                return r
    return pool[0] if pool else None


def main() -> int:
    store = Store(settings.db_path)
    docs = {d["doc_id"]: d["filename"] for d in store.documents()}
    if not docs:
        print("Nothing ingested yet. Run: python scripts/ingest.py --all")
        return 1

    s = store.summary()
    print(f"{'=' * W}\nFACT KNOWLEDGE LAYER — the four required cases\n{'=' * W}")
    print(f"{s['documents']} documents · {s['claims_active']} active claims · "
          f"{s['relations']} relations ({s['relations_cross_doc']} cross-document)")
    print(f"verdicts: {s['verdicts']}")

    banner(1, "A fact corroborated across documents, expressed differently",
           "Two documents state the same fact in different words and different units. "
           "The system must recognise them as the same measure and confirm they agree.")
    if (r := pick(store, "corroborates", cross_doc=True) or pick(store, "corroborates", cross_doc=False)):
        show(r, store, docs)
    else:
        print("  none found in the current database")

    banner(2, "A genuine or likely contradiction",
           "Same measure, same period, same scope and basis -- but the values disagree "
           "beyond rounding. Nothing in the qualifiers explains the gap.")
    if (r := pick(store, "contradicts", cross_doc=True) or pick(store, "contradicts", cross_doc=False)):
        show(r, store, docs)
    else:
        print("  none found in the current database")

    banner(3, "An apparent contradiction explained by context",
           "Two figures that look like a conflict, until a qualifier explains them: a "
           "different period, a different reporting scope, or a different basis.")
    shown = 0
    for want in ("period-containment", "scope", "basis"):
        if shown >= 2:
            break
        if (r := pick(store, "complementary", cross_doc=False, want_trace=want)):
            show(r, store, docs)
            shown += 1
            print()
    if not shown:
        print("  none found in the current database")

    banner(4, "An extraction or reasoning failure, and how it was handled",
           "Claims whose evidence did not hold up. These are kept and reported rather "
           "than silently dropped, so the failure is visible and measurable.")
    q = store.quarantined(limit=5)
    g = s["grounding"]
    total = sum(g.values()) or 1
    print(f"  Grounding across the corpus: "
          f"{(total - g.get('not_found', 0)) / total:.1%} of claims verified against source")
    print(f"    exact {g.get('exact', 0)} · normalised {g.get('fuzzy', 0)} · "
          f"elided {g.get('gapped', 0)} · REJECTED {g.get('not_found', 0)}\n")
    print("  Quarantined claims -- quote could not be found on the cited page:")
    for c in q:
        print(f"    · {c.measure_raw[:34]:36s} p{c.evidence.page_no}")
        for line in textwrap.wrap(f'claimed: "{c.evidence.quote}"', W - 8):
            print(f"        {line}")

    dropped: dict[str, int] = {}
    for c in store.all_claims(active_only=False):
        for k, v in c.qualifiers.extra.items():
            if k.startswith("dropped_"):
                key = f"{k.replace('dropped_', '')} = {v!r}"
                dropped[key] = dropped.get(key, 0) + 1
    if dropped:
        print("\n  Qualifiers the model invented, caught by grounding and removed:")
        for name, n in sorted(dropped.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {n:4d} x  {name}")

    print(f"\n{'=' * W}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
