"""Hunt for anomalies in the knowledge layer, including our own failures.

This is deliberately adversarial towards the system that produced the data. It
looks for the shapes that indicate the pipeline got something wrong -- verdicts
that should not be possible, claims whose evidence does not support them,
measures that were merged when they should not have been -- rather than for
pleasing examples.

    python scripts/audit.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factlayer.config import settings          # noqa: E402
from factlayer.db import Store                 # noqa: E402
from factlayer.models import ValueKind         # noqa: E402
from factlayer.registry import MeasureRegistry  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    store = Store(settings.db_path)
    claims = store.all_claims(active_only=False)
    by_id = {c.id: c for c in claims}
    if not claims:
        print("No claims stored. Run: python scripts/ingest.py --all")
        return 1

    rule("1. Coverage")
    print(f"claims          : {len(claims)}")
    print(f"active          : {sum(c.status.value == 'active' for c in claims)}")
    print(f"quarantined     : {sum(c.status.value == 'quarantined' for c in claims)}")
    print("grounding       :", dict(Counter(c.evidence.grounding.value for c in claims)))
    print("value kinds     :", dict(Counter(c.value_kind.value for c in claims)))

    rule("2. Qualifier completeness (drives UNDERSPECIFIED verdicts)")
    active = [c for c in claims if c.status.value == "active"]
    missing = {
        "period":  sum(c.qualifiers.period is None for c in active),
        "period unresolved": sum(
            c.qualifiers.period is not None and not c.qualifiers.period.is_resolved
            for c in active),
        "scope":   sum(c.qualifiers.scope is None for c in active),
        "basis":   sum(c.qualifiers.basis is None for c in active),
        "as_of":   sum(c.qualifiers.as_of is None for c in active),
    }
    for k, v in missing.items():
        print(f"  missing {k:20s} {v:5d} / {len(active)}  ({v / max(len(active),1):.0%})")

    rule("3. Qualifiers the model invented and grounding removed")
    dropped = Counter()
    for c in claims:
        for k, v in c.qualifiers.extra.items():
            if k.startswith("dropped_"):
                dropped[f"{k.replace('dropped_','')}={v}"] += 1
    for name, n in dropped.most_common(12):
        print(f"  {n:5d}  {name}")
    if not dropped:
        print("  none")

    rule("4. Unparsed period labels (each one blocks a comparison)")
    bad = Counter(
        c.qualifiers.period.label
        for c in active
        if c.qualifiers.period is not None and not c.qualifiers.period.is_resolved
    )
    for label, n in bad.most_common(15):
        print(f"  {n:5d}  {label!r}")
    if not bad:
        print("  none")

    rule("5. Units the parser did not recognise (blocks value comparison)")
    unknown = Counter(
        c.quantity.unit_raw or "(none)"
        for c in active
        if c.quantity is not None and not c.quantity.is_canonical
    )
    for unit, n in unknown.most_common(15):
        print(f"  {n:5d}  {unit!r}")
    if not unknown:
        print("  none")

    rule("6. Suspiciously generic measure names (lost table row labels)")
    per_measure = defaultdict(set)
    for c in active:
        per_measure[c.measure_key].add(round(c.quantity.canonical_low, 4)
                                       if c.quantity and c.quantity.canonical_low is not None
                                       else None)
    generic = sorted(
        ((k, len(v)) for k, v in per_measure.items() if k and len(k) < 22 and len(v) > 12),
        key=lambda kv: -kv[1],
    )
    for key, n in generic[:12]:
        print(f"  {n:5d} distinct values under measure {key!r}")
    if not generic:
        print("  none")

    rule("7. Registry entries with many surface forms (possible over-merge)")
    reg = MeasureRegistry.from_dict(store.load_registry() or {})
    wide = sorted(reg.entries.values(), key=lambda e: -len(e.surface_forms))[:10]
    for e in wide:
        if len(e.surface_forms) > 1:
            print(f"  {len(e.surface_forms):3d}  {e.key:34s} {sorted(e.surface_forms)[:5]}")

    rule("8. Contradictions, ranked -- the ones to check by hand")
    for r in store.relations(verdict="contradicts", limit=15):
        a, b = by_id.get(r.claim_a_id), by_id.get(r.claim_b_id)
        if not (a and b):
            continue
        tag = "CROSS-DOC" if a.doc_id != b.doc_id else "same-doc "
        print(f"  [{tag}] {a.measure_raw[:30]:32s} {a.display_value():>16s} (p{a.evidence.page_no})"
              f"  vs {b.display_value():>16s} (p{b.evidence.page_no})")
        print(f"             period={a.qualifiers.period.label if a.qualifiers.period else None}"
              f" / {b.qualifiers.period.label if b.qualifiers.period else None}"
              f"   scope={a.qualifiers.scope}/{b.qualifiers.scope}"
              f"   basis={a.qualifiers.basis}/{b.qualifiers.basis}")

    rule("9. Verdict mix")
    print(" ", store.summary()["verdicts"])
    print("  cross-document relations:", store.summary()["relations_cross_doc"])

    rule("10. Quarantined claims -- did we discard anything real?")
    for c in store.quarantined(limit=8):
        print(f"  p{c.evidence.page_no:<4d} {c.measure_raw[:28]:30s} quote={c.evidence.quote[:64]!r}")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
