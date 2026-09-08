"""Re-apply deterministic normalisation to stored claims, then rebuild relations.

Extraction is the only expensive stage. Everything after it -- unit parsing,
period resolution, scope comparison, the reconciler itself -- is deterministic
code, and improving any of it should not mean paying to read the PDFs again.

This script re-parses every stored claim's period and unit *from the labels the
documents actually used*, which are kept verbatim on each claim for exactly this
reason, then recomputes every relation. It makes no model calls at all.

    python scripts/rebuild.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factlayer.config import settings                    # noqa: E402
from factlayer.db import Store                           # noqa: E402
from factlayer.normalize.periods import parse_period     # noqa: E402
from factlayer.normalize.units import canonicalise, parse_unit  # noqa: E402
from factlayer.reconcile import reconcile_all            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    store = Store(settings.db_path)
    claims = store.all_claims(active_only=False)
    if not claims:
        print("nothing stored")
        return 1

    periods_changed = units_changed = 0
    for c in claims:
        if (p := c.qualifiers.period) is not None:
            reparsed = parse_period(p.label)
            if reparsed and (reparsed.start, reparsed.end) != (p.start, p.end):
                periods_changed += 1
                if not args.dry_run:
                    c.qualifiers.period = reparsed
        if c.quantity is not None and c.quantity.unit_raw:
            unit = parse_unit(c.quantity.unit_raw)
            low = canonicalise(c.quantity.low, unit)
            high = canonicalise(c.quantity.high, unit)
            if (low, high) != (c.quantity.canonical_low, c.quantity.canonical_high):
                units_changed += 1
                if not args.dry_run:
                    c.quantity.canonical_low, c.quantity.canonical_high = low, high
                    c.quantity.dimension = unit.dimension if unit else None
                    c.quantity.currency = unit.currency if unit else None

    print(f"claims re-normalised: {periods_changed} periods, {units_changed} units "
          f"(of {len(claims)})")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    store.save_claims(claims, {c.id: c.subject_key or "" for c in claims})

    active = [c for c in claims if c.status.value == "active"]
    relations = reconcile_all(active)
    by_id = {c.id: c for c in active}
    cross = {
        r.id: by_id[r.claim_a_id].doc_id != by_id[r.claim_b_id].doc_id
        for r in relations if r.claim_a_id in by_id and r.claim_b_id in by_id
    }
    with store.tx() as conn:
        conn.execute("DELETE FROM relations")
    store.save_relations(relations, cross)

    s = store.summary()
    print(f"relations: {s['relations']} ({s['relations_cross_doc']} cross-document)")
    print(f"verdicts : {s['verdicts']}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
