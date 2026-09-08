"""Measure page-selection recall for the triage scorer.

Triage saves a large amount of time and API budget, but only if it keeps the
pages that actually carry facts. This harness pins a handful of known
fact-bearing spans in the starter corpus and reports whether the scorer's
budget retains their pages.

The fixture is an evaluation artifact, not pipeline input: nothing in
`factlayer/` reads it. Run it after changing the scorer.

    python scripts/eval_triage.py [--budget 40]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factlayer.ingest import load_pdf  # noqa: E402
from factlayer.triage import score_page, triage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "starter-datasets"
FIXTURE = ROOT / "tests" / "fixtures" / "triage_targets.json"


def run(budget: int | None, quiet: bool = False) -> tuple[int, int]:
    targets = json.loads(FIXTURE.read_text())["targets"]
    by_doc: dict[str, list[dict]] = {}
    for t in targets:
        by_doc.setdefault(t["doc"], []).append(t)

    found = missed = 0
    for rel, items in by_doc.items():
        doc = load_pdf(CORPUS / rel)
        kept = {s.page_no for s in triage(doc, budget=budget)}
        for t in items:
            pages = [p.page_no for p in doc.pages if t["needle"] in p.text]
            if not pages:
                if not quiet:
                    print(f"  ??  needle absent from corpus: {t['needle']!r}")
                missed += 1
                continue
            hit = sorted(pages, key=lambda n: score_page(doc.page(n)).score, reverse=True)[0]
            ok = any(p in kept for p in pages)
            best = score_page(doc.page(hit)).score
            if not quiet:
                print(
                    f"  {'OK ' if ok else 'MISS'}  p{hit:<3d} score={best:5.2f}  "
                    f"{t['needle'][:44]:46s} {t['why']}"
                )
            found += ok
            missed += not ok
    return found, missed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--budget", type=int, default=None,
        help="pages kept per document (default: unlimited -- drop empty pages only)",
    )
    ap.add_argument(
        "--sweep", action="store_true", help="report recall across several budgets",
    )
    args = ap.parse_args()

    if args.sweep:
        print("budget  recall   note")
        for b in (20, 40, 60, 80, None):
            found, missed = run(b, quiet=True)
            total = found + missed
            label = str(b) if b else "none"
            print(f"{label:>6}  {found}/{total} = {found / total:3.0%}")
        return 0

    label = args.budget if args.budget else "unlimited"
    print(f"Triage recall @ budget={label} pages/doc\n")
    found, missed = run(args.budget)
    total = found + missed
    print(f"\nrecall: {found}/{total} = {found / total:.0%}")
    return 0 if missed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
