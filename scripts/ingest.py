"""Ingest PDFs into the knowledge layer.

    python scripts/ingest.py starter-datasets/delhivery/*.pdf
    python scripts/ingest.py --all            # both starter corpora
    python scripts/ingest.py --reset --all    # rebuild from scratch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factlayer.config import settings          # noqa: E402
from factlayer.db import Store                 # noqa: E402
from factlayer.llm import LLMClient            # noqa: E402
from factlayer.pipeline import ingest_pdf      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="ingest both starter corpora")
    ap.add_argument("--budget", type=int, default=None, help="max pages per document")
    ap.add_argument("--reset", action="store_true", help="delete the database first")
    ap.add_argument("--force", action="store_true", help="re-ingest documents already stored")
    args = ap.parse_args()

    paths = list(args.paths)
    if args.all:
        paths = sorted((ROOT / "starter-datasets").glob("*/*.pdf"))
    if not paths:
        ap.error("give some PDF paths, or --all")

    settings.ensure_dirs()
    if args.reset and settings.db_path.exists():
        settings.db_path.unlink()
        print(f"removed {settings.db_path}")

    store = Store(settings.db_path)
    llm = LLMClient()
    started = time.monotonic()

    for path in paths:
        print(f"\n=== {path.name} ===", flush=True)
        result = await ingest_pdf(
            path, store, llm, budget=args.budget, force=args.force,
            progress=lambda m: print(f"    {m}", flush=True),
        )
        if result.skipped:
            print("    already ingested (content hash unchanged); skipping")
            continue
        d = result.as_dict()
        print(f"    claims      : {d['claims_extracted']} "
              f"({d['claims_active']} active, {d['claims_quarantined']} quarantined)")
        print(f"    grounding   : {d['grounding']['grounding_rate']:.1%} "
              f"({d['grounding']['qualifiers_dropped']} qualifiers dropped)")
        print(f"    relations   : {d['relations_new']}  {d['verdicts']}")
        print(f"    measures +{d['measures_added']}   {d['seconds']}s")

    print(f"\n=== summary ({time.monotonic() - started:.0f}s) ===")
    print(json.dumps(store.summary(), indent=2))
    print("llm:", json.dumps(llm.usage.as_dict()))
    await llm.aclose()
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
