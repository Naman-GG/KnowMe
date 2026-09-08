"""Re-profile stored documents and propagate corrected subjects to their claims.

Claims inherit their subject from the document profile -- the extraction prompt
is given the profile's subject as context -- so a profile that named the
publisher instead of the subject poisoned every claim in that document. Since
reconciliation blocks on (subject, measure), the effect was silent: the RBI and
IMF reports produced no relations between them at all.

Re-extracting the corpus to fix this would cost far more of the provider's
daily allowance than re-profiling does, and it is unnecessary: the claims
themselves are sound, only the inherited subject is wrong. So this re-profiles
each document (one call each), updates the claims that carried the stale
subject, and rebuilds the relations from the stored claims with no further
model calls.

    python scripts/refresh_subjects.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factlayer.config import settings                        # noqa: E402
from factlayer.db import Store                               # noqa: E402
from factlayer.extract import profile_document               # noqa: E402
from factlayer.ingest import load_pdf                        # noqa: E402
from factlayer.llm import LLMClient, LLMError, QuotaExhausted  # noqa: E402
from factlayer.reconcile import reconcile_all                # noqa: E402
from factlayer.registry import normalise_subject             # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = ap.parse_args()

    store = Store(settings.db_path)
    llm = LLMClient()
    changed_docs = 0
    failed = 0

    for doc_row in store.documents():
        path = Path(doc_row["path"])
        if not path.exists():
            print(f"  {doc_row['filename']}: source PDF missing; skipped")
            continue

        old_subject = (doc_row["profile"] or {}).get("subject")
        try:
            profile = await profile_document(load_pdf(path), llm, raise_on_error=True)
        except QuotaExhausted as exc:
            print(f"  {doc_row['filename'][:40]:42s} NOT RE-PROFILED ({exc})"[:160])
            failed += 1
            continue
        except LLMError as exc:
            print(f"  {doc_row['filename'][:40]:42s} NOT RE-PROFILED ({exc})"[:160])
            failed += 1
            continue

        new_subject = profile.subject
        if not new_subject:
            # profile_document returns an empty profile when a call fails, which
            # would otherwise be reported as "unchanged" -- a failure dressed up
            # as a no-op.
            print(f"  {doc_row['filename'][:40]:42s} NOT RE-PROFILED (no subject returned)")
            failed += 1
            continue
        if new_subject == old_subject:
            print(f"  {doc_row['filename'][:40]:42s} subject unchanged ({old_subject!r})")
            continue

        claims = store.claims_for_document(doc_row["doc_id"])
        stale_key = normalise_subject(old_subject or "")
        # Only claims that carried the document's own (wrong) subject are
        # touched. A claim the model gave a subject of its own is left alone.
        affected = [c for c in claims if normalise_subject(c.subject_raw) == stale_key]

        print(f"  {doc_row['filename'][:40]:42s} {old_subject!r} -> {new_subject!r}  "
              f"({len(affected)}/{len(claims)} claims)")
        changed_docs += 1
        if args.dry_run:
            continue

        for c in affected:
            c.subject_raw = new_subject
            c.subject_key = normalise_subject(new_subject)
        store.save_claims(claims, {c.id: c.subject_key or "" for c in claims})
        store.upsert_document(
            doc_id=doc_row["doc_id"], filename=doc_row["filename"], path=str(path),
            sha256=doc_row["sha256"], title=doc_row["title"],
            page_count=doc_row["page_count"], profile=profile.as_dict(),
            stats=doc_row["stats"],
        )

    if failed:
        print(f"\n{failed} document(s) could not be re-profiled; their subjects are unchanged.")
    if args.dry_run:
        print("\ndry run: nothing written")
    elif changed_docs:
        print("\nrebuilding relations from stored claims (no model calls)...")
        claims = store.all_claims()
        relations = reconcile_all(claims)
        by_id = {c.id: c for c in claims}
        cross = {
            r.id: by_id[r.claim_a_id].doc_id != by_id[r.claim_b_id].doc_id
            for r in relations if r.claim_a_id in by_id and r.claim_b_id in by_id
        }
        with store.tx() as conn:
            conn.execute("DELETE FROM relations")
        store.save_relations(relations, cross)
        s = store.summary()
        print(f"  relations: {s['relations']} ({s['relations_cross_doc']} cross-document)")
        print(f"  verdicts : {s['verdicts']}")

    print("llm:", llm.usage.as_dict())
    await llm.aclose()
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
