"""Ingest PDFs into the knowledge layer.

    python scripts/ingest.py starter-datasets/delhivery/*.pdf
    python scripts/ingest.py --all            # both starter corpora
    python scripts/ingest.py --reset --all    # rebuild from scratch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factlayer.config import settings          # noqa: E402
from factlayer.db import Store                 # noqa: E402
from factlayer.llm import LLMClient            # noqa: E402
from factlayer.pipeline import ingest_pdf      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def exclusive_run(db_path: Path):
    """Refuse to start if another ingest is already using this database.

    Two concurrent runs against one SQLite file deadlock each other, and if
    either was started with --reset it deletes the database out from under the
    other. The symptom is a bare "disk I/O error" from a completely unrelated
    line, which is a miserable thing to debug -- so this fails fast with a
    message that says what actually happened.
    """
    # Kept out of the database's own name so that clearing the database (a
    # "rm data/factlayer.db*" during development, say) cannot silently remove
    # the guard that stops two runs colliding.
    lock = db_path.parent / ".ingest.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = lock.read_text().strip() or "unknown"
        stale = not holder.isdigit() or not _pid_alive(int(holder))
        if not stale:
            raise SystemExit(
                f"Another ingest (pid {holder}) is already running against "
                f"{db_path}.\nWait for it to finish, or stop it first."
            )
        print(f"clearing stale lock from pid {holder}")
        lock.unlink(missing_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError) as exc:
        return isinstance(exc, PermissionError)
    return True


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
    if args.reset:
        # SQLite in WAL mode keeps -wal and -shm siblings. Removing only the
        # main file leaves those behind, and the next connection fails with a
        # "disk I/O error" that looks nothing like the actual cause.
        for suffix in ("", "-wal", "-shm", "-journal"):
            victim = settings.db_path.with_name(settings.db_path.name + suffix)
            if victim.exists():
                victim.unlink()
                print(f"removed {victim.name}")

    started = time.monotonic()
    with exclusive_run(settings.db_path):
      store = Store(settings.db_path)
      llm = LLMClient()

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
          if d['pages_skipped_no_quota']:
              print(f"    !! {d['pages_skipped_no_quota']} pages NOT extracted "
                    f"(provider daily quota exhausted)")
          print(f"    measures +{d['measures_added']}   {d['seconds']}s   "
                f"llm={d['llm_usage']['calls']} calls, "
                f"{d['llm_usage']['cache_hits']} cached", flush=True)

      print(f"\n=== summary ({time.monotonic() - started:.0f}s) ===")
      print(json.dumps(store.summary(), indent=2))
      print("llm:", json.dumps(llm.usage.as_dict()))
      await llm.aclose()
      store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
