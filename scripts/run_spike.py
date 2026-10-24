"""Run the feasibility spike.

    python scripts/run_spike.py                      # all 180 runs (resumes if interrupted)
    python scripts/run_spike.py --limit 2            # first 2 pending runs only
    python scripts/run_spike.py --dry-run            # print one cell's prompts, no API calls

Model: $SEER_MODEL (default in the config). Records append to runs/<name>.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import REPO_ROOT, cells, load_config, model_id  # noqa: E402
from sim.runner import run_all  # noqa: E402
from sim.scenario import Scenario  # noqa: E402


def dry_run(cfg: dict, cell_id: str | None, seed_index: int) -> None:
    by_id = {c.id: c for c in cells(cfg)}
    cell = by_id[cell_id] if cell_id else next(iter(by_id.values()))
    seed = cfg["seeds"][seed_index]
    sc = Scenario(cfg)
    print(f"# {cell.id}, seed {seed}, model {model_id(cfg)}\n")
    print("## system\n")
    print(sc.system_prompt(cell))
    for step in sc.steps(cell, seed):
        tool = f" (expects {step.tool})" if step.tool else ""
        print(f"\n## {step.name}{tool}\n")
        print(step.text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/spike.yaml")
    ap.add_argument("--out", help="JSONL output (default runs/<config name>.jsonl)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent trials")
    ap.add_argument("--limit", type=int, help="run at most this many pending trials")
    ap.add_argument("--dry-run", action="store_true", help="print rendered prompts and exit")
    ap.add_argument("--cell", help="cell id for --dry-run, e.g. seer-public-peers")
    ap.add_argument("--seed-index", type=int, default=0, help="run index whose seed --dry-run uses")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        dry_run(cfg, args.cell, args.seed_index)
        return

    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"{cfg['name']}.jsonl"
    total = len(cells(cfg)) * cfg["runs_per_cell"]
    print(f"{cfg['name']}: {total} runs planned, model {model_id(cfg)}, effort {cfg['model']['effort']}, out {out}")
    records = run_all(cfg, out, workers=args.workers, limit=args.limit)
    errors = sum(r["error"] is not None for r in records)
    print(f"done: {len(records)} runs this session, {errors} errors (re-run to retry them)")


if __name__ == "__main__":
    main()
