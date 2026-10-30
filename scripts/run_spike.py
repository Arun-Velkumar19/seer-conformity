"""Run the feasibility spike.

    python scripts/run_spike.py --dry-run            # full pipeline, fake model, no keys, no cost
    python scripts/run_spike.py                      # real runs for every model in the config
    python scripts/run_spike.py --limit 2            # first 2 pending runs only
    python scripts/run_spike.py --model gpt-6-luna   # one model from the config
    python scripts/run_spike.py --show-prompts --cell seer-public-peers

Real runs append to runs/<name>.jsonl and resume if interrupted (errored runs
are retried). Dry runs write runs/<name>.dryrun.jsonl. Models, effort and
token limits come from the config and configs/models.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import REPO_ROOT, cells, load_config  # noqa: E402
from sim.runner import run_all  # noqa: E402
from sim.scenario import Scenario  # noqa: E402


def records_path(cfg: dict, dry_run: bool) -> Path:
    return REPO_ROOT / "runs" / f"{cfg['name']}{'.dryrun' if dry_run else ''}.jsonl"


def show_prompts(cfg: dict, cell_id: str | None, seed_index: int) -> None:
    by_id = {c.id: c for c in cells(cfg)}
    cell = by_id[cell_id] if cell_id else next(iter(by_id.values()))
    seed = cfg["seeds"][seed_index]
    sc = Scenario(cfg)
    print(f"# {cell.id}, seed {seed}\n")
    print("## system\n")
    print(sc.system_prompt(cell))
    for step in sc.steps(cell, seed):
        tool = f" (expects {step.tool})" if step.tool else ""
        print(f"\n## {step.name}{tool}\n")
        print(step.text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/spike.yaml")
    ap.add_argument("--dry-run", action="store_true", help="use the deterministic fake model (no API calls)")
    ap.add_argument("--model", action="append", help="model key(s) from the config; default all")
    ap.add_argument("--out", help="JSONL output path (overrides the default)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent trials")
    ap.add_argument("--limit", type=int, help="run at most this many pending trials")
    ap.add_argument("--show-prompts", action="store_true", help="print one cell's rendered prompts and exit")
    ap.add_argument("--cell", help="cell id for --show-prompts, e.g. seer-public-peers")
    ap.add_argument("--seed-index", type=int, default=0, help="run index whose seed --show-prompts uses")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.show_prompts:
        show_prompts(cfg, args.cell, args.seed_index)
        return
    for m in args.model or []:
        if m not in cfg["models"]:
            ap.error(f"{m} is not in this config's models: {cfg['models']}")

    out = Path(args.out) if args.out else records_path(cfg, args.dry_run)
    models = args.model or cfg["models"]
    total = len(models) * len(cells(cfg)) * cfg["runs_per_cell"]
    print(f"{cfg['name']}{' [DRY RUN, fake model]' if args.dry_run else ''}: {total} runs planned")
    for m in models:
        s = cfg["model_specs"][m]
        print(f"  {m}: {s['provider']} {s['model']}, effort {s['effort']}, max_output_tokens {s['max_output_tokens']}")
    print(f"  out: {out}")
    records = run_all(cfg, out, dry_run=args.dry_run, workers=args.workers, limit=args.limit, models=models)
    errors = sum(r["error"] is not None for r in records)
    print(f"done: {len(records)} runs this session, {errors} errors (re-run to retry them)")


if __name__ == "__main__":
    main()
