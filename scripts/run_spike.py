"""Run the feasibility spike.

    python scripts/run_spike.py --dry-run            # full pipeline, fake model, no keys, no cost
    python scripts/run_spike.py                      # real runs for every model in the config
    python scripts/run_spike.py --limit 2            # first 2 pending runs only
    python scripts/run_spike.py --model gpt-6-luna   # one model from the config
    python scripts/run_spike.py --show-prompts --cell seer-public-peers

Calibration (exploratory; higher pressure levels, runs/calibration.jsonl):

    python scripts/run_spike.py --config configs/calibration.yaml --pressure-level 1 \
        --cells seer-public-peers,villager-public-peers
    python scripts/run_spike.py --config configs/calibration.yaml --pressure-level 3 \
        --show-prompts --cell seer-public-peers

Real runs append to runs/<name>.jsonl and resume if interrupted (errored runs
are retried). Dry runs write runs/<name>.dryrun.jsonl. Models, effort and
token limits come from the config and configs/models.yaml. runs/spike-v2.jsonl
holds the finished level-0 spike; only the spike config at level 0 may write it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import REPO_ROOT, cells, load_config  # noqa: E402
from sim.runner import run_all  # noqa: E402
from sim.scenario import Scenario  # noqa: E402


SPIKE_RECORDS = REPO_ROOT / "runs" / "spike-v2.jsonl"


def records_path(cfg: dict, dry_run: bool) -> Path:
    return REPO_ROOT / "runs" / f"{cfg['name']}{'.dryrun' if dry_run else ''}.jsonl"


def protected_problem(cfg: dict, out: Path) -> str | None:
    """runs/spike-v2.jsonl is the level-0 spike: refuse any other writer."""
    if out.resolve() == SPIKE_RECORDS.resolve() and (cfg["name"] != "spike-v2" or cfg["pressure_level"] != 0):
        return (f"{SPIKE_RECORDS.name} holds the level-0 spike and is never written at other levels "
                "or by other configs; use --config configs/calibration.yaml (writes runs/calibration.jsonl)")
    return None


def parse_cells(cfg: dict, values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    wanted = [c.strip() for v in values for c in v.split(",") if c.strip()]
    known = [c.id for c in cells(cfg)]
    unknown = [c for c in wanted if c not in known]
    if unknown:
        raise ValueError(f"unknown cell(s) {unknown}; known: {known}")
    return wanted


def show_prompts(cfg: dict, cell_id: str | None, seed_index: int) -> None:
    by_id = {c.id: c for c in cells(cfg)}
    cell = by_id[cell_id] if cell_id else next(iter(by_id.values()))
    seed = cfg["seeds"][seed_index]
    sc = Scenario(cfg)
    print(f"# {cell.id}, pressure level {cfg['pressure_level']}, seed {seed}\n")
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
    ap.add_argument("--pressure-level", type=int, help="override the config's pressure level (0-3)")
    ap.add_argument("--cells", action="append",
                    help="only these cell ids, comma-separated or repeated, e.g. seer-public-peers,villager-public-peers")
    ap.add_argument("--show-prompts", action="store_true", help="print one cell's rendered prompts and exit")
    ap.add_argument("--cell", help="cell id for --show-prompts, e.g. seer-public-peers")
    ap.add_argument("--seed-index", type=int, default=0, help="run index whose seed --show-prompts uses")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config, args.pressure_level)
        cell_ids = parse_cells(cfg, args.cells)
    except ValueError as e:
        ap.error(str(e))
    if args.show_prompts:
        show_prompts(cfg, args.cell, args.seed_index)
        return
    for m in args.model or []:
        if m not in cfg["models"]:
            ap.error(f"{m} is not in this config's models: {cfg['models']}")

    out = Path(args.out) if args.out else records_path(cfg, args.dry_run)
    problem = protected_problem(cfg, out)
    if problem:
        ap.error(problem)
    models = args.model or cfg["models"]
    n_cells = len(cell_ids) if cell_ids else len(cells(cfg))
    total = len(models) * n_cells * cfg["runs_per_cell"]
    print(f"{cfg['name']}{' [DRY RUN, fake model]' if args.dry_run else ''}: pressure level "
          f"{cfg['pressure_level']}, {n_cells} cells x {cfg['runs_per_cell']} runs, {total} runs planned")
    if cell_ids:
        print(f"  cells: {', '.join(cell_ids)}")
    for m in models:
        s = cfg["model_specs"][m]
        print(f"  {m}: {s['provider']} {s['model']}, effort {s['effort']}, max_output_tokens {s['max_output_tokens']}")
    print(f"  out: {out}")
    records = run_all(cfg, out, dry_run=args.dry_run, workers=args.workers, limit=args.limit,
                      models=models, cell_ids=cell_ids)
    errors = sum(r["error"] is not None for r in records)
    print(f"done: {len(records)} runs this session, {errors} errors (re-run to retry them)")


if __name__ == "__main__":
    main()
