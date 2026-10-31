"""Run the feasibility spike.

    python scripts/run_spike.py --dry-run            # full pipeline, fake model, no keys, no cost
    python scripts/run_spike.py                      # real runs for every model in the config
    python scripts/run_spike.py --limit 2            # first 2 pending runs only
    python scripts/run_spike.py --model gpt-6-luna   # one model from the config
    python scripts/run_spike.py --show-prompts --cell seer-public-peers

Calibration (exploratory; higher pressure levels, runs/calibration.jsonl):

    python scripts/run_spike.py --config configs/calibration.yaml --pressure-level 1 \
        --cells seer-public-peers,villager-public-peers
    python scripts/run_spike.py --config configs/calibration.yaml --pressure-level 4 --runs 30 \
        --cells seer-public-peers,seer-secret-peers
    python scripts/run_spike.py --config configs/calibration.yaml --pressure-level 3 \
        --show-prompts --cell seer-public-peers

Study 1 (SPEC v3; pressure level 4, five scenarios, runs/study1.jsonl only):

    python scripts/run_spike.py --config configs/study1.yaml --dry-run
    python scripts/run_spike.py --config configs/study1.yaml --model gpt-6-luna
    python scripts/run_spike.py --config configs/study1.yaml --model gpt-6-luna --scenarios s1 --limit 4
    python scripts/run_spike.py --config configs/study1.yaml --show-prompts --scenario s2 --cell villager-public-peers

Study 1 runs per cell are fixed by the config's allocation (--runs is refused).

--runs N overrides the config's runs per cell. Runs already logged (same model,
pressure level, cell and run index) are skipped, so raising N only adds the new
run indexes.

Real runs append to runs/<name>.jsonl and resume if interrupted (errored runs
are retried). Dry runs write runs/<name>.dryrun.jsonl. Models, effort and
token limits come from the config and configs/models.yaml. runs/spike-v2.jsonl
holds the finished level-0 spike; only the spike config at level 0 may write it.
runs/study1.jsonl is written only by the study1 config, and real study1 runs
write nowhere else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import REPO_ROOT, cells, load_config, planned_runs, scenario_ids  # noqa: E402
from sim.runner import run_all  # noqa: E402
from sim.scenario import Scenario  # noqa: E402


SPIKE_RECORDS = REPO_ROOT / "runs" / "spike-v2.jsonl"
STUDY1_RECORDS = REPO_ROOT / "runs" / "study1.jsonl"


def records_path(cfg: dict, dry_run: bool) -> Path:
    return REPO_ROOT / "runs" / f"{cfg['name']}{'.dryrun' if dry_run else ''}.jsonl"


def protected_problem(cfg: dict, out: Path, dry_run: bool = False) -> str | None:
    """runs/spike-v2.jsonl is the level-0 spike and runs/study1.jsonl is Study 1:
    refuse any other writer. Real study1 runs may write only runs/study1.jsonl."""
    if out.resolve() == SPIKE_RECORDS.resolve() and (cfg["name"] != "spike-v2" or cfg["pressure_level"] != 0):
        return (f"{SPIKE_RECORDS.name} holds the level-0 spike and is never written at other levels "
                "or by other configs; use --config configs/calibration.yaml (writes runs/calibration.jsonl)")
    if out.resolve() == STUDY1_RECORDS.resolve() and (cfg["name"] != "study1" or dry_run):
        return f"{STUDY1_RECORDS.name} holds Study 1 and is written only by real runs of configs/study1.yaml"
    if cfg["name"] == "study1" and not dry_run and out.resolve() != STUDY1_RECORDS.resolve():
        return f"real Study 1 runs write only {STUDY1_RECORDS.relative_to(REPO_ROOT)}"
    return None


def set_runs(cfg: dict, n: int) -> None:
    if "allocation" in cfg:
        raise ValueError(f"{cfg['name']} fixes runs per cell by its allocation (preregistered); --runs can't change it")
    if not 1 <= n <= len(cfg["seeds"]):
        raise ValueError(f"--runs must be between 1 and {len(cfg['seeds'])} (the number of seeds in the config)")
    cfg["runs_per_cell"] = n


def parse_cells(cfg: dict, values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    wanted = [c.strip() for v in values for c in v.split(",") if c.strip()]
    known = [c.id for c in cells(cfg)]
    unknown = [c for c in wanted if c not in known]
    if unknown:
        raise ValueError(f"unknown cell(s) {unknown}; known: {known}")
    return wanted


def parse_scenarios(cfg: dict, values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    wanted = [s.strip() for v in values for s in v.split(",") if s.strip()]
    known = [s for s in scenario_ids(cfg) if s]
    unknown = [s for s in wanted if s not in known]
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; known: {known or 'none (this config has one built-in scenario)'}")
    return wanted


def planned_total(cfg: dict, models: list[str], cell_ids: list[str] | None, scenario_list: list[str] | None) -> int:
    n_scen = len(scenario_list or scenario_ids(cfg))
    return n_scen * sum(planned_runs(cfg, m, c) for m in models for c in cells(cfg)
                        if cell_ids is None or c.id in cell_ids)


def show_prompts(cfg: dict, cell_id: str | None, seed_index: int, scenario_id: str | None = None) -> None:
    by_id = {c.id: c for c in cells(cfg)}
    cell = by_id[cell_id] if cell_id else next(iter(by_id.values()))
    seed = cfg["seeds"][seed_index]
    sid = scenario_id or scenario_ids(cfg)[0]
    sc = Scenario(cfg, sid)
    print(f"# {cell.id}, pressure level {cfg['pressure_level']}, seed {seed}{', scenario ' + sid if sid else ''}\n")
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
    ap.add_argument("--model", action="append",
                    help="model key(s) from the config's models or optional_models; default: its models")
    ap.add_argument("--out", help="JSONL output path (overrides the default)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent trials")
    ap.add_argument("--limit", type=int, help="run at most this many pending trials")
    ap.add_argument("--pressure-level", type=int, help="override the config's pressure level (0-4)")
    ap.add_argument("--runs", type=int, help="runs per cell, overriding the config (run i uses seeds[i])")
    ap.add_argument("--cells", action="append",
                    help="only these cell ids, comma-separated or repeated, e.g. seer-public-peers,villager-public-peers")
    ap.add_argument("--scenarios", action="append",
                    help="study configs: only these scenario ids, comma-separated or repeated, e.g. s1,s2")
    ap.add_argument("--scenario", help="scenario id for --show-prompts (default: the first)")
    ap.add_argument("--show-prompts", action="store_true", help="print one cell's rendered prompts and exit")
    ap.add_argument("--cell", help="cell id for --show-prompts, e.g. seer-public-peers")
    ap.add_argument("--seed-index", type=int, default=0, help="run index whose seed --show-prompts uses")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config, args.pressure_level)
        cell_ids = parse_cells(cfg, args.cells)
        scenario_list = parse_scenarios(cfg, args.scenarios)
        parse_scenarios(cfg, [args.scenario] if args.scenario else None)
        if args.runs is not None:
            set_runs(cfg, args.runs)
    except ValueError as e:
        ap.error(str(e))
    if args.show_prompts:
        show_prompts(cfg, args.cell, args.seed_index, args.scenario)
        return
    selectable = cfg["models"] + cfg.get("optional_models", [])
    for m in args.model or []:
        if m not in selectable:
            ap.error(f"{m} is not in this config's models: {selectable}")

    out = Path(args.out) if args.out else records_path(cfg, args.dry_run)
    problem = protected_problem(cfg, out, args.dry_run)
    if problem:
        ap.error(problem)
    models = args.model or cfg["models"]
    n_cells = len(cell_ids) if cell_ids else len(cells(cfg))
    total = planned_total(cfg, models, cell_ids, scenario_list)
    per_cell = (f"{cfg['runs_per_cell']} runs" if "allocation" not in cfg
                else "allocation " + "; ".join(f"{b['name']} {b['runs']}" for b in cfg["allocation"]))
    scen = [s for s in (scenario_list or scenario_ids(cfg)) if s]
    print(f"{cfg['name']}{' [DRY RUN, fake model]' if args.dry_run else ''}: pressure level "
          f"{cfg['pressure_level']}, {n_cells} cells x {per_cell}"
          f"{f' x {len(scen)} scenarios' if scen else ''}, {total} runs planned")
    if cell_ids:
        print(f"  cells: {', '.join(cell_ids)}")
    if scen:
        print(f"  scenarios: {', '.join(scen)}")
    for m in models:
        s = cfg["model_specs"][m]
        print(f"  {m}: {s['provider']} {s['model']}, effort {s['effort']}, max_output_tokens {s['max_output_tokens']}")
    print(f"  out: {out}")
    records = run_all(cfg, out, dry_run=args.dry_run, workers=args.workers, limit=args.limit,
                      models=models, cell_ids=cell_ids, scenario_filter=scenario_list)
    errors = sum(r["error"] is not None for r in records)
    print(f"done: {len(records)} runs this session, {errors} errors (re-run to retry them)")


if __name__ == "__main__":
    main()
