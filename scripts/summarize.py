"""Summarize spike runs: per-cell counts and rates with Wilson 95% intervals,
plus the SPEC v2 spike checks and an exploratory difference-in-differences.

    python scripts/summarize.py                     # reads runs/spike-v2.jsonl
    python scripts/summarize.py runs/other.jsonl --out runs/summary.csv

Rates use initially correct runs as the denominator (excluded and errored runs
are counted but left out).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import labeling  # noqa: E402
from sim.config import REPO_ROOT, cells, load_config  # noqa: E402
from sim.records import load_records, user_texts  # noqa: E402

Z = 1.959963984540054  # two-sided 95%


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> float:
    return k / n if n else math.nan


def cell_rows(cfg: dict, records: list[dict]) -> list[dict]:
    by_cell = defaultdict(list)
    for r in records:
        by_cell[r["cell_id"]].append(r)
    rows = []
    for cell in cells(cfg):
        rs = by_cell.get(cell.id, [])
        count = lambda lab: sum(r["outcome"] == lab for r in rs)
        held, caved, other = count(labeling.HELD), count(labeling.CAVED), count(labeling.OTHER)
        excluded, errors = count(labeling.EXCLUDED), count(labeling.ERROR)
        n = held + caved + other
        voted = n + excluded  # runs with a usable private vote
        c_lo, c_hi = wilson(caved, n)
        h_lo, h_hi = wilson(held, n)
        rows.append({
            "cell_id": cell.id, "role": cell.role, "visibility": cell.visibility, "pressure": cell.pressure,
            "runs_logged": len(rs), "errors": errors, "excluded": excluded, "initially_correct": n,
            "held": held, "caved": caved, "other": other,
            "caved_rate": rate(caved, n), "caved_ci_low": c_lo, "caved_ci_high": c_hi,
            "held_rate": rate(held, n), "held_ci_low": h_lo, "held_ci_high": h_hi,
            "initial_correct_rate": rate(n, voted),
            "reminders": sum(r["reminders"] for r in rs),
            "opening_names_target": sum(bool(r["opening_names_target"]) for r in rs),
        })
    return rows


def did(rows: list[dict], role: str = "seer") -> tuple[float, float]:
    """Δ = (P_pub,peers − P_sec,peers) − (P_pub,ns − P_sec,ns), caved rates, with a
    normal-approximation SE. Exploratory only."""
    get = {(r["visibility"], r["pressure"]): r for r in rows if r["role"] == role}
    parts = [("public", "peers", 1), ("secret", "peers", -1), ("public", "no_speaker", -1), ("secret", "no_speaker", 1)]
    est, var = 0.0, 0.0
    for vis, pres, sign in parts:
        r = get.get((vis, pres))
        if not r or r["initially_correct"] == 0:
            return (math.nan, math.nan)
        p, n = r["caved_rate"], r["initially_correct"]
        est += sign * p
        var += p * (1 - p) / n
    return (est, math.sqrt(var))


def fidelity_problems(cfg: dict, records: list[dict]) -> list[str]:
    """Spike check 2: for each (role, visibility, seed), the no-speaker run must show
    exactly the peer run's text with speaker names swapped for the note prefix, and
    the none run must have the same number of turns."""
    sc = cfg["scenario"]
    prefix = sc["attribution"]["no_speaker"]
    groups = defaultdict(dict)
    for r in records:
        if r["outcome"] != labeling.ERROR:
            groups[(r["role"], r["visibility"], r["seed"])][r["pressure"]] = user_texts(r)
    problems = []
    for key, by_p in sorted(groups.items()):
        if "peers" in by_p and "no_speaker" in by_p:
            swapped = by_p["peers"]
            for name in sc["confederates"]:
                swapped = [t.replace(f"{name}: ", prefix) for t in swapped]
            if swapped != by_p["no_speaker"]:
                problems.append(f"{key}: no-speaker text differs from peers")
        lengths = {p: len(t) for p, t in by_p.items()}
        if len(set(lengths.values())) > 1:
            problems.append(f"{key}: turn counts differ {lengths}")
    return problems


def fmt(x: float) -> str:
    return "  n/a" if math.isnan(x) else f"{x:5.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="?", help="run records (default runs/<config name>.jsonl)")
    ap.add_argument("--config", default="configs/spike.yaml")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "summary.csv"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = Path(args.jsonl) if args.jsonl else REPO_ROOT / "runs" / f"{cfg['name']}.jsonl"
    records = load_records(path)
    rows = cell_rows(cfg, records)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else ("" if isinstance(v, float) else v))
                        for k, v in r.items()})

    print(f"{path}: {len(records)} runs\n")
    print(f"{'cell':28} {'n':>3} {'excl':>4} {'err':>3}  caved [95% CI]       held")
    for r in rows:
        print(f"{r['cell_id']:28} {r['initially_correct']:3} {r['excluded']:4} {r['errors']:3}  "
              f"{fmt(r['caved_rate'])} [{fmt(r['caved_ci_low'])},{fmt(r['caved_ci_high'])}]  {fmt(r['held_rate'])}")

    print("\nSpike checks")
    seer = [r for r in rows if r["role"] == "seer"]
    correct = sum(r["initially_correct"] for r in seer)
    voted = correct + sum(r["excluded"] for r in seer)
    ic = rate(correct, voted)
    print(f"  1. initial correctness (Seer): {correct}/{voted} = {fmt(ic)}  "
          f"{'OK' if voted and ic >= 0.95 else 'BELOW 0.95: fix the setup'}")
    problems = fidelity_problems(cfg, records)
    print(f"  2. manipulation fidelity: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    for p in problems[:10]:
        print(f"       {p}")
    print("  3. plausibility: human read of 10 peer transcripts (scripts/sample_transcripts.py)")
    print("  4. measurability, Seer caved rate per cell: "
          + ", ".join(f"{r['visibility']}/{r['pressure']}={fmt(r['caved_rate']).strip()}" for r in seer))
    expected = len(cells(cfg)) * cfg["runs_per_cell"]
    clean = sum(r["outcome"] != labeling.ERROR for r in records)
    print(f"  5. pipeline: {clean}/{expected} runs logged without error"
          f"; {sum(r['reminders'] for r in rows)} vote reminders sent")

    est, se = did(rows)
    if not math.isnan(est):
        print(f"\nExploratory Δ (Seer, caved): {est:+.3f}  (±{Z * se:.3f}, normal approx.; not confirmatory)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
