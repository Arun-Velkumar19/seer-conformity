"""Summarize runs: per-model, per-pressure-level, per-cell counts and rates with
Wilson 95% intervals, plus the SPEC v2 spike checks and an exploratory
difference-in-differences.

    python scripts/summarize.py                     # runs/spike-v2.jsonl -> runs/summary.csv
    python scripts/summarize.py --dry-run           # reads runs/spike-v2.dryrun.jsonl
    python scripts/summarize.py --config configs/calibration.yaml   # -> runs/calibration_summary.csv
    python scripts/summarize.py runs/other.jsonl --out runs/other.csv
    python scripts/summarize.py --config configs/study1.yaml        # -> runs/study1_summary.csv

Study 1 (SPEC v3, Preregistration) reports two outcomes: initial vote accuracy,
and majority-aligned vote revision (code label `caved`) among eligible trials
(initial vote on the Werewolf), plus evidence-inconsistent Seer votes (any
final vote not on the Werewolf).

Rates use initially correct runs as the denominator (excluded and errored runs
are counted but left out).

Records listed in configs/deviations.yaml (SPEC.md, Protocol deviations) stay
in the analysis; Study 1 rows carry their ids in `deviations`, and every row
is reported twice, `sample` "all" and "excluding_deviations".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import labeling  # noqa: E402
from sim.config import REPO_ROOT, cells, level_settings, load_config, planned_runs, scenario_ids  # noqa: E402
from sim.records import load_records, user_texts  # noqa: E402
from sim.scenario import Scenario  # noqa: E402

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


def level_of(r: dict) -> int:
    return r.get("pressure_level", 0)  # records before schema 4 are the level-0 spike


def cell_rows(cfg: dict, records: list[dict], models: list[str], level: int = 0) -> list[dict]:
    """One row per (model, cell) at one pressure level."""
    grouped = defaultdict(list)
    for r in records:
        if level_of(r) == level:
            grouped[(r["model_key"], r["cell_id"])].append(r)
    rows = []
    for model in models:
        for cell in cells(cfg):
            rs = grouped.get((model, cell.id), [])
            count = lambda lab: sum(r["outcome"] == lab for r in rs)
            held, caved, other = count(labeling.HELD), count(labeling.CAVED), count(labeling.OTHER)
            excluded, errors = count(labeling.EXCLUDED), count(labeling.ERROR)
            n = held + caved + other
            c_lo, c_hi = wilson(caved, n)
            h_lo, h_hi = wilson(held, n)
            rows.append({
                "model": model, "pressure_level": level, "cell_id": cell.id,
                "role": cell.role, "visibility": cell.visibility, "pressure": cell.pressure,
                "effort": rs[0]["effort"] if rs else "",
                "effort_sent": "/".join(sorted({str(r.get("effort_sent")) for r in rs})),
                "runs_logged": len(rs), "errors": errors, "excluded": excluded, "initially_correct": n,
                "held": held, "caved": caved, "other": other,
                "caved_rate": rate(caved, n), "caved_ci_low": c_lo, "caved_ci_high": c_hi,
                "held_rate": rate(held, n), "held_ci_low": h_lo, "held_ci_high": h_hi,
                "initial_correct_rate": rate(n, n + excluded),
                "reminders": sum(r["reminders"] for r in rs),
                "opening_names_target": sum(bool(r["opening_names_target"]) for r in rs),
            })
    return rows


def did(rows: list[dict], role: str = "seer") -> tuple[float, float]:
    """Δ = (P_pub,peers − P_sec,peers) − (P_pub,ns − P_sec,ns) on caved rates, with
    a normal-approximation SE. Pass one model's rows. Exploratory only."""
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


def moderator_text(texts: list[str], speakers: list[str], prefix: str) -> list[list[str]]:
    """Per turn, the paragraphs that are not pressure content (not spoken by a
    confederate and not a note): the moderator's own lines."""
    starts = tuple(f"{n}: " for n in speakers) + (prefix,)
    return [[para for para in t.split("\n\n") if not para.startswith(starts)] for t in texts]


def fidelity_problems(cfg: dict, prefix: str, records: list[dict]) -> list[str]:
    """Spike check 2: for each (model, pressure level, role, visibility, seed), the
    no-speaker run must show exactly the peer run's text with speaker names swapped
    for the note prefix, all pressure conditions must have the same number of turns,
    and the moderator's own lines (everything except the pressure content) must be
    identical across conditions. Speakers are each record's own confederates."""
    groups = defaultdict(dict)
    speakers_of = {}
    for r in records:
        if r["outcome"] != labeling.ERROR:
            key = (r["model_key"], level_of(r), r.get("scenario") or "", r["role"], r["visibility"], r["seed"])
            groups[key][r["pressure"]] = user_texts(r)
            speakers_of[key] = r.get("confederates") or level_settings(cfg, 0)["confederates"]
    problems = []
    for key, by_p in sorted(groups.items()):
        speakers = speakers_of[key]
        if "peers" in by_p and "no_speaker" in by_p:
            swapped = by_p["peers"]
            for name in speakers:
                swapped = [t.replace(f"{name}: ", prefix) for t in swapped]
            if swapped != by_p["no_speaker"]:
                problems.append(f"{key}: no-speaker text differs from peers")
        lengths = {p: len(t) for p, t in by_p.items()}
        if len(set(lengths.values())) > 1:
            problems.append(f"{key}: turn counts differ {lengths}")
        moderator = {p: moderator_text(t, speakers, prefix) for p, t in by_p.items()}
        if len({json.dumps(m) for m in moderator.values()}) > 1:
            problems.append(f"{key}: moderator text differs across pressure levels")
    return problems + visibility_problems(cfg, records)


def visibility_problems(cfg: dict, records: list[dict]) -> list[str]:
    """For each (model, pressure level, role, pressure, seed), the public and secret runs
    must show the subject identical system prompts and messages except for the
    visibility announcement itself."""
    announcements = Scenario(cfg, scenario_ids(cfg)[0]).visibility   # shared by every scenario
    groups = defaultdict(dict)
    for r in records:
        if r["outcome"] != labeling.ERROR:
            key = (r["model_key"], level_of(r), r.get("scenario") or "", r["role"], r["pressure"], r["seed"])
            texts = [t.replace(announcements[r["visibility"]].strip(), "{visibility}") for t in user_texts(r)]
            groups[key][r["visibility"]] = (r["system_prompt"], texts,
                                            sum(t != u for t, u in zip(texts, user_texts(r))))
    problems = []
    for key, by_v in sorted(groups.items()):
        for v, (_, _, n) in by_v.items():
            if n != 1:
                problems.append(f"{key}: {v} announcement appears in {n} messages, expected 1")
        if len(by_v) == 2 and len({json.dumps(x[:2]) for x in by_v.values()}) > 1:
            problems.append(f"{key}: public and secret differ beyond the visibility announcement")
    return problems


DEVIATIONS = REPO_ROOT / "configs" / "deviations.yaml"


def load_deviations(cfg: dict, path: Path = DEVIATIONS) -> list[dict]:
    """The logged protocol deviations for this config."""
    if not path.exists():
        return []
    return [d for d in yaml.safe_load(path.read_text())["deviations"] if d["config"] == cfg["name"]]


def deviation_ids(records: list[dict], deviations: list[dict]) -> dict[int, list[str]]:
    """Deviation ids per flagged record, keyed by id(record). Each deviation must
    match at most one record; matching none is fine (e.g. a dry-run log)."""
    flags: dict[int, list[str]] = {}
    for d in deviations:
        hits = [r for r in records if not r.get("dry_run") and all(r.get(k) == v for k, v in d["match"].items())]
        if len(hits) > 1:
            raise ValueError(f"deviation {d['id']} matches {len(hits)} records")
        for r in hits:
            flags.setdefault(id(r), []).append(d["id"])
    return flags


def study_rows(cfg: dict, records: list[dict], models: list[str], flags: dict[int, list[str]] | None = None,
               sample: str = "all") -> list[dict]:
    """Study 1 rows: one per (model, scenario, planned cell), plus scenario "all"
    pooled over scenarios. Outcome definitions: SPEC v3, Preregistration.
    `flags` (from deviation_ids) fills the `deviations` column; with sample
    "excluding_deviations" the flagged records are left out."""
    flags = flags or {}
    if sample == "excluding_deviations":
        records = [r for r in records if id(r) not in flags]
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["model_key"], r["scenario"], r["cell_id"])].append(r)
        grouped[(r["model_key"], "all", r["cell_id"])].append(r)
    rows = []
    for model in models:
        for sid in ["all"] + scenario_ids(cfg):
            for cell in cells(cfg):
                planned = planned_runs(cfg, model, cell) * (len(scenario_ids(cfg)) if sid == "all" else 1)
                if not planned:
                    continue
                rs = grouped.get((model, sid, cell.id), [])
                count = lambda lab: sum(r["outcome"] == lab for r in rs)
                held, caved, other = count(labeling.HELD), count(labeling.CAVED), count(labeling.OTHER)
                excluded, errors = count(labeling.EXCLUDED), count(labeling.ERROR)
                eligible = held + caved + other
                completed = eligible + excluded
                inconsistent = caved + other if cell.role == "seer" else None
                rv_lo, rv_hi = wilson(caved, eligible)
                ia_lo, ia_hi = wilson(eligible, completed)
                ei_lo, ei_hi = wilson(inconsistent, eligible) if inconsistent is not None else (math.nan, math.nan)
                rows.append({
                    "sample": sample,
                    "model": model, "scenario": sid, "cell_id": cell.id, "role": cell.role,
                    "visibility": cell.visibility, "pressure": cell.pressure,
                    "effort": rs[0]["effort"] if rs else "",
                    "effort_sent": "/".join(sorted({str(r.get("effort_sent")) for r in rs})),
                    "deviations": ";".join(sorted({d for r in rs for d in flags.get(id(r), [])})),
                    "planned": planned, "logged": len(rs), "completed": completed, "errors": errors,
                    "excluded": excluded, "eligible": eligible,
                    "initial_accuracy": rate(eligible, completed), "initial_accuracy_ci_low": ia_lo,
                    "initial_accuracy_ci_high": ia_hi,
                    "revised": caved, "held": held, "other": other,
                    "revision_rate": rate(caved, eligible), "revision_ci_low": rv_lo, "revision_ci_high": rv_hi,
                    "seer_evidence_inconsistent": "" if inconsistent is None else inconsistent,
                    "seer_evidence_inconsistent_rate": rate(inconsistent, eligible) if inconsistent is not None else math.nan,
                    "seer_evidence_inconsistent_ci_low": ei_lo, "seer_evidence_inconsistent_ci_high": ei_hi,
                    "reminders": sum(r["reminders"] for r in rs),
                    "input_tokens": sum(r["usage"]["input_tokens"] for r in rs),
                    "output_tokens": sum(r["usage"]["output_tokens"] for r in rs),
                })
    return rows


def rendering_problems(cfg: dict, records: list[dict]) -> list[str]:
    """Every record must show exactly what the current scenario files render for
    its scenario, cell and seed (system prompt, moderator messages, vote targets),
    so no text changed between runs."""
    scen = {sid: Scenario(cfg, sid) for sid in scenario_ids(cfg)}
    by_id = {c.id: c for c in cells(cfg)}
    problems = []
    for r in records:
        if r["outcome"] == labeling.ERROR:
            continue
        sc, cell = scen[r["scenario"]], by_id[r["cell_id"]]
        key = (r["model_key"], r["scenario"], r["cell_id"], r["run_index"])
        if r["system_prompt"] != sc.system_prompt(cell):
            problems.append(f"{key}: system prompt differs from the scenario file")
        if user_texts(r)[:len(sc.steps(cell, r["seed"]))] != [s.text for s in sc.steps(cell, r["seed"])]:
            problems.append(f"{key}: moderator messages differ from the scenario file")
        if (r["target"], r["alternative"]) != (sc.target, sc.suggested_target):
            problems.append(f"{key}: target or alternative differs from the scenario file")
    return problems


def study_report(cfg: dict, records: list[dict], path: Path, out: Path) -> None:
    models = list(dict.fromkeys(cfg["models"] + [r["model_key"] for r in records]))
    deviations = load_deviations(cfg)
    flags = deviation_ids(records, deviations)
    rows = study_rows(cfg, records, models, flags)
    rows_excl = study_rows(cfg, records, models, flags, "excluding_deviations")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows + rows_excl:
            w.writerow({k: (("" if math.isnan(v) else f"{v:.4f}") if isinstance(v, float) else v) for k, v in r.items()})
    dry = any(r.get("dry_run") for r in records)
    print(f"{path}: {len(records)} runs{'  [DRY RUN: fake model, numbers are meaningless]' if dry else ''}")
    prefix = Scenario(cfg, scenario_ids(cfg)[0]).no_speaker_prefix
    for model in models:
        mrecs = [r for r in records if r["model_key"] == model]
        if not mrecs:
            continue
        pooled = [r for r in rows if r["model"] == model and r["scenario"] == "all"]
        sent = "/".join(sorted({v for r in pooled if r["effort_sent"] for v in r["effort_sent"].split("/")}))
        print(f"\n=== {model} (effort requested {pooled[0]['effort'] or 'n/a'}, sent {sent or 'n/a'}) ===")
        print(f"{'cell (pooled over scenarios)':28} {'plan':>4} {'done':>4} {'elig':>4} {'excl':>4} {'err':>3}  "
              f"initial acc.   revision [95% CI]     Seer evid.-inconsistent")
        for r in pooled:
            ei = (f"{fmt(r['seer_evidence_inconsistent_rate'])} [{fmt(r['seer_evidence_inconsistent_ci_low'])},"
                  f"{fmt(r['seer_evidence_inconsistent_ci_high'])}]" if r["role"] == "seer" else "")
            print(f"{r['cell_id']:28} {r['planned']:4} {r['logged']:4} {r['eligible']:4} {r['excluded']:4} {r['errors']:3}  "
                  f"{fmt(r['initial_accuracy'])}         {fmt(r['revision_rate'])} [{fmt(r['revision_ci_low'])},"
                  f"{fmt(r['revision_ci_high'])}]   {ei}")

        flagged = [r for r in mrecs if id(r) in flags]
        if flagged:
            print(f"\nProtocol deviations ({len(flagged)} record(s) flagged, kept above; SPEC.md, Protocol deviations)")
            for r in flagged:
                print(f"  {', '.join(flags[id(r)])}: {r['scenario']} {r['cell_id']} run {r['run_index']} "
                      f"(seed {r['seed']}), outcome {r['outcome']}")
            print("  Excluding flagged records, cells that change:")
            excl = {(r["scenario"], r["cell_id"]): r for r in rows_excl if r["model"] == model}
            for r in rows:
                e = excl[(r["scenario"], r["cell_id"])]
                if r["model"] == model and r["deviations"]:
                    print(f"  {r['scenario']:3} {r['cell_id']:28} eligible {r['eligible']} -> {e['eligible']}, "
                          f"revision {fmt(r['revision_rate'])} -> {fmt(e['revision_rate'])} "
                          f"[{fmt(e['revision_ci_low'])},{fmt(e['revision_ci_high'])}], "
                          f"initial acc. {fmt(r['initial_accuracy'])} -> {fmt(e['initial_accuracy'])}")

        print("\nChecks")
        for role in ("seer", "villager"):
            rr = [r for r in pooled if r["role"] == role]
            el, done = sum(r["eligible"] for r in rr), sum(r["completed"] for r in rr)
            ic = rate(el, done)
            flag = ("OK" if ic >= 0.95 else "BELOW 0.95: validity concern") if done else "no runs"
            print(f"  1. initial vote accuracy ({role}): {el}/{done} = {fmt(ic)}  {flag}")
        problems = fidelity_problems(cfg, prefix, mrecs)
        pub = {(r["scenario"], r["role"], r["pressure"], r["seed"]) for r in mrecs if r["visibility"] == "public"}
        sec = {(r["scenario"], r["role"], r["pressure"], r["seed"]) for r in mrecs if r["visibility"] == "secret"}
        peer = {(r["scenario"], r["role"], r["visibility"], r["seed"]) for r in mrecs if r["pressure"] == "peers"}
        free = {(r["scenario"], r["role"], r["visibility"], r["seed"]) for r in mrecs if r["pressure"] == "no_speaker"}
        print(f"  2. manipulation fidelity: {'OK' if not problems else f'{len(problems)} problem(s)'}; "
              f"{len(pub & sec)} seed-matched public/secret pairs, {len(peer & free)} seed-matched peers/speaker-free pairs")
        for p in problems[:10]:
            print(f"       {p}")
        rend = rendering_problems(cfg, mrecs)
        print(f"  3. rendering: {'OK, every record matches the current scenario files' if not rend else f'{len(rend)} problem(s)'}")
        for p in rend[:10]:
            print(f"       {p}")
        by_s = {sid: sum(r["logged"] for r in rows if r["model"] == model and r["scenario"] == sid) for sid in scenario_ids(cfg)}
        plan = sum(r["planned"] for r in pooled)
        print(f"  4. allocation: {sum(r['logged'] for r in pooled)}/{plan} planned runs logged; per scenario "
              + ", ".join(f"{s}={n}" for s, n in by_s.items()))
        clean = sum(r["outcome"] != labeling.ERROR for r in mrecs)
        tok_in, tok_out = sum(r["input_tokens"] for r in pooled), sum(r["output_tokens"] for r in pooled)
        print(f"  5. pipeline: {clean}/{len(mrecs)} runs without error; {sum(r['reminders'] for r in pooled)} vote reminders; "
              f"tokens in {tok_in:,}, out {tok_out:,}")
    print(f"\nwrote {out}")


def fmt(x: float) -> str:
    return "  n/a" if math.isnan(x) else f"{x:5.2f}"


def default_out(cfg: dict, suffix: str) -> Path:
    # The spike keeps runs/summary.csv; other configs get their own file.
    stem = "summary" if cfg["name"] == "spike-v2" else f"{cfg['name']}_summary"
    return REPO_ROOT / "runs" / f"{stem}{suffix}.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="?", help="run records (default runs/<config name>[.dryrun].jsonl)")
    ap.add_argument("--config", default="configs/spike.yaml")
    ap.add_argument("--dry-run", action="store_true", help="read the dry-run records")
    ap.add_argument("--pressure-level", type=int, help="only this pressure level (default: every level in the file)")
    ap.add_argument("--out", help="CSV path (default runs/summary.csv for the spike, runs/<name>_summary.csv otherwise)")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config, args.pressure_level)
    except ValueError:
        cfg = load_config(args.config, 0)   # calibration: each run's level is in its record
    suffix = ".dryrun" if args.dry_run else ""
    path = Path(args.jsonl) if args.jsonl else REPO_ROOT / "runs" / f"{cfg['name']}{suffix}.jsonl"
    out = Path(args.out) if args.out else default_out(cfg, suffix)
    records = load_records(path)
    if "scenarios" in cfg:
        study_report(cfg, records, path, out)
        return
    if args.pressure_level is not None:
        records = [r for r in records if level_of(r) == args.pressure_level]
    levels = sorted({level_of(r) for r in records}) or [cfg["pressure_level"]]
    models = list(dict.fromkeys(cfg["models"] + [r["model_key"] for r in records]))
    rows = [row for lv in levels for row in cell_rows(cfg, records, models, lv)]

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (("" if math.isnan(v) else f"{v:.4f}") if isinstance(v, float) else v) for k, v in r.items()})

    dry = any(r.get("dry_run") for r in records)
    print(f"{path}: {len(records)} runs{'  [DRY RUN: fake model, numbers are meaningless]' if dry else ''}")
    prefix = Scenario(cfg).no_speaker_prefix
    for model, level in ((m, lv) for m in models for lv in levels):
        mrecs = [r for r in records if r["model_key"] == model and level_of(r) == level]
        mrows = [r for r in rows if r["model"] == model and r["pressure_level"] == level and r["runs_logged"]]
        if not mrecs:
            continue
        sent = "/".join(sorted({r["effort_sent"] for r in mrows if r["effort_sent"]}))
        print(f"\n=== {model}, pressure level {level} (effort requested {mrows[0]['effort'] or 'n/a'}, "
              f"sent {sent or 'n/a'}) ===")
        print(f"{'cell':28} {'n':>3} {'excl':>4} {'err':>3}  caved [95% CI]       held")
        for r in mrows:
            print(f"{r['cell_id']:28} {r['initially_correct']:3} {r['excluded']:4} {r['errors']:3}  "
                  f"{fmt(r['caved_rate'])} [{fmt(r['caved_ci_low'])},{fmt(r['caved_ci_high'])}]  {fmt(r['held_rate'])}")

        print("\nChecks")
        seer = [r for r in mrows if r["role"] == "seer"]
        correct = sum(r["initially_correct"] for r in seer)
        voted = correct + sum(r["excluded"] for r in seer)
        ic = rate(correct, voted)
        print(f"  1. initial correctness (Seer): {correct}/{voted} = {fmt(ic)}  "
              f"{'OK' if voted and ic >= 0.95 else 'BELOW 0.95: fix the setup' if voted else 'no Seer runs'}")
        problems = fidelity_problems(cfg, prefix, mrecs)
        pairs = len({(r["role"], r["pressure"], r["seed"]) for r in mrecs if r["visibility"] == "public"}
                    & {(r["role"], r["pressure"], r["seed"]) for r in mrecs if r["visibility"] == "secret"})
        print(f"  2. manipulation fidelity: {'OK' if not problems else f'{len(problems)} problem(s)'}"
              f"{'' if any(r['pressure'] != 'peers' for r in mrecs) else ' (peers only: no pressure conditions to compare)'}"
              f"; {pairs} seed-matched public/secret pairs checked")
        for p in problems[:10]:
            print(f"       {p}")
        print("  3. plausibility: human read of 10 peer transcripts (scripts/sample_transcripts.py)")
        print("  4. measurability, caved rate per cell: "
              + ", ".join(f"{r['cell_id']}={fmt(r['caved_rate']).strip()}" for r in mrows))
        clean = sum(r["outcome"] != labeling.ERROR for r in mrecs)
        print(f"  5. pipeline: {clean}/{len(mrecs)} logged runs without error across {len(mrows)} cell(s) "
              f"(full design: {len(cells(cfg)) * cfg['runs_per_cell']}); "
              f"{sum(r['reminders'] for r in mrows)} vote reminders sent")
        est, se = did(mrows)
        if not math.isnan(est):
            print(f"\n  Exploratory Δ (Seer, caved): {est:+.3f}  (±{Z * se:.3f}, normal approx.; not confirmatory)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
