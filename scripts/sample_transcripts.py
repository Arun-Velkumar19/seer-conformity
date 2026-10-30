"""Render sampled transcripts as Markdown for human reading.

    python scripts/sample_transcripts.py                       # 10 random peer transcripts (spike check 3)
    python scripts/sample_transcripts.py --outcome caved --n 3 # notable transcripts for RESULTS.md
    python scripts/sample_transcripts.py --cell seer-public-peers --run 4
    python scripts/sample_transcripts.py --dry-run             # from the dry-run records

Sampling is seeded (--seed) so a reviewer's sample can be reproduced. Output goes
to runs/samples.md by default (gitignored with the rest of runs/).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import REPO_ROOT, load_config  # noqa: E402
from sim.records import load_records  # noqa: E402


def render_turn(m: dict) -> list[str]:
    out = []
    if m["role"] == "user":
        for i in m["items"]:
            if i["type"] == "text":
                out.append(f"**Moderator:**\n\n{i['text'].strip()}")
            else:
                out.append(f"*(tool result: {i['content']})*")
    else:
        if m["text"]:
            out.append(f"**Subject:**\n\n{m['text'].strip()}")
        for c in m["tool_calls"]:
            out.append(f"**Subject → `{c['name']}`:** `{json.dumps(c['input'], ensure_ascii=False)}`")
    return out


def render(r: dict, show_system: bool) -> str:
    head = (f"## {r['model_key']} · {r['cell_id']} · run {r['run_index']} · seed {r['seed']} · "
            f"outcome **{r['outcome']}**\n\n"
            f"{r['provider']} `{r['model']}`, effort {r['effort']} (sent: {r.get('effort_sent')})"
            f"{' (DRY RUN, fake model)' if r.get('dry_run') else ''}. Private vote: "
            f"{(r['private_initial_vote'] or {}).get('target')}; final vote: {(r['final_vote'] or {}).get('target')}"
            f"{'; error: ' + r['error'] if r['error'] else ''}.")
    parts = [head]
    if show_system:
        parts.append("<details><summary>System prompt</summary>\n\n" + r["system_prompt"] + "\n\n</details>")
    for m in r["transcript"]:
        parts += render_turn(m)
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="?", help="run records (default runs/<config name>[.dryrun].jsonl)")
    ap.add_argument("--config", default="configs/spike.yaml")
    ap.add_argument("--dry-run", action="store_true", help="read the dry-run records")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--pressure", default="peers", help="filter; 'any' for all")
    ap.add_argument("--model", help="filter by model key")
    ap.add_argument("--role", help="filter: seer or villager")
    ap.add_argument("--outcome", help="filter, e.g. caved")
    ap.add_argument("--cell", help="filter by cell id (overrides --pressure)")
    ap.add_argument("--run", type=int, help="filter by run index")
    ap.add_argument("--system", action="store_true", help="include the system prompt")
    ap.add_argument("--out", help="Markdown path (default runs/samples.md, or runs/samples.dryrun.md)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    suffix = ".dryrun" if args.dry_run else ""
    path = Path(args.jsonl) if args.jsonl else REPO_ROOT / "runs" / f"{cfg['name']}{suffix}.jsonl"
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"samples{suffix}.md"
    rs = load_records(path)
    keep = lambda r: ((args.cell is not None or args.pressure == "any" or r["pressure"] == args.pressure)
                      and (not args.model or r["model_key"] == args.model)
                      and (not args.role or r["role"] == args.role)
                      and (not args.outcome or r["outcome"] == args.outcome)
                      and (not args.cell or r["cell_id"] == args.cell)
                      and (args.run is None or r["run_index"] == args.run))
    pool = [r for r in rs if keep(r)]
    picked = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    picked.sort(key=lambda r: (r["model_key"], r["cell_id"], r["run_index"]))

    names = ("pressure", "model", "role", "outcome", "cell", "run")
    filters = {k: v for k, v in vars(args).items() if k in names and v is not None}
    if args.cell:
        filters.pop("pressure")
    doc = [f"# Sampled transcripts\n\nSource `{path.name}`, {len(picked)} of {len(pool)} matching runs, "
           f"filters {filters}, sampling seed {args.seed}."]
    doc += [render(r, args.system) for r in picked]
    out.write_text("\n\n---\n\n".join(doc) + "\n")
    print(f"wrote {len(picked)} transcripts to {out}")


if __name__ == "__main__":
    main()
