"""Render sampled transcripts as Markdown for human reading.

    python scripts/sample_transcripts.py                       # 10 random peer transcripts (spike check 3)
    python scripts/sample_transcripts.py --outcome caved --n 3 # notable transcripts for RESULTS.md
    python scripts/sample_transcripts.py --cell seer-public-peers --run 4

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


def render_block(role: str, b: dict) -> str | None:
    t = b.get("type")
    if t == "text":
        who = "Moderator" if role == "user" else "Subject"
        return f"**{who}:**\n\n{b['text'].strip()}"
    if t == "tool_use":
        return f"**Subject → `{b['name']}`:** `{json.dumps(b['input'], ensure_ascii=False)}`"
    if t == "tool_result":
        return f"*(tool result: {b.get('content')})*"
    return None  # thinking blocks carry no readable text by default


def render(r: dict, show_system: bool) -> str:
    head = (f"## {r['cell_id']} · run {r['run_index']} · seed {r['seed']} · outcome **{r['outcome']}**\n\n"
            f"Model `{r['model']}`, effort {r['effort']}. Private vote: "
            f"{(r['private_initial_vote'] or {}).get('target')}; final vote: {(r['final_vote'] or {}).get('target')}"
            f"{'; error: ' + r['error'] if r['error'] else ''}.")
    parts = [head]
    if show_system:
        parts.append("<details><summary>System prompt</summary>\n\n" + r["system_prompt"] + "\n\n</details>")
    for m in r["messages"]:
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for b in content:
            s = render_block(m["role"], b)
            if s:
                parts.append(s)
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="?", help="run records (default runs/<config name>.jsonl)")
    ap.add_argument("--config", default="configs/spike.yaml")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--pressure", default="peers", help="filter; 'any' for all")
    ap.add_argument("--role", help="filter: seer or villager")
    ap.add_argument("--outcome", help="filter, e.g. caved")
    ap.add_argument("--cell", help="filter by cell id")
    ap.add_argument("--run", type=int, help="filter by run index")
    ap.add_argument("--system", action="store_true", help="include the system prompt")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "samples.md"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = Path(args.jsonl) if args.jsonl else REPO_ROOT / "runs" / f"{cfg['name']}.jsonl"
    rs = load_records(path)
    # --cell already fixes the pressure level, so it overrides the --pressure default.
    keep = lambda r: ((args.cell is not None or args.pressure == "any" or r["pressure"] == args.pressure)
                      and (not args.role or r["role"] == args.role)
                      and (not args.outcome or r["outcome"] == args.outcome)
                      and (not args.cell or r["cell_id"] == args.cell)
                      and (args.run is None or r["run_index"] == args.run))
    pool = [r for r in rs if keep(r)]
    picked = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    picked.sort(key=lambda r: (r["cell_id"], r["run_index"]))

    filters = {k: v for k, v in vars(args).items() if k in ("pressure", "role", "outcome", "cell", "run") and v is not None}
    if args.cell:
        filters.pop("pressure")
    doc = [f"# Sampled transcripts\n\nSource `{path.name}`, {len(picked)} of {len(pool)} matching runs, "
           f"filters {filters}, sampling seed {args.seed}."]
    doc += [render(r, args.system) for r in picked]
    Path(args.out).write_text("\n\n---\n\n".join(doc) + "\n")
    print(f"wrote {len(picked)} transcripts to {args.out}")


if __name__ == "__main__":
    main()
