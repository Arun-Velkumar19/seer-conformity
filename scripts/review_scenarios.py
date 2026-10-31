"""Check and render the Study 1 scenario drafts for owner review.

Reads scenarios/study1/*.yaml, checks the writing rules (below), and writes a
review page to runs/study1_scenarios_review.html: per scenario, the cast, the
full system prompt for both roles, every level-4 text with its filler and word
counts, and a full peers transcript for an example slot order, plus the
speaker-free wording options. Makes no model calls.

Rendering here is a preview built from the calibration templates
(prompts/rules.md, role blocks, turns.yaml) with the scenario's names swapped
in; the Study 1 machinery that the runs will use is built after approval.

Usage: python scripts/review_scenarios.py [--out PATH]
Exits non-zero if any rule check fails.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sim.scenario import and_list  # noqa: E402

SCENARIO_DIR = REPO / "scenarios" / "study1"
CALIBRATION_NAMES = {"Quinn", "Avery", "Morgan", "Drew", "Riley", "Casey", "Jordan", "Taylor", "Jamie"}

ROUND_IDS = {
    1: ["r1_timing", "cc1_doubt", "r1_helpful", "r1_first_move", "r1_kill_choice", "cc_claim"],
    2: ["cc2_followup", "cc2_fits", "cc2_credible", "cc2_landed", "cc2_minion"],
    3: ["cc3_followup", "cc3_alone", "cc3_weighed", "cc3_together", "cc3_reconsider"],
}
# Example slot order for the rendered transcript (the seed varies it in runs):
# the counterclaimer is slot 3; slot 3's round-1 argument (r1_first_move) is dropped.
EXAMPLE_ORDER = {
    1: ["r1_timing", "cc1_doubt", "cc_claim", "r1_helpful", "r1_kill_choice"],
    2: ["cc2_fits", "cc2_credible", "cc2_followup", "cc2_landed", "cc2_minion"],
    3: ["cc3_alone", "cc3_weighed", "cc3_followup", "cc3_together", "cc3_reconsider"],
}
DROPPED_IN_EXAMPLE = "r1_first_move"

FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'd|I'll)\b|\b(me|my|mine|we|us|our|ours)\b", re.I)
FIRST_PERSON_I = re.compile(r"\b(I|I'm|I've|I'd|I'll)\b")
FIRST_PERSON_OTHER = re.compile(r"\b(me|my|mine|myself|we|us|our|ours)\b", re.I)
ROLE_WORDS = re.compile(r"\b(seer|werewolf|villager|minion|claim\w*|check\w*|accus\w*)\b", re.I)
VISIBILITY_WORDS = re.compile(r"\b(public\w*|secret\w*|ballot\w*|announc\w*|reveal\w*|anonymous\w*)\b", re.I)

SPEAKER_FREE_OPTIONS = {
    "A": "A note appears: ",
    "B": "Unsigned message: ",
}


def words(text: str) -> int:
    return len(text.split())


def load() -> list[dict]:
    out = []
    for path in sorted(SCENARIO_DIR.glob("s*.yaml")):
        s = yaml.safe_load(path.read_text())
        for a in s["arguments"]:
            if "both" in a:
                a["seer"] = a["villager"] = a.pop("both")
            for k in ("seer", "villager", "filler"):
                a[k] = " ".join(a[k].split())
        s["evening"] = " ".join(s["evening"].split())
        s["by_id"] = {a["id"]: a for a in s["arguments"]}
        s["names"] = [s["cast"][k] for k in ("subject", "claimant", "werewolf", "alternative")] + s["cast"]["confederates"]
        out.append(s)
    return out


def check(scenarios: list[dict]) -> list[str]:
    problems = []
    all_names = [n for s in scenarios for n in s["names"]]
    dupes = {n for n in all_names if all_names.count(n) > 1}
    if dupes:
        problems.append(f"names used in more than one scenario: {sorted(dupes)}")
    clash = set(all_names) & CALIBRATION_NAMES
    if clash:
        problems.append(f"names reused from calibration: {sorted(clash)}")
    required = {i for ids in ROUND_IDS.values() for i in ids}
    for s in scenarios:
        sid, cast = s["id"], s["cast"]
        p = lambda msg: problems.append(f"{sid}: {msg}")  # noqa: E731
        if len(s["names"]) != len(set(s["names"])):
            p("cast has a repeated name")
        if len({n[0] for n in s["names"]}) != len(s["names"]):
            p("two cast members share an initial")
        if set(s["by_id"]) != required:
            p(f"argument ids differ from the level-4 set: missing {sorted(required - set(s['by_id']))}, "
              f"extra {sorted(set(s['by_id']) - required)}")
        others = (set(all_names) | CALIBRATION_NAMES) - set(s["names"])
        for a in s["arguments"]:
            r = a["round"]
            if a["id"] not in ROUND_IDS.get(r, []):
                p(f"{a['id']} has round {r}")
            for role in ("seer", "villager"):
                t = a[role]
                if FIRST_PERSON.search(t):
                    p(f"{a['id']} ({role}) uses first person: {FIRST_PERSON.search(t).group(0)!r}")
                if re.search(rf"\b{cast['subject']}\b", t):
                    p(f"{a['id']} ({role}) names the subject; address the subject as 'you'")
                if VISIBILITY_WORDS.search(t):
                    p(f"{a['id']} ({role}) mentions vote visibility: {VISIBILITY_WORDS.search(t).group(0)!r}")
                if re.search(r"\byou(?:'re| are)\b[^.]*\bthe Werewolf\b", t, re.I):
                    p(f"{a['id']} ({role}) calls the subject the Werewolf")
                for n in others:
                    if re.search(rf"\b{n}\b", t):
                        p(f"{a['id']} ({role}) names {n}, who is not in this scenario")
                if abs(words(a["filler"]) - words(t)) > 3:
                    p(f"{a['id']} filler is {words(a['filler'])} words, {role} is {words(t)} (limit ±3)")
            if abs(words(a["seer"]) - words(a["villager"])) > 6:
                p(f"{a['id']} seer and villager differ by more than 6 words")
            f = a["filler"]
            for n in set(all_names) | CALIBRATION_NAMES:
                if re.search(rf"\b{n}\b", f):
                    p(f"{a['id']} filler names {n}")
            if ROLE_WORDS.search(f):
                p(f"{a['id']} filler mentions {ROLE_WORDS.search(f).group(0)!r}")
            if FIRST_PERSON.search(f):
                p(f"{a['id']} filler uses first person: {FIRST_PERSON.search(f).group(0)!r}")
        decls = s["vote_declarations"]
        if len(decls) != len(cast["confederates"]):
            p("need one vote declaration per confederate")
        for d in decls:
            if cast["alternative"] not in d["text"]:
                p(f"{d['id']} does not name {cast['alternative']}")
            if FIRST_PERSON.search(d["text"]) or FIRST_PERSON.search(d["filler"]):
                p(f"{d['id']} uses first person")
            if abs(words(d["text"]) - words(d["filler"])) > 1:
                p(f"{d['id']} filler is {words(d['filler'])} words, text is {words(d['text'])} (limit ±1)")
            if ROLE_WORDS.search(d["filler"]) or any(re.search(rf"\b{n}\b", d["filler"]) for n in all_names):
                p(f"{d['id']} filler names a player or role")
        for n in others:
            if re.search(rf"\b{n}\b", s["evening"]):
                p(f"evening chat names {n}, who is not in this scenario")
        for n in s["names"]:
            if not re.search(rf"\b{n}\b", s["evening"]):
                p(f"evening chat never mentions {n}")
        if cast["werewolf"] not in s["final_message"]["villager"]:
            p("the claimant's final message must name the werewolf")
    return problems


# ---------- Rendering (preview from the calibration templates) ----------

def _strip_comment(text: str) -> str:
    return re.sub(r"<!--.*?-->\s*", "", text, flags=re.S)


def _swap(text: str, s: dict) -> str:
    c = s["cast"]
    for old, new in (("Quinn", c["subject"]), ("Avery", c["claimant"]), ("Morgan", c["werewolf"]), ("Drew", c["alternative"])):
        text = re.sub(rf"\b{old}\b", new, text)
    return text


def system_prompt(s: dict, role: str) -> str:
    c = s["cast"]
    rules = _strip_comment((REPO / "prompts/rules.md").read_text())
    rules = re.sub(r"\*\*Evening chat \(before the night\)\.\*\*.*", "**Evening chat (before the night).** {evening}", rules)
    block = _strip_comment((REPO / f"prompts/role_{role}.md").read_text())
    block = re.sub(r'final message this morning read: \*".*"\*', 'final message this morning read: *"{final}"*', block)
    rules = _swap(rules, s)
    block = _swap(block, s).replace("{final}", s["final_message"][role])
    roster = c["confederates"] + [c["werewolf"], c["alternative"]]
    values = {
        "player_count": str(len(roster) + 2),
        "players": ", ".join(roster),
        "living": and_list(roster),
        "greeters": and_list(c["confederates"]),
        "minion_rule": (REPO / "prompts/rule_minion.md").read_text().strip(),
        "role_block": block.strip(),
        "evening": s["evening"],
    }
    for k, v in values.items():
        rules = rules.replace("{" + k + "}", v)
    return rules.strip()


def round_lines(s: dict, r: int, role: str, mode: str, prefix: str = "") -> list[str]:
    conf = s["cast"]["confederates"]
    lines = []
    for name, aid in zip(conf, EXAMPLE_ORDER[r]):
        a = s["by_id"][aid]
        text = a["filler"] if mode == "none" else a[role]
        lines.append(f"{prefix}{text}" if mode == "speaker-free" else f"{name}: {text}")
    if r == 3:
        for name, d in zip(conf, s["vote_declarations"]):
            text = d["filler"] if mode == "none" else d["text"]
            lines.append(f"{prefix}{text}" if mode == "speaker-free" else f"{name}: {text}")
    return lines


def steps(s: dict, role: str, visibility: str = "public", mode: str = "peers", prefix: str = "") -> list[tuple[str, str]]:
    turns = yaml.safe_load((REPO / "prompts/turns.yaml").read_text())
    vis = (REPO / f"prompts/visibility_{visibility}.md").read_text().strip()
    opening = " ".join(turns["opening_requirement"][role].split())
    out = []
    for st in turns["steps_three_rounds"]:
        text = st["text"]
        if "round" in st:
            text = text.replace("{round_messages}", "\n\n".join(round_lines(s, st["round"], role, mode, prefix)))
        text = text.replace("{visibility_block}", vis).replace("{opening_requirement}", opening)
        out.append((st["step"], _swap(" ".join(text.split()) if "round" not in st and "\n" not in text else text, s).strip()))
    return out


def esc(t: str) -> str:
    return html.escape(t)


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e4e1d8;--accent:#2f5d8a;--warn:#a2461f;--ok:#2e6b3a;--quote:#f3f1ea}
@media (prefers-color-scheme:dark){:root{--bg:#171715;--fg:#ecebe6;--muted:#a19f97;--card:#201f1c;--line:#36342f;--accent:#8db6e0;--warn:#e39a74;--ok:#8fc79b;--quote:#26251f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:24px 16px 80px}h1{font-size:1.6rem;margin:.2em 0}h2{margin-top:2.2em;border-bottom:2px solid var(--line);padding-bottom:.2em}
h3{margin-top:1.6em}.muted{color:var(--muted)}nav a{margin-right:12px;color:var(--accent)}
table{border-collapse:collapse;width:100%;margin:.6em 0;font-size:.92rem}th,td{border:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left}
th{background:var(--quote)}td.n{white-space:nowrap;color:var(--muted);font-variant-numeric:tabular-nums}
pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px;font:13px/1.5 ui-monospace,Menlo,monospace;overflow-x:auto}
details{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px;margin:.6em 0}summary{cursor:pointer;font-weight:600}
.ok{color:var(--ok)}.warn{color:var(--warn)}.pill{display:inline-block;padding:1px 8px;border-radius:10px;background:var(--quote);font-size:.85rem}
.step{font-weight:600;color:var(--accent);margin-top:1em}.rec{border-left:4px solid var(--accent);padding:4px 12px;background:var(--card)}
"""


def render(scenarios: list[dict], problems: list[str]) -> str:
    h = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         f"<title>Study 1 Scenarios</title><style>{CSS}</style></head><body><main>",
         "<h1>Study 1 scenarios (approved 2026-10-30)</h1>",
         '<p class="muted">Generated by <code>scripts/review_scenarios.py</code> from '
         "<code>scenarios/study1/*.yaml</code>. Pressure level 4 only. Approved by owner 2026-10-30.</p>",
         "<nav>" + "".join(f'<a href="#{s["id"]}">{s["id"]}</a>' for s in scenarios)
         + '<a href="#speaker-free">speaker-free wording</a><a href="#checks">checks</a></nav>']

    h.append("<h2>Overview</h2><table><tr><th>Scenario</th><th>Subject</th><th>Dead claimant</th>"
             "<th>Werewolf</th><th>Alternative target</th><th>Confederates</th><th>Werewolf's evening move</th><th>Alternative's only post</th></tr>")
    for s in scenarios:
        c = s["cast"]
        move = s["evening"].split(". ")[1]
        quiet = re.search(rf'{c["alternative"]} (?:posted|replied)[^.]*\.', s["evening"]).group(0)
        h.append(f'<tr><td><a href="#{s["id"]}">{s["id"]}</a></td><td>{c["subject"]}</td><td>{c["claimant"]}</td>'
                 f'<td>{c["werewolf"]}</td><td>{c["alternative"]}</td><td>{", ".join(c["confederates"])}</td>'
                 f"<td>{esc(move)}.</td><td>{esc(quiet)}</td></tr>")
    h.append("</table>")
    h.append("<p>Kept fixed across scenarios (so scenario varies only the content): 9 players, one Werewolf, one Minion, "
             "level-4 structure (5 confederates, 3 rounds, counterclaim in round 1, follow-ups, support lines, "
             "vote declarations), the rule text, moderator messages and visibility wording. Argument ids match "
             "the calibration bank, so slot functions can be compared across scenarios.</p>")

    for s in scenarios:
        c = s["cast"]
        h.append(f'<h2 id="{s["id"]}">{s["id"]}: {c["subject"]} (subject), {c["werewolf"]} (Werewolf), {c["alternative"]} (target)</h2>')
        h.append(f'<p><b>Evening chat.</b> {esc(s["evening"])}</p>')
        h.append(f'<p><b>Night.</b> {c["claimant"]} was killed by the Werewolf. '
                 f'Seer role: checked {c["werewolf"]} → Werewolf; {c["claimant"]}\'s final message '
                 f'<i>"{esc(s["final_message"]["seer"])}"</i>. Villager role: {c["claimant"]}\'s final message '
                 f'<i>"{esc(s["final_message"]["villager"])}"</i></p>')
        h.append("<h3>Level-4 texts</h3><table><tr><th>id</th><th>Seer</th><th>Villager</th><th>Filler (none)</th><th>words S/V/F</th></tr>")
        for r in (1, 2, 3):
            h.append(f'<tr><th colspan="5">Round {r}</th></tr>')
            for aid in ROUND_IDS[r]:
                a = s["by_id"][aid]
                same = a["seer"] == a["villager"]
                vill = '<span class="muted">(same)</span>' if same else esc(a["villager"])
                h.append(f'<tr><td><code>{aid}</code></td><td>{esc(a["seer"])}</td><td>{vill}</td><td>{esc(a["filler"])}</td>'
                         f'<td class="n">{words(a["seer"])}/{words(a["villager"])}/{words(a["filler"])}</td></tr>')
        h.append('<tr><th colspan="5">Vote declarations (after round 3)</th></tr>')
        for d in s["vote_declarations"]:
            h.append(f'<tr><td><code>{d["id"]}</code></td><td colspan="2">{esc(d["text"])}</td><td>{esc(d["filler"])}</td>'
                     f'<td class="n">{words(d["text"])}/–/{words(d["filler"])}</td></tr>')
        h.append("</table>")
        for role in ("seer", "villager"):
            h.append(f"<details><summary>System prompt — {role}</summary><pre>{esc(system_prompt(s, role))}</pre></details>")
        for role in ("seer", "villager"):
            body = "".join(f'<div class="step">{esc(name)}</div><pre>{esc(text)}</pre>' for name, text in steps(s, role))
            h.append(f"<details><summary>Full peers transcript (moderator messages) — {role}, public vote, example slot order</summary>"
                     f'<p class="muted">Example order: counterclaimer in slot 3 ({c["confederates"][2]}); '
                     f"<code>{DROPPED_IN_EXAMPLE}</code> is the round-1 argument dropped for that slot. The seed varies both.</p>{body}</details>")

    s = scenarios[0]
    c = s["cast"]
    h.append('<h2 id="speaker-free">Speaker-free wording (sensitivity analysis)</h2>')
    h.append("<p>Calibration's counterclaim was first person (\"I'm the real Seer. I checked Drew…\"), "
             'so the speaker-free version read <i>"A note appears: I\'m the real Seer…"</i>. '
             "These drafts rewrite every line, including the counterclaim and follow-ups, <b>without first person</b> "
             "(the same rule as the approved vote declarations), so one text works under both attributions. "
             "The open choice is the prefix that replaces the speaker's name.</p>")
    claim = s["by_id"]["cc_claim"]
    h.append("<table><tr><th>Condition</th><th>Counterclaim as delivered (s1, villager)</th></tr>")
    h.append(f'<tr><td>Peers</td><td><code>{esc(c["confederates"][2])}: {esc(claim["villager"])}</code></td></tr>')
    for key, prefix in SPEAKER_FREE_OPTIONS.items():
        h.append(f"<tr><td>Speaker-free, option {key}</td><td><code>{esc(prefix + claim['villager'])}</code></td></tr>")
    h.append("</table>")
    h.append('<div class="rec"><p><b>Approved by owner (2026-10-30): option B, "Unsigned message: ".</b> '
             "With a claim in the mix there is no reading in which nobody wrote the message, so "
             '"A note appears" leaves a Seer claim with no plausible source. '
             '"Unsigned message" says honestly what the condition removes: the sender\'s identity, '
             "and with it any sense of how many different players are talking. "
             "The text is byte-identical to peers; only the label before the colon changes "
             "(the fidelity check stays the same). It also reads naturally for the other lines.</p>"
             "<p>Either way, as SPEC v3 notes, a Seer claim implies a claimant, so the speaker-free condition is not a "
             "clean test of attribution. It is a sensitivity analysis.</p></div>")
    h.append("<details><summary>Round 1, s1 villager, all three deliveries</summary>")
    for label, mode, prefix in (("Peers", "peers", ""), ("Speaker-free (B)", "speaker-free", SPEAKER_FREE_OPTIONS["B"]),
                                ("Speaker-free (A)", "speaker-free", SPEAKER_FREE_OPTIONS["A"]), ("None (filler)", "none", "")):
        text = dict(steps(s, "villager", mode=mode, prefix=prefix))["pressure_round_1"]
        h.append(f'<div class="step">{label}</div><pre>{esc(text)}</pre>')
    h.append("</details>")

    h.append('<h2 id="checks">Rule checks</h2>')
    h.append("<ul><li>Each scenario has every level-4 id (5 round-1 arguments, claim, 2 follow-ups, 8 support lines) and 5 declarations.</li>"
             "<li>No first person anywhere (arguments, claim, follow-ups, declarations, fillers).</li>"
             "<li>Arguments address the subject as \"you\", never by name, never call the subject the Werewolf, and never mention vote visibility.</li>"
             "<li>Each text names only its own scenario's cast; no name repeats across scenarios or from calibration; no two cast members share an initial.</li>"
             "<li>Fillers name no player or role and no claim or check; within ±3 words of both variants (declarations ±1).</li>"
             "<li>Seer and Villager variants within 6 words of each other.</li></ul>")
    if problems:
        h.append(f'<p class="warn"><b>{len(problems)} problem(s):</b></p><ul>' + "".join(f"<li>{esc(p)}</li>" for p in problems) + "</ul>")
    else:
        h.append('<p class="ok"><b>All checks pass.</b></p>')
    h.append("</main></body></html>")
    return "\n".join(h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO / "runs" / "study1_scenarios_review.html"))
    args = ap.parse_args()
    scenarios = load()
    problems = check(scenarios)
    Path(args.out).write_text(render(scenarios, problems))
    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(scenarios)} scenarios, {len(problems)} problem(s); wrote {args.out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
