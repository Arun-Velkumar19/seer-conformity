# Seer Conformity

Under what conditions does an AI agent vote against its own information because a group of other agents disagrees?

Arizona State University, Barrett Honors College. HON 492 / HON 493 thesis project.

## Research question (SPEC v3)

In a one-round Werewolf game, an AI agent holds information about who the Werewolf is: a verified check (**Seer**) or a dead player's testimony (**Villager**). Five scripted confederates counterclaim Seer and push a different target. Study 1 tests three hypotheses:

- **H1:** the Seer makes fewer evidence-inconsistent votes than the Villager under matched opposing arguments.
- **H2:** within a provider, lower model tiers make more majority-aligned vote revisions than higher tiers (Villager role).
- **H3 (secondary):** Villagers revise toward the majority more under public than secret votes.

The outcome is behavioral: a vote is a decision, not a belief. The original v2 question (a Seer × visibility difference-in-differences) is superseded, because no Seer voted against its check in calibration (0 of 240 runs). It is kept in [`SPEC.md`](SPEC.md), Part B.

## Design at a glance

- **Study 1:** pressure level 4 (counterclaim plus Minion), 5 independently written scenarios, peers only. 6 models × 2 roles × 2 visibilities × 5 scenarios × 10 runs = 1,200 games, plus 200 speaker-free Villager games on `gpt-6-luna`. A dated preregistration is in [`SPEC.md`](SPEC.md).
- **Before it (exploratory):** the feasibility spike (pressure level 0) and calibration (levels 1-4), kept as run.

Models: the spike uses `gpt-6-luna`. Study 1 (SPEC v3) uses `gpt-6-luna`, `gpt-6.1-terra` and `gpt-6.1-sol`, plus `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5` once credits are in place (these run only with `--model`). Each model's provider and effort level are set in [`configs/models.yaml`](configs/models.yaml) and logged with every run.

## Running

Install into your own virtual environment (Python 3.11+):

```bash
pip install -r requirements.txt
```

Test the whole pipeline with a deterministic fake model (no API keys, no cost):

```bash
python scripts/run_spike.py --dry-run && python scripts/summarize.py --dry-run
```

Real runs read `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from your environment (the SDKs read them; nothing in this repo does):

```bash
python scripts/run_spike.py
```

Then `python scripts/summarize.py` writes `runs/summary.csv`, and `python scripts/sample_transcripts.py` writes 10 peer transcripts to `runs/samples.md` for the plausibility read. `python scripts/review_texts.py` writes every experiment text to `runs/review_texts.md` for review. `python -m unittest discover tests` runs the offline tests.

Calibration (exploratory; the spike hit a floor of 0 caves in 180 runs): pressure levels 1-4 add vote declarations, five confederates, a round of isolation pressure, and a Seer counterclaim ([`configs/pressure_levels.yaml`](configs/pressure_levels.yaml)). Records go to `runs/calibration.jsonl`; `runs/spike-v2.jsonl` stays as level 0. For example, level 1 on two cells, 10 runs each:

```bash
python scripts/run_spike.py --config configs/calibration.yaml --pressure-level 1 --cells seer-public-peers,villager-public-peers
```

`--runs N` overrides the 10 runs per cell (up to 30); runs already logged are kept and only new run indexes are added. `python scripts/summarize.py --config configs/calibration.yaml` summarizes each level separately into `runs/calibration_summary.csv`.

Study 1 (SPEC v3, preregistered): pressure level 4, five approved scenarios ([`scenarios/study1/`](scenarios/study1/)), runs fixed by the allocation in [`configs/study1.yaml`](configs/study1.yaml). Records go to `runs/study1.jsonl` only. Check everything with the fake model first:

```bash
python scripts/run_spike.py --config configs/study1.yaml --dry-run && python scripts/summarize.py --config configs/study1.yaml --dry-run
```

Then one model at a time, for example:

```bash
python scripts/run_spike.py --config configs/study1.yaml --model gpt-6-luna
```

`--scenarios s1,s2` and `--limit N` run part of the allocation (the rest is added when rerun); `--show-prompts --scenario s2 --cell villager-public-peers` prints one rendered game. `python scripts/summarize.py --config configs/study1.yaml` writes `runs/study1_summary.csv` (per model, scenario and cell, plus pooled) and prints the checks. `python scripts/review_scenarios.py` renders all scenario texts for review.

## Repository layout

```text
SPEC.md         authoritative spec
configs/        condition definitions, pressure levels, Study 1 allocation
scenarios/      Study 1 scenario files (cast, evening chat, level-4 texts, fillers)
confederates/   argument bank + filler bank + review notes
prompts/        rules, role, and visibility prompt text
sim/            runner, vote tool, outcome labeling
runs/           JSONL (gitignored) + summary.csv
scripts/        run_spike.py, summarize.py, sample_transcripts.py
tests/          offline tests (fake model, stub API clients)
```

## Team workflow

- **Owner (Arun):** final authority on methodology, scope, major architecture, merges, and deployment.
- **Claude Code:** primary engineer for implementation, tests, and technical docs.

Rules:

- No merges without explicit owner approval.
- Never commit API keys, credentials, or private experimental secrets.
- Per-run JSONL stays local; `runs/summary.csv` is committed.

### Changes that require owner review before use

- The argument bank and filler bank (plausibility review is a spike check).
- Rules, role, and visibility prompt text.
- Condition definitions, outcome labels, and the analysis contrast.

## History

This repository previously held the form accessibility agent spike. That work is preserved on the `archive/form-a11y-spike` branch and the `form-a11y-final` tag.
