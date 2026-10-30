# Seer Conformity

Does public voting make an AI agent vote against verified private evidence under group pressure?

Arizona State University, Barrett Honors College. HON 492 / HON 493 thesis project.

## Research question

In a one-round Werewolf scenario, an AI agent playing the **Seer** has verified that player X is a werewolf. Opposing arguments push it toward a different target. We ask whether **public voting** increases the rate at which the Seer votes against its verified evidence, beyond the effect of the same arguments delivered **without speakers**.

The confirmatory contrast is a difference-in-differences:

```text
Δ = (P_public,peers − P_secret,peers) − (P_public,no-speaker − P_secret,no-speaker)
```

where P is the proportion of initially correct Seer trials whose final vote goes to the confederates' target.

This is a behavioral measure of susceptibility to social framing. It is not a claim about belief change, irrationality, or human-like conformity. A vote is a decision, not a belief.

## Design at a glance

- **Role:** Seer (primary) or Villager (secondary; pressure under uncertain testimony).
- **Visibility:** public (votes announced by name) or secret (totals only).
- **Pressure:** none (neutral filler), no-speaker (unattributed arguments), or peers (the same arguments from 3 named, scripted confederates).

## Current stage: feasibility spike

Engineering requirements live in [`SPEC.md`](SPEC.md), which is the authoritative specification. The spike checks that the protocol runs, that the conditions are delivered as intended, and that the outcome varies enough to measure. Effect estimates from the spike are exploratory.

Models: the spike uses `gpt-6-luna`; the full study adds `gpt-6.1-sol`, `claude-haiku-5-5`, `claude-sonnet-5-5` and an open-weight Qwen model (not yet configured). Each model's provider and effort level are set in [`configs/models.yaml`](configs/models.yaml) and logged with every run.

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

Then `python scripts/summarize.py` writes `runs/summary.csv`, and `python scripts/sample_transcripts.py` writes 10 peer transcripts to `runs/samples.md` for the plausibility read. `python -m unittest discover tests` runs the offline tests.

## Repository layout

```text
SPEC.md         authoritative spec
configs/        condition definitions
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
