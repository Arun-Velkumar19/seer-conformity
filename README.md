# Seer Conformity

Do LLM agents abandon verified knowledge under social pressure?

Arizona State University, Barrett Honors College. HON 492 / HON 493 thesis project.

## Research question

In a stripped-down Werewolf scenario, an AI agent playing the **Seer** has *verified* that a player is a werewolf. Scripted confederate agents push back with plausible in-game arguments. Does the Seer abandon what it knows and vote with the group?

Prior work shows LLM conformity on quiz tasks and normative conformity on preference tasks with no correct answer. This project tests normative conformity when the agent **knows the correct answer for certain**, in a multi-agent setting where conforming costs its team.

Key comparisons:

- **Seer vs. Villager:** conformity beyond what the arguments justify.
- **Public vs. secret ballot:** normative conformity (being watched).
- **Seer vs. no-speaker control:** social pressure vs. plain revision under questioning.

## Current stage: feasibility spike

Engineering requirements live in [`SPEC.md`](SPEC.md), which is the authoritative specification. The spike checks that the scenario is believable, that conformity is measurable (not stuck at 0% or 100%), and that the manipulations change behavior.

## Repository layout

```text
SPEC.md         authoritative spec
configs/        condition definitions
confederates/   confederate message bank (+ review notes)
prompts/        subject system prompts per role/visibility
sim/            scenario runner, structured vote tool, outcome labeling
runs/           JSONL transcripts (gitignored) + summary.csv
scripts/        run_spike.py, summarize.py, sample_transcripts.py
```

## Team workflow

- **Owner (Arun):** final authority on methodology, scope, major architecture, merges, and deployment.
- **Claude Code:** primary engineer for implementation, tests, and technical docs.

Rules:

- No merges without explicit owner approval.
- Never commit API keys, credentials, or private experimental secrets.
- Raw run transcripts stay local; `runs/summary.csv` is committed.

### Changes that require owner review before use

- The confederate message bank (plausibility review is a spike pass criterion).
- Subject system prompts, including the vote-visibility wording.
- Condition definitions, outcome labels, and the conformity metric.

## History

This repository previously held the form accessibility agent spike. That work is preserved on the `archive/form-a11y-spike` branch and the `form-a11y-final` tag.
