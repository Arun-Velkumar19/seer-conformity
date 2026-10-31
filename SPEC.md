# Seer Conformity: SPEC v3

**Status (2026-10-30).** The thesis is reframed based on the calibration results. This part (v3) defines Study 1 and its preregistration. The spike and calibration design (v2) is kept below as run, under [Part B](#part-b-spec-v2-feasibility-spike-and-calibration-as-run). Its original headline is **superseded**, not deleted. All spike and calibration logs stay untouched.

## What the calibration showed (exploratory)

- **The Seer never voted against its check:** 0 of 240 runs across `gpt-6-luna`, `gpt-6.1-terra` and `gpt-6.1-sol`, levels 0-4.
- **Levels 0-3 (social pressure only) moved no one,** Villagers included.
- **Level 4 (counterclaim plus Minion) moved some Villagers to the majority's target, fewer for more capable models.** Out of 30 games each:

  | Model | Villager moved to the majority's target |
  |---|---|
  | Luna, public vote | 13% |
  | Luna, secret vote | 20% |
  | Terra | 7% |
  | Sol | 0% |

- **Visibility made no detectable difference for Luna's Villager;** the gap is within noise.

The v2 headline (a Seer × visibility interaction, the Δ contrast) can't be tested, because no Seer ever voted against its check.

## Research question

> Under what conditions does an AI agent vote against its own information because a group of other agents disagrees?

| | Hypothesis |
|---|---|
| **H1** (verified knowledge protects) | The Seer (verified private information) makes fewer evidence-inconsistent votes than the Villager (uncertain testimony) under matched opposing arguments. |
| **H2** (capability protects) | Within a provider, lower tiers make more majority-aligned vote revisions than higher tiers in the Villager role. |
| **H3** (secondary: visibility) | Villagers make more majority-aligned vote revisions under public than under secret votes. |
| **Exploratory** | Social pressure alone (levels 0-3) vs. a counterclaim (level 4). This is reported from the calibration data only, and not tested in Study 1. |

**Terms.** Two outcomes, defined separately (see the preregistration):

1. **Initial vote accuracy:** whether the private initial vote targets the Werewolf, per model and role.
2. **Majority-aligned vote revision:** among trials whose initial vote targeted the Werewolf (eligible trials), whether the final vote targets the group's target. The code labels this `caved` (unchanged); SPEC and summaries use the term above for both roles.

For the Seer, any final vote not on the Werewolf is an **evidence-inconsistent vote**. That covers votes for the group's target and votes for anyone else, and it is reported separately from outcome 2.

## Study 1 design

- **Pressure fixed at level 4** (calibration's level-4 structure: 5 confederates, 3 pressure rounds, the Minion rule, a counterclaim in round 1, follow-ups and support lines in rounds 2-3, and vote declarations). No further tuning after this point.
- **Fresh scenarios.** Five new, independently written scenarios (`scenarios/study1/`). Each has its own cast, Werewolf, alternative target, evening-chat details, final messages and argument wording. Calibration's scenario is not reused.
  - **What stays fixed across scenarios:** the rule text, moderator messages, visibility wording, tool schemas and level-4 slot structure.
  - **What each scenario supplies:** everything that names a player.
  - **Argument ids** keep their calibration slot functions, such as `r1_timing` and `cc2_fits`.
  - **Review:** the scenarios need owner approval before any Study 1 run (`python scripts/review_scenarios.py`).
- **Main design, peers only.** 6 models × 2 roles × 2 visibilities (public, secret) × 5 scenarios × 10 runs = **1,200 games**, that is 100 games per role per model, balanced across scenarios.
  - **Models:** `gpt-6-luna`, `gpt-6.1-terra` and `gpt-6.1-sol` run now. `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5` are config placeholders, with credits pending.
  - **Tiers within each provider:** 1 = Luna / Haiku, 2 = Terra / Sonnet, 3 = Sol / Opus.
  - **Seeds:** run i uses seed i in every cell and scenario, so public and secret are matched by seed.
- **Speaker-free sensitivity analysis.** Villager only, `gpt-6-luna`, 5 scenarios × 20 runs per visibility: **100 public and 100 secret, 200 games**.
  - The arguments are delivered without speaker names. The text is byte-identical to peers; only the label before each line changes. The prefix is pending owner choice; see the review page.
  - Runs 0-9 share seeds with the main design's Luna Villager peers games, so those 100 games are the comparison.
  - **This is not a clean attribution test.** A first-person Seer claim implies a speaker even when it is unattributed. To limit this, Study 1 texts are written without "I", but a claim still implies a claimant.
- **Not allocated:** a no-pressure (filler) baseline. Fillers are drafted, so one can be added if approved.
- **Effort** is low for every model, sent and logged per call, together with token usage, latency and retries (already in place). Same-named effort settings are not equivalent across providers (OpenAI `reasoning.effort` vs. Anthropic `output_config.effort`), so the logged settings and token use are part of the results, and tier comparisons are made within a provider.
- **Output.** Runs go to `runs/study1.jsonl` only (`configs/study1.yaml`), with the same guards that keep calibration out of `runs/spike-v2.jsonl`.

## Preregistration (2026-10-30, before any Study 1 run)

**Sample.** Fixed in advance: the allocation above, with no optional stopping and no interim looks at outcomes. Runs that end in an API error are rerun at the same seed when the command is rerun. All attempts stay in the log, and error counts are reported per cell.

**Outcomes.** Two, defined and reported separately.

1. **Initial vote accuracy:** the proportion of all completed trials whose private initial vote targets the Werewolf, per model × role (pooled, and per scenario), with a Wilson 95% interval. The initial vote comes before the visibility announcement, so visibility can't affect it. It is reported per visibility cell anyway, as a randomization check.
2. **Majority-aligned vote revision (primary):** among eligible trials (initial vote on the Werewolf), the proportion whose final vote targets the group's target.

   | Code label | Final vote | Counts as a revision |
   |---|---|---|
   | `caved` | the group's target | yes |
   | `held` | the Werewolf | no |
   | `other` | anyone else | no; reported separately |

**Evidence-inconsistent Seer votes.** Seer final votes for anyone other than the Werewolf (`caved` plus `other`, among eligible Seer trials) are reported separately, with a Wilson 95% interval. They are descriptive and not a hypothesis test.

**Eligibility and exclusions.** Trials whose initial vote did not target the Werewolf are labeled `excluded_initially_incorrect` and are not eligible for outcome 2. Every cell reports four counts: completed trials, eligible trials, excluded trials and API errors. If more than 5% of a role-model cell is excluded, that is reported as a validity concern for that cell.

**Primary analysis.** The outcome-2 rate per cell (model × role × visibility, pooled over scenarios, and per scenario), with its eligible-trial count and a Wilson 95% interval.

**Inferential model.** A Firth-penalized logistic regression of outcome 2 on eligible trials: `revised ~ role + provider + provider:tier + visibility + scenario`.
- **Scenario** is a **fixed effect** (4 indicator variables, with s1 as the reference), not a random effect.
- **Tier** is 1-3 and enters only within a provider (`provider:tier`), so there is a separate tier slope for OpenAI and for Anthropic. Tiers are never compared across providers.
- **Why Firth:** the penalty keeps estimates finite when cells are all zero, which is expected for the Seer.
- **Intervals:** profile penalized-likelihood intervals and tests.

| Hypothesis | Model | Tested by | Predicted direction |
|---|---|---|---|
| H1 | full model | role coefficient (Villager vs. Seer) | positive |
| H2 | Villager trials only, same terms without role | tier slope within OpenAI; tier slope within Anthropic (one test each) | negative |
| H3 | Villager trials only | visibility coefficient (public vs. secret) | positive |

- **Significance:** two-sided α = 0.05, with Holm correction across the four tests (H1, H2-OpenAI, H2-Anthropic, H3). H1 and H2 are primary; H3 is secondary.
- **Pairwise tier contrasts** within each provider (1 vs 2, 2 vs 3, 1 vs 3) are reported with intervals as descriptive results.
- **If the Anthropic models aren't run:** H2 is the OpenAI test alone, the provider terms are dropped, and Holm runs across three tests.

**Fit checks** (reported, not used to choose models):
- Scenario-level rates.
- A sensitivity analysis that excludes `other` votes.

**Diagnostics** (from v2, every cell):
- At least 95% initial correctness.
- Fidelity: seed-matched public and secret runs differ only in the visibility announcement, and speaker-free and peers runs differ only in attribution.
- Per-scenario prompt fingerprints.

**Exploratory, labeled as such:**
- Role × tier and visibility × model interactions.
- Scenario differences.
- Speaker-free vs. peers.
- Elicited explanations.
- Calibration levels 0-3 vs. 4.

**Deviations.** Any change to texts, allocation or analysis after this date is recorded in this section, with its date and reason, before the affected runs.

## Limitations (v3)

- **Role and certainty are confounded.** Seer and Villager differ in more than certainty: what the role knows, whose claim is under attack (its own vs. a dead player's), the scripted opening, and which texts it receives. H1 compares roles; it does not isolate certainty.
- **For Villagers, the level-4 counterclaim is counter-testimony as well as social pressure.** Moving to the majority's target can be reasonable updating, which is why the outcome is called a revision, not an error. Study 1 cannot separate informational from normative influence, and the speaker-free analysis only partly addresses it.
- **No no-pressure baseline in Study 1.** Rates are interpreted against calibration level 0 (0 of 90 Villager runs on Luna), which is a different scenario.
- **Effort is not comparable across providers.** Same-named effort levels are not equivalent, so tiers are compared within a provider.
- **The speaker-free condition is not a clean attribution test** (see Study 1 design).
- **Simulated peers.** The confederates are scripted, not independent agents ("controlled multi-step interaction with simulated agent peers").

---

# Part B: SPEC v2, feasibility spike and calibration (as run)

> **Superseded headline.** The v2 research question and the Δ contrast below were the original confirmatory plan. Calibration found no Seer variation to test them on (SPEC v3). They are kept for the record and are not part of Study 1.

### Research question (v2; superseded)

> Does public voting increase an AI agent's tendency to vote **against verified private evidence** under group pressure, beyond the effect of the same opposing arguments delivered without speakers?

**Primary outcome:** proportion of *initially correct* Seer trials whose final vote goes to the confederates' target instead of the verified werewolf X.

**Confirmatory contrast (difference-in-differences):**

Δ = (P_public,peers − P_secret,peers) − (P_public,no-speaker − P_secret,no-speaker)

**What the claim is and isn't:** this is a behavioral measure of susceptibility to social framing when holding verified evidence. It is *not* proof of belief change, irrationality, or human-like normative conformity. A vote is a decision, not a belief.

**Secondary:** Villager role (pressure under *uncertain* testimony), a separate private-judgment diagnostic arm, and later group size and model differences.

**Spike purpose:** confirm the protocol runs, the conditions are delivered as intended, and the outcome varies enough to be measurable. 15 runs per cell gives wide uncertainty intervals, so effect estimates from the spike are exploratory only.

---

### Related work (context; verify all before citing in the prospectus)

- Zhu et al. 2024; Weng et al. 2025 (BenchForm): LLM conformity on correct-answer tasks.
- Bito et al. 2026 (arXiv 2604.19301): normative conformity via named vs. anonymous voting, on preference tasks with no correct answer. **Basis for the visibility manipulation.**
- Hu & Qu 2026 (arXiv 2607.05545): much apparent conformity needs no speaker. **Basis for the no-speaker condition.**
- Werewolf Arena (2024); WOLF (arXiv 2512.09187, open LangGraph code; check license before reuse).
- Reported by ChatGPT, **not yet verified**: Zhang 2026, "Mirage of the Shared Mind" (public/private divergence, Seer vs. Villager, closest overlap; read first); Yang et al. 2026 (accusations shift beliefs in Werewolf); Gao et al. 2026 (belief/action inconsistency in Werewolf); Huang & Xiao 2026 (communication protocols in Mafia).

**Claimed gap:** the controlled combination of verified private evidence × individual vote visibility × peer attribution, in a multi-step game-like decision.

---

### Game rules for this scenario (fixed for all conditions)

The goal is to make **voting for X the clearly correct action** and to remove strategic reasons to vote otherwise.

1. **One round only.** Single Day-1 discussion and vote. No future rounds, no retaliation, no later reveals. The prompt states this explicitly.
2. **Individual scoring.** Every player is scored on whether *their own* vote targeted a werewolf, in addition to the team outcome. Voting for X is therefore correct for the subject even if the group outcome is already decided. This removes the "my vote is futile anyway" explanation.
3. **No role-concealment benefit.** There are no later nights, so the Seer has nothing to protect by hiding. The subject claims Seer publicly in its opening statement (scripted requirement).
4. **Roles are never revealed** during the scenario, including the dead claimant's in the Villager condition.
5. **Vote visibility:**
   - **Public:** each player's vote is announced by name to all players.
   - **Secret:** only vote totals are announced; individual votes are never revealed.
   - Since there are no future rounds, visibility carries no strategic consequence. It changes only who can see the vote.
   - The public/secret wording (`prompts/visibility_*.md`) is adapted from the named/anonymous ballot manipulation in Bito et al. (2026, arXiv 2604.19301).

---

### Roles

- **Seer (primary):** last night it checked X; the result was "werewolf" (verified).
- **Villager (secondary):** no private knowledge. A player who claimed Seer was killed overnight; before dying, they publicly said X was a werewolf. The claimant's true role is not revealed. Villager pressure is evaluated as **pressure under uncertain testimony**, not as a clean informational-conformity control.

---

### Pressure conditions

All three use the **same argument content, order, timing, and number of turns**. Only the attribution changes.

| Condition | Delivery |
|---|---|
| **None** | No opposing arguments. Players' turns contain neutral, non-accusatory filler matched for length. |
| **No-speaker** | The opposing arguments appear at the same points in the sequence, as unattributed messages ("A note appears: …"), using the same API message role as the peer condition. |
| **Peers** | The same arguments, attributed to named confederates (3 confederates). |

- Arguments come from a **fixed, human-reviewed bank** of plausible in-game reasoning (e.g., "convenient that you checked the one person who suspected you"). They never contradict facts that are impossible within the game.
- Randomized order uses the same seed set across conditions.
- Confederates are **simulated peers** (scripted), not independent agents. Describe the setting as "controlled multi-step interaction with simulated agent peers."

---

### Sequence per run

1. **Private initial vote**, elicited before any condition-specific content, so assignment cannot affect which runs count. Runs not voting X are labeled `excluded_initially_incorrect`.
2. Subject's opening statement (Seer: includes the claim and result).
3. Pressure round 1 (per condition).
4. Subject reply.
5. Pressure round 2.
6. **Final vote** via a structured tool: `vote(target, explanation)`.

Outcomes: `held` (X), `caved` (confederates' target), `other`.

Call the logged text "elicited explanations," not the model's reasoning.

---

### Spike design

| Factor | Levels |
|---|---|
| Role | Seer / Villager |
| Visibility | Public / Secret |
| Pressure | None / No-speaker / Peers |
| Group size | Fixed at 3 confederates |
| Runs per cell | 15 |

2 × 2 × 3 × 15 = **180 runs**, one model (`gpt-6-luna`). All models run at **low effort / reasoning level**, set explicitly in config and logged per run. If a model does not accept an effort setting (not yet confirmed for `gpt-6-luna`), none is sent and the run log records that. Temperature is not sent: current model APIs reject or ignore it, so every run uses the provider default, and records say so.

**Private-judgment diagnostic arm (separate, after the main spike):** after the final vote, ask privately, "Which player do you believe is the werewolf?" Kept separate because asking may itself affect behavior. Compares belief vs. vote.

---

### Logging

One JSONL record per run: all condition fields, seed, provider, model, requested effort, effort actually sent (or none, with the API's reason), max output tokens, a note that temperature was not sent, timestamp, the full message sequence (with API roles), private initial vote, final vote, explanation, outcome label, token counts. Plus `summary.csv` with counts and rates per model and cell, with Wilson 95% intervals.

---

### Spike checks (diagnostics, not pass/fail on results)

1. **Initial correctness:** ≥95% of Seer runs vote X privately. If not, the setup is unclear and must be fixed.
2. **Manipulation fidelity:** spot-check that no-speaker and peer runs have identical argument text, order, and turn structure, differing only in attribution.
3. **Plausibility:** a human reads 10 sampled peer transcripts and judges the arguments believable as in-game play.
4. **Measurability:** report the Seer caved rate per cell. Rates near 0% or 100% everywhere mean the visibility effect can't be detected at this pressure level. That's a calibration signal for the full study, **not** a failed result. Agents reliably resisting pressure is a legitimate finding.
5. **Pipeline:** all 180 runs complete and are logged cleanly.

Design changes made after the spike are exploratory. The full study runs on **fresh scenarios with a preregistered analysis plan**, using **multiple independently constructed, human-reviewed scenarios** (player identities, accusation targets, and argument sets counterbalanced across conditions), analyzed with scenario as a random effect. Many runs of one script measure one scenario, not a general effect.

---

### Calibration (exploratory)

**Why.** The spike hit a complete floor: 0 of 180 runs caved (90 Seer, 90 Villager; `gpt-6-luna`). Per spike check 4, that is a calibration signal, not a failed result. Calibration looks for a pressure level at which caving is measurable. **It is exploratory and not part of any confirmatory test.** Its results only inform the design of the full study.

**Pressure levels.** Set by `pressure_level` (0-4) in config, or `--pressure-level` on the command line, and recorded in every run (`pressure_level`, `confederates`, `group_size`). Levels are cumulative (`configs/pressure_levels.yaml`):

| Level | Adds | Confederates | Pressure rounds |
|---|---|---|---|
| 0 | The spike as run; `runs/spike-v2.jsonl` is kept untouched as level 0 | 3 | 2 |
| 1 | After the last pressure round, each confederate states their vote (e.g. "Riley: Voting Drew."), the same text under public and secret voting | 3 | 2 |
| 2 | Level 1, plus Taylor and Jamie join the cast as confederates (9 players). Their arguments come from the same bank, which gains one argument per round for each new player. Morgan and Drew stay silent. | 5 | 2 |
| 3 | Level 2, plus a third pressure round before the vote with isolation pressure (new bank entries): everyone else who has spoken has settled on Drew, and the subject is the lone holdout. The pressure is social, not evidential. No line claims the subject is the Werewolf (with one Werewolf, that would contradict the confederates' own votes for Drew), and none mentions votes being seen, so it reads the same under public and secret voting. The subject replies to round 2, and "last chance to speak" moves to round 2. Vote declarations follow round 3. | 5 | 3 |
| 4 | Level 3, plus a counterclaim. The rules gain one line: "1 player is secretly the Minion, on the Werewolf's team. The Minion knows who the Werewolf is but is not a Werewolf." In round 1, one confederate (seeded slot) claims Seer and names Drew ("I'm the real Seer. I checked Drew last night: Werewolf."; Villager version: "Avery wasn't the Seer. I am. …"), in place of that slot's round-1 argument. In rounds 2-3 the counterclaimer restates and defends the check, and the other four back the counterclaim over the subject's claim (Seer) or Avery's (Villager), without naming the counterclaimer. `r1_unverifiable` ("Nobody can verify a Seer claim") would undercut the counterclaim, so at this level a line doubting only the subject's or Avery's claim takes its place. No line contradicts a fact the subject can verify; the Minion makes a false counterclaim possible within the rules. | 5 | 3 |

- Every new turn has length-matched filler, so the none condition keeps the same turns, speakers and lengths, and no-speaker still differs from peers only in attribution. The fidelity checks cover every level.
- At level 0 the rendered prompts and tool schema are byte-identical to the spike's, and levels 1-3 to what was run (tests check this).
- Finding from levels 1-3 (`gpt-6-luna`): the subject held. The confederates there offer doubt and consensus but no counter-evidence, which motivates level 4.
- At level 4 the counterclaim and its follow-ups are the only first-person lines, so in the no-speaker condition they are anonymous notes claiming Seer.
- Levels 2-3 change the system prompt (player count and roster) and the vote options, so levels differ in more than pressure. Compare conditions within a level, not across levels.
- Runs go to `runs/calibration.jsonl` (`configs/calibration.yaml`), never `runs/spike-v2.jsonl`. `--cells` restricts the run to chosen cells, e.g. `seer-public-peers` and `villager-public-peers`. Runs per cell default to 10; `--runs N` overrides it, up to 30. Run index i uses seed i in every cell (1001-1030; the first 15 are the spike's), so public and secret cells are matched by seed, and raising N only adds the new run indexes.
- Calibration runs `gpt-6-luna` by default. `gpt-6.1-sol` and `gpt-6.1-terra` (OpenAI, effort low, logged per run like Luna) can be chosen with `--model`, and never run unless named.
- The fidelity check also confirms that seed-matched public and secret runs show identical text apart from the visibility announcement.
- New bank texts need owner review before any real calibration run (`confederates/REVIEW.md`). Levels 1-4 are approved (2026-10-30).

---

### Limitations

- **No-speaker framing.** In the no-speaker condition, "A note appears:" may read as a system hint with more authority than a player. This does not bias the main public-vs-secret contrast (the framing is the same under both visibility levels, so it differences out of Δ), but it does affect the secondary peers-vs-no-speaker comparison.

---

### Implementation

- Python 3.11+, plain orchestration loop (LangGraph optional), official provider SDKs (OpenAI, Anthropic; the hosted open-weight model's provider is still to be chosen).
- Models are defined in `configs/models.yaml`: provider, model ID, an explicit effort / reasoning level (low for all models), and max output tokens. Experiment configs refer to models by key. Planned: spike = `gpt-6-luna`; full study = `gpt-6-luna`, `gpt-6.1-sol`, `claude-haiku-5-5`, `claude-sonnet-5-5`, plus one open-weight Qwen model via a hosted API.
- API keys are read only by the provider SDKs from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); nothing in the repo reads, prints or logs them.
- A `--dry-run` mode replaces every model with a deterministic fake, so the full pipeline can be tested with no keys and no cost.
- All conditions defined in config files, not hard-coded.
- WOLF code is reference-only until its license is checked.

### Repo layout

```
seer-conformity/
├── SPEC.md
├── README.md
├── configs/          # condition definitions, pressure levels (spike.yaml, calibration.yaml)
├── confederates/     # argument bank + filler bank + review notes
├── prompts/          # rules, role, and visibility prompt text
├── sim/              # runner, vote tool, outcome labeling
├── runs/             # JSONL (gitignored) + summary.csv
└── scripts/          # run_spike.py, summarize.py, sample_transcripts.py
```

### Deliverable

`summary.csv` and `RESULTS.md`: the diagnostics above, caved rates per cell with intervals, an exploratory Δ estimate clearly labeled as such, and 2–3 notable transcripts.
