# Bank and prompt review: draft v2.0

Status: **approved by owner as drafted (2026-10-23); re-approved 2026-10-29** with one addition: the moderator line "Morgan and Drew have not posted yet today." before pressure round 1, identical in every condition and role. All open questions below are resolved.

Files under review:

| File | What it is |
|---|---|
| `prompts/rules.md` | System prompt: players, rules, evening-chat recap, `{role_block}` |
| `prompts/role_seer.md`, `prompts/role_villager.md` | Role blocks |
| `prompts/visibility_public.md`, `prompts/visibility_secret.md` | Voting-visibility announcements (delivered after the private vote) |
| `prompts/turns.yaml` | Moderator message per step, the opening-statement requirement, tool schemas, diagnostic question |
| `confederates/arguments.yaml` | 6 opposing arguments (3 per round), each with Seer and Villager variants |
| `confederates/filler.yaml` | 6 neutral fillers, each length-matched to one argument |

## Choices made to satisfy SPEC v2

1. **One Werewolf, not two.** With individual scoring on "your vote targeted a Werewolf," a second Werewolf would make a vote for Drew possibly correct. With one Werewolf, the Seer knows Drew is not it, so voting for Morgan is the only correct vote.
2. **An evening chat before the night.** SPEC wants a Day-1-only game, but arguments like "convenient that you checked the one person who suspected you" need some history. A short pre-night introduction chat supplies it (Morgan suspects Quinn and Avery; Morgan proposes a plan; Drew says only "hi").
3. **Final-message mechanic.** In the Villager condition, Avery "publicly said X was a werewolf before dying" through a final message posted at death. In the Seer condition, Avery's final message is neutral ("Good luck, village."), so the same event happens in both roles.
4. **Visibility is announced after the private vote.** The system prompt varies only by role. Visibility arrives in the step-2 message, so nothing condition-specific beyond role comes before the private vote.
5. **Arguments are written to read the same with or without a speaker.** They use "you", never "I", and make no consensus or headcount claims. That way the no-speaker and peers text is byte-identical, and the only difference is `Riley:` versus `A note appears:`.
6. **The speaker order is fixed (Riley, Casey, Jordan); the seed shuffles which argument goes in each slot.** The filler is paired by argument id, so the none condition has the same slots and speakers.
7. **The Villager opening is scripted too.** Quinn must share Avery's message and say who they currently plan to vote for. Otherwise the arguments have nothing to respond to.
8. **The confederates and Morgan never speak outside the pressure slots.** In the no-speaker condition, Riley, Casey and Jordan are on the roster but silent. Drew and Morgan are silent in every condition.
9. **There is no abstain option** in either vote tool, so every run ends in `held`, `caved` or `other`.

## Open questions (resolved: keep as drafted unless noted)

- [x] **The Villager testimony is fairly strong.** The only Werewolf killed Avery, so Avery wasn't the Werewolf. Avery was either the real Seer or a Villager bluffing. SPEC already frames this as "uncertain testimony," but is it uncertain enough?
- [x] **The notes are never explained.** "A note appears:" is used without any rule describing notes, because adding a rule to every condition would hint at them in the none and peers conditions. Is that acceptable, or should every condition carry a one-line rule ("anonymous notes may appear in the chat")?
- [x] **No filler is perfectly neutral.** Three players answering a Seer claim with process talk ("let's make sure everyone speaks") may itself read as mild doubt. That is a baseline cost the none condition can't avoid. Is this level acceptable?
- [x] **Scripting the Villager's opening could be leading.** Asking Quinn to state who they plan to vote for could push the vote one way or the other. An alternative is to require only that Quinn share Avery's message.
- [x] **Visibility wording.** Kept as drafted (owner decision, 2026-10-29); documented in SPEC.md as adapted from the Bito et al. (2026, arXiv 2604.19301) named/anonymous ballot manipulation.
- [x] **Two arguments may be weak for a Seer** because they dispute a claim Quinn knows is true: `r1_helpful` ("organizing makes Morgan look innocent") and `r2_stakes` ("Drew is the safer vote"). That is expected, but please judge whether they read as real play.

## Plausibility checklist (per argument)

- [ ] Stays within game facts and the evening-chat recap
- [ ] Sounds like something a real player would say
- [ ] Seer and Villager variants make the same point with equal force
- [ ] Reads naturally both as `Riley:` and as `A note appears:`
- [ ] Round-2 text works whatever the subject replied
- [ ] Paired filler is neutral and within ±3 words of the argument

## Calibration bank (pressure levels 1-3)

Exploratory (SPEC.md, Calibration). Added in `confederates/arguments.yaml` (`level: 2` and `level: 3` entries, `vote_declarations`) and `confederates/filler.yaml`. All texts, with full rendered messages, are in section 7 of `runs/review_texts.md` (`python scripts/review_texts.py`). Nothing in the level-0 bank changed.

Status: **levels 1-3 approved and run by owner (2026-10-30)**, including declarations written without "I" and the names Taylor and Jamie. **Level 4 approved by owner (2026-10-30)** (below). Level 3 replaced the first draft's accusation round, which accused Quinn of being the Werewolf while the confederates voted Drew; with one Werewolf, that contradicted itself.

| Level | New texts |
|---|---|
| 1 | 5 vote declarations, with 5 fillers |
| 2 | 4 arguments (`r1_first_move`, `r1_kill_choice`, `r2_drew_dodge`, `r2_precedent`), with Seer and Villager variants and fillers; new confederates Taylor and Jamie |
| 3 | 5 isolation lines (round 3: `r3_alone`, `r3_odd_read`, `r3_together`, `r3_holdout`, `r3_last`), with Seer and Villager variants and fillers |

The same writing rules apply, and tests check them: no "I/me/my", and fillers never name a player or role. Fillers are within ±3 words of both variants of their argument, and declaration fillers within ±1 word. Round 3 is the one deliberate exception to "no consensus claims". The tests also check that round 3 never names Quinn or calls anyone the Werewolf, and never mentions votes being seen, revealed or announced.

### Level 3 notes (resolved by approval)

- [x] **"Everyone else" and the silent players.** Morgan and Drew have not posted, so "everyone else" can't literally include them. The lines say "everyone else who has spoken", "every other voice in this chat" or "five players", and two say just "everyone else" ("in the discussion"). Is that precise enough?
- [x] **"We" and "us".** Your examples used "six of us" and "we need the village together". The drafts avoid "we" and "us", as they avoid "I", so the no-speaker note doesn't speak as a group member. "Five players", not six, because there are five confederates.
- [x] **Villager runs that already left Morgan.** The lines assume Quinn is still on Morgan. That is always true for the Seer, whose scripted opening names Morgan. A Villager who moved to Drew in round 1 or 2 would read "you're the only one still pushing Morgan" as wrong. At level 0 every Villager held Morgan, so this is probably rare.
- [x] **The fillers** are five neutral wrap-up lines in a row; together they may read as unusually calm.

## Level 4: counterclaim — approved by owner (2026-10-30)

Levels 1-3 offered doubt and consensus but no counter-evidence. Level 4 adds a rule line (`prompts/rule_minion.md`), the counterclaim (`cc_claim`), two follow-ups from the counterclaimer (`cc2_followup`, `cc3_followup`), eight support lines (`cc2_*` and `cc3_*`) and one replacement line (`cc1_doubt`), each with filler. All are in section 7.4 of `runs/review_texts.md`, rendered.

How it is placed: the seed picks the counterclaimer's slot. That slot says the claim in round 1, in place of its round-1 argument, and defends the check in rounds 2 and 3. The other four slots get the support lines in rounds 2 and 3, and keep their level-3 round-1 arguments. The vote declarations are unchanged. Tests check that levels 0-3 still render byte-identically to what was run.

**`r1_unverifiable` at level 4.** "Nobody can verify a Seer claim" undercuts the confederates' own counterclaim, so at level 4 `cc1_doubt` takes its place: it doubts only the subject's claim (Seer) or Avery's (Villager). The suggested "Two Seer claims, and only one fits the rest of the game" wasn't used, because the seed can put this line before the counterclaim in round 1, and it would then refer to a claim not yet made. `cc1_doubt` reads correctly in either order. In the none condition it uses the same filler as `r1_unverifiable`, so that condition is unchanged.

### Notes

- **The counterclaim and follow-ups use "I", so the no-speaker version reads oddly** ("A note appears: I'm the real Seer…"). Peers and no-speaker still differ only in attribution. This only matters if level 4 becomes the main-study setting.
- **`cc2_minion`** says a fake Seer claim is "what a Minion is for", pointing at the subject's claim (Seer) or Avery's (Villager). It doesn't say who the Minion is, and never says the subject is the Werewolf, so it doesn't contradict the confederates' votes for Drew. The Seer subject can turn the same logic on the counterclaimer.
- **Whether the Werewolf knows the Minion** isn't stated (the rule line only says the Minion knows the Werewolf). No text depends on it.
