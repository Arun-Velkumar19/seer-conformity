# Bank and prompt review: draft v2.0

Status: **approved by owner as drafted (2026-10-23).** The open questions below are resolved as "keep the draft as written," except the visibility wording check, which is still to do before the main runs.

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
- [ ] **Still open: the visibility wording is my paraphrase.** Check it against Bito et al. (arXiv 2604.19301) before the main runs.
- [x] **Two arguments may be weak for a Seer** because they dispute a claim Quinn knows is true: `r1_helpful` ("organizing makes Morgan look innocent") and `r2_stakes` ("Drew is the safer vote"). That is expected, but please judge whether they read as real play.

## Plausibility checklist (per argument)

- [ ] Stays within game facts and the evening-chat recap
- [ ] Sounds like something a real player would say
- [ ] Seer and Villager variants make the same point with equal force
- [ ] Reads naturally both as `Riley:` and as `A note appears:`
- [ ] Round-2 text works whatever the subject replied
- [ ] Paired filler is neutral and within ±3 words of the argument
