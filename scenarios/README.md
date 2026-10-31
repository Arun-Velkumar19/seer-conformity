# Scenarios

`study1/s1.yaml` … `s5.yaml`: the five Study 1 scenarios (SPEC.md, Study 1 design). **Approved by owner (2026-10-30)**, with the speaker-free label "Unsigned message: ". Changes after approval go in SPEC.md, Deviations.

Each file supplies everything that names a player; the rule text, moderator messages, visibility wording and level-4 structure stay shared.

| Key | What it is |
|---|---|
| `cast` | `subject` (the model's player), `claimant` (killed at night; the Villager role's Seer testimony), `werewolf` (the true Werewolf, which the Seer checked), `alternative` (the confederates' target; silent all game), `confederates` (5 speaking players, in slot order) |
| `evening` | The evening-chat recap in the system prompt |
| `final_message` | The claimant's final message: neutral (Seer role) or the Seer claim on the Werewolf (Villager role) |
| `arguments` | Level-4 texts, with calibration's ids and slot functions: 5 round-1 arguments (`cc1_doubt` takes `r1_unverifiable`'s place), the counterclaim `cc_claim`, follow-ups `cc2_followup` / `cc3_followup`, and 4 support lines per round in rounds 2-3. Each has `seer` and `villager` variants (or `both`) and a length-matched `filler` |
| `vote_declarations` | 5 declarations naming the alternative target, each with a filler |

Writing rules, checked by `python scripts/review_scenarios.py` (which also writes the review page, `runs/study1_scenarios_review.html`):

- **No first person** in any line, including the counterclaim and follow-ups, so a text reads the same under peers and speaker-free delivery.
- **Arguments** address the subject as "you", never call the subject the Werewolf, and never mention vote visibility.
- **Names:** each text names only its own scenario's cast. No name repeats across scenarios or from calibration.
- **Fillers** name no player, role, claim or check. Each is within ±3 words of both variants of its argument (±1 for declarations).
