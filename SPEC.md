# Form Accessibility Agent: Week 1 Feasibility Spike

**Spec version:** v2.1 (owner/research approved). Revisions from v1 are listed in the last section.

## Purpose

Test whether the core thesis approach is technically feasible before committing to the full build:

> An LLM agent restricted to an accessibility-tree observation channel and keyboard input (no screenshots, no DOM, no mouse) can (a) complete fixed versions of short web forms, and (b) detect and localize the specific accessibility defects in their broken twins, with effects that depend on the defect and on the agent's navigation strategy.

This setup approximates the *information channel* available to keyboard-only and screen-reader users. It does **not** simulate a screen reader, and results must not be described as reproducing the experience of people using assistive technology (AT). It has no speech output, no AT verbosity settings or virtual cursor, and none of the browser/AT heuristics a real screen reader applies.

If this works, the approach is technically sound. If it doesn't, we find out now.

**Out of scope this week:** repair generation or suggestions, the browser extension, real-world or production websites, authenticated systems, multi-page workflows. Do not build these yet. Automation only targets the local test server.

---

## Stack

- Python 3.11+
- Playwright (Python), Chromium
- Anthropic SDK (Claude API), model name configurable via env var
- axe-core (via `axe-playwright-python` or injected script) for the baseline comparison
- Local static server for test forms (e.g. a minimal FastAPI or `http.server` app with a `/submit` endpoint), bound to localhost only

---

## Test forms

Six HTML pages: three broken forms and a fixed twin for each. Each form is a short sign-up (name, email, one extra field, submit). The **only** difference between a broken form and its fixed twin is the defect.

| ID | Defect (broken version) | Fixed version |
|---|---|---|
| F1 | Submit button is an icon-only `<button>` with no accessible name (exposed as just "button"). Includes a second unlabeled icon button (e.g. "clear form") so guessing is risky. | Buttons have accessible names (`aria-label` or visible text). |
| F2 | "Plan" selection is a custom `<div>` dropdown that only opens on mouse click: no `tabindex`, no keyboard handlers, no ARIA roles. Required field. | Native `<select>`, or a custom dropdown with correct roles + keyboard support. |
| F3 | Submitting with an invalid email shows a red error message visually, but it is not in a live region, not linked via `aria-describedby`, and focus does not move. | Error is announced (live region or focus moves to it) and linked to the field. |

### Shared form behavior (identical across all six pages)

- Every form validates the email field the same way: syntax (`type=email`) plus a fixed accepted-domain rule, so a well-formed address on an unaccepted domain is rejected with an "email couldn't be verified" error. Outside of F3's defect, validation errors are announced accessibly (live region and `aria-describedby`).
- After a valid submission, every form shows the same success confirmation in a polite live region (e.g. "Thanks, you're signed up."). This lets the agent tell success from failure through the observation channel on every form.
- Page `<title>`, headings, field labels, and success text are identical between twins and contain no hint of form ID, defect, or broken/fixed status.
- Forms are served at neutral URLs that don't reveal identity (e.g. `/form?session=<random>`). File names, IDs, classes, and comments are never exposed to the agent (see Observation boundary).

### Ground truth

Two separate ground truths, never conflated:

1. **Submission ground truth:** the form POSTs to `/submit` and the server records a valid submission (all required fields present and valid). The harness checks the server log, not the agent's claim.
2. **Defect ground truth:** a defect manifest (`forms/manifest.json`, harness-only) lists for each broken form its defect ID, the affected element's identity (field/control plus its role and accessible name or lack of one), the relevant WCAG success criterion, and the accepted localization targets used for scoring. The agent never sees the manifest.

### Harness preflight (before any agent run)

A scripted, non-LLM keyboard-only driver must show, for each pair:
- The fixed form can be completed and submitted with keyboard actions from the allowed tool set, producing a server-verified submission.
- The broken form exhibits its defect as specified (e.g. the F2 dropdown can't be opened or set from the keyboard).
- **F3 missing feedback:** after a failed submit with `email_initial` on F3-broken, the observation channel shows no feedback: no live-region announcement, no focus change, and no change to the email field's accessible description or invalid state. On F3-fixed, the same step does produce the error through the observation channel.
- **Correction sequence:** the F3 correction procedure below, using `ControlOrMeta+A`, works on F3-fixed (and on the other fixed forms) and leaves the email field holding exactly `email_corrected`.

If preflight fails, fix the forms before running the agent. Preflight results are logged.

---

## Agent constraints (the core of the experiment)

### Observation boundary

The agent receives **only** the observation schema below. It never receives screenshots, raw HTML, DOM nodes, CSS, attribute names or values outside the schema, element IDs or classes, file paths, URLs, page source comments, network responses, console output, or raw Playwright/exception text.

**Observation schema (returned after every action):**

| Field | Content |
|---|---|
| `focused` | The focused element as `{role, name, description, value, states}`, taken from Chromium's computed accessibility tree, or the literal `"no focus"` (document body or nothing focused). `states` is limited to: focused, required, invalid, expanded, checked, selected, disabled. |
| `announcements` | Text newly added since the last step inside live regions (`aria-live`, `role="alert"`, `role="status"`), captured by an injected MutationObserver. Content that is `aria-hidden` or not in a live region is never included. |
| `step` | Step number and remaining step budget. |
| `tool_result` | For non-observation tools: a fixed, generic status (`ok`, `invalid_key`, `nothing_focused_to_type_into`). Never a raw error message. |

Implementation notes:
- If `page.locator(":focus").aria_snapshot()` does not expose accessible description, derive `focused` from the Chromium accessibility tree (e.g. CDP `Accessibility.getPartialAXTree` on the focused node) and keep only the schema fields.
- Strip link targets (`/url:`), element references, and any other non-schema output from snapshots before they reach the agent.
- Every observation sent to the model is logged verbatim so leakage can be audited after the fact.

### Observation modes (both run on every form)

- **Browse mode:** `read_page()` is available. It returns the full-page accessibility tree serialized with the same field restrictions as `focused` (roles, names, descriptions, values, states, and the static text exposed in the accessibility tree). This is the closest analogue to a screen reader's browse mode, and it is still not a screen reader. Calls are logged and counted separately.
- **Focus-only mode:** `read_page()` does not exist in the tool list at all. It isn't disabled with an error. The agent only knows what's in `focused`, `announcements`, and `step`. No page title, heading list, landmark list, or page-level summary is provided by any other route.

Browse mode may let the agent find the F3 error in page text even though it was never announced. That's an expected, mode-dependent effect and gets reported as one.

### Actions (tool calls)

| Tool | Behavior |
|---|---|
| `press_key(key)` | One of: `Tab`, `Shift+Tab`, `Enter`, `Space`, `ArrowUp`, `ArrowDown`, `ArrowLeft`, `ArrowRight`, `Escape`, `Home`, `End`, `Backspace`, `Delete`, `ControlOrMeta+A`, `Shift+Home`, `Shift+End`. Key names are passed to Playwright `keyboard.press` as-is; `ControlOrMeta+A` resolves to Control on Linux/Windows and Meta on macOS. Anything else returns `invalid_key` and counts as a step. |
| `type_text(text)` | Types `text` at the caret of the focused element. It does **not** clear existing content. Use `ControlOrMeta+A` then `Backspace`/`Delete` (or `type_text` over a selection) to replace a value. |
| `read_page()` | Browse mode only (see above). |
| `report_barrier(description, where)` | Non-terminal. The agent records a suspected accessibility barrier and keeps going. `where` is its description of the element or point in the flow. Can be called multiple times. |
| `give_up(reason, where)` | Terminal. The agent declares it can't complete the task. |
| `done()` | Terminal. The agent believes the form was submitted successfully. |

### Task prompt and data

- The agent is told it's assisting a keyboard-only user who relies on the accessibility tree (the prompt must not claim the agent *is* a screen reader). It gets the form data, the observation and action rules, and nothing about defects, form IDs, or broken/fixed status.
- **Task data (identical for all six forms):** name, extra-field value, `email_initial` (well-formed but rejected by the accepted-domain rule, so it can't be recognized as invalid from syntax alone, e.g. `alex.rivera@northmail.co`), and `email_corrected` (accepted, e.g. `alex.rivera@northmail.com`). Instruction: enter `email_initial` first. If the site indicates the email is invalid, replace it with `email_corrected` and resubmit.
- **F3 correction procedure:** to correct the email, the agent must move focus back to the email field (e.g. `Shift+Tab`), select the existing value (`ControlOrMeta+A`), remove it (`Backspace` or `Delete`), type `email_corrected`, then navigate to and activate submit. Preflight must confirm this exact sequence works on F3-fixed.
- The system prompt and tool list are identical across all forms. The only difference between modes is whether `read_page` is in the tool list.

**Limits:** step budget of 40 actions per run (every tool call counts). Hitting the budget ends the run.

---

## Runs

- 6 forms × 2 observation modes × 5 runs (LLM output varies) = 60 runs.
- Fresh browser context and a fresh server session per run.
- Run order randomized; model, temperature, and prompt version fixed and logged.
- Log the observation mode on every run.

With 5 runs per cell, results are descriptive feasibility evidence, not statistical tests.

---

## Outcome measures (kept separate)

Each run is scored on three independent axes. None is inferred from another.

1. **Task outcome** (from the server log and terminal action):
   `submitted` (server-verified) / `gave_up` / `budget_exhausted` / `claimed_done_but_not_submitted`.
2. **Defect detection** (broken forms): did the agent emit at least one `report_barrier` or `give_up` that describes an accessibility barrier? Barrier reports on fixed forms count as **false-positive reports**.
3. **Defect localization** (broken forms, among detecting runs): does any report's `where`/`description` match an accepted localization target in the manifest (e.g. for F1: the unnamed submit or icon button; F2: the plan dropdown; F3: the email field or its error/feedback after submit)? Scored with a written rubric applied by a scorer who doesn't see the task outcome. Ambiguous cases are flagged, not forced.

Also recorded per run: steps used, `read_page` count, whether an unnamed control was activated (F1), whether the corrected email was entered (all forms), and whether the agent noticed missing confirmation.

---

## Logging

One JSONL trajectory file per run:
- run ID, form ID, broken/fixed, mode, run index, model, prompt version, timestamp
- every step: action, arguments, the exact observation sent to the model
- final task outcome (see above)
- all `report_barrier` and `give_up` entries with reason and location text
- token counts

Plus one summary CSV across all runs with the three outcome axes as separate columns. Detection and localization scores are added by the scoring step, never by the agent loop.

---

## Baseline

Run axe-core on every form (broken and fixed) in comparable states so the comparison is fair:
- initial page load
- after the scripted preflight's keyboard interaction states (e.g. after focusing or attempting to open the F2 control)
- after a failed submit with `email_initial` (all forms)

Map axe rule IDs to manifest defects with a predefined mapping. Record which defects it flags and any violations it reports on fixed forms. Expected (hypothesis, not assumption): axe flags F1, likely misses F2's keyboard inaccessibility and F3's unannounced error.

---

## Feasibility criteria (defect-specific)

Broken forms aren't all expected to fail submission. Each defect has its own criterion, evaluated per mode.

**Fixed forms (all, both modes):**
- ≥4 of 5 runs per form are server-verified `submitted`. If not, the interface is too hard (spike **fails**; rework the observation/action interface first).
- False-positive barrier reports are tabulated and discussed; they don't by themselves fail the spike.

**F1 (unnamed buttons):** Submission may succeed by guessing, so submission outcome is reported but not required to fail.
- Criterion: in ≥4 of 5 runs, the agent detects the barrier and localizes it to the unnamed control(s).
- Activating an unnamed control is recorded as a guessing behavior.

**F2 (keyboard-inaccessible required dropdown):** Assuming preflight confirms it can't be set by keyboard, no valid submission should be possible.
- Criterion: 0 of 5 server-verified submissions (any submission means a harness or form bug that must be investigated before interpreting results), and in ≥4 of 5 runs the agent detects and localizes the barrier to the plan control.

**F3 (unannounced email error): exploratory, no pass/fail criterion.** The missing feedback itself is established by scripted preflight, not by agent behavior.
- For each mode, report separately: (a) server-verified submission rate, (b) barrier detection and localization (to the email field or its post-submit feedback), (c) recovery attempts (whether the agent entered `email_corrected` and resubmitted), and (d) false completion claims (`claimed_done_but_not_submitted`).
- `claimed_done_but_not_submitted` is never counted as defect detection. It's reported as a false completion claim.
- *Focus-only:* the agent's only cue is the absence of the success confirmation.
- *Browse mode:* the agent may find the error with `read_page` and recover. That's a legitimate, mode-dependent result. Also report how often the error was found via `read_page`.

**Uncaught false successes:** any `done()` without a verified submission is always logged as `claimed_done_but_not_submitted`.

**Overall verdict:**
- **Pass:** the fixed-form criterion holds, the F1 and F2 criteria hold, and preflight confirms F3's missing feedback. F3 agent results are reported as exploratory and don't affect the verdict.
- **Partial pass (worth discussing):** fixed forms pass, but one or more defect criteria miss (e.g. the agent works around a defect real users likely couldn't, or detects without localizing). That's a finding to report, not hide.
- **Fail:** fixed forms don't meet their criterion.

---

## Design checkpoint (required before the agent loop)

Before implementing the agent loop, Claude Code delivers a design checkpoint for owner/research review containing:
1. The complete proposed agent system prompt and task-prompt template, verbatim.
2. The observation schema as implemented, with example observations from F1–F3 broken and fixed.
3. Full tool definitions (names, JSON schemas, descriptions as the model will see them) for both modes.
4. An explanation of how experimental information leakage is prevented: what the agent can and can't observe, how form identity and broken/fixed status are hidden, how tool errors are sanitized, and how logs make leakage auditable.

The forms, server, preflight driver, and axe baseline may be built before the checkpoint. The agent loop may not.

---

## Repo layout

```
form-a11y-agent/
├── SPEC.md
├── forms/            # F1–F3 broken + fixed HTML, manifest.json (harness-only)
├── server/           # localhost server + /submit endpoint + submission log
├── agent/            # observation builder, tools, agent loop, prompt
├── baseline/         # axe-core runner
├── preflight/        # scripted keyboard-only driver
├── runs/             # JSONL trajectories (gitignored) + summary.csv
└── scripts/          # run_spike.py, summarize.py, score.py
```

## Deliverable at end of week

`summary.csv`, a short `RESULTS.md` (verdict against the defect-specific criteria above, with 2–3 example trajectories), the preflight report, and the axe-core baseline table.

---

## Changes from v1

1. **Outcome separation:** task outcome, defect detection, and defect localization are scored as separate axes. Added the non-terminal `report_barrier` tool, a harness-only defect manifest, and a blinded localization rubric.
2. **F3 correction:** task data now has `email_initial` (well-formed but rejected by an accepted-domain rule) and `email_corrected`, with an explicit correction procedure. Added `Home`, `End`, `Backspace`, `Delete`, `ControlOrMeta+A` (cross-platform select-all), `Shift+Home`, `Shift+End`. `type_text` is documented as non-clearing. Preflight verifies the sequence and F3's missing feedback.
3. **Observation boundary:** a fixed observation schema, sanitized tool results, neutral URLs and titles, `read_page` removed (not just disabled) in focus-only mode, and an explicit statement that this doesn't simulate a screen reader.
4. **Feasibility criteria:** "all broken forms must fail" is replaced by per-defect criteria (F1 guessing allowed, F2 must not submit). F3 agent outcomes are exploratory in both modes, with its missing feedback verified by preflight, and false completion claims are never counted as detection.

Unchanged: six forms, two modes, 60 runs. Repair generation, the browser extension, and production sites stay out of scope.
