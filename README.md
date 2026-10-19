# Form A11y Agent

AI-assisted workflow testing and verified remediation of web form accessibility barriers.

Arizona State University, Barrett Honors College. HON 492 / HON 493 thesis project (working software product plus an empirical research paper).

## Research problem

Conventional automated accessibility checkers catch many rule-based WCAG violations, but some barriers only appear during real user workflows. Examples:

- Forms that can't be completed using only a keyboard.
- Buttons or inputs without meaningful accessible labels.
- Error messages that aren't communicated through accessible interfaces.
- Poor focus management during dynamic interactions.
- Multistep flows that become inaccessible after a state change.

## Research objective

Investigate whether an AI-powered workflow-testing system can identify, and help remediate, accessibility barriers that conventional automated testing misses. The agent will be compared against established tools such as axe-core on detection performance, false positives, workflow completion, and verified remediation. Baselines get fair access to the same relevant interaction states as the agent.

This project does not claim that browser automation perfectly reproduces the experience of people using screen readers or other assistive technology.

## Product objective

A tool that:

1. Navigates controlled web forms.
2. Tests specific accessibility-related interaction workflows.
3. Identifies potential accessibility failures.
4. Produces developer-readable evidence and recommendations.
5. Suggests possible fixes.
6. Retests the relevant workflows to check whether repairs resolve the problem.

A fix proposed by an LLM is not automatically a verified fix.

## Current stage: feasibility spike

The project is starting with a small feasibility spike, not the full product. Engineering requirements live in [`SPEC.md`](SPEC.md), which is the authoritative specification. Automation is restricted to locally controlled test sites; the agent must not test arbitrary third-party sites or authenticated production systems.

## Team workflow

Owner defines goals, then ChatGPT reviews research and architecture, then Claude Code implements, then automated verification runs, then ChatGPT and Grok review, then the owner approves the merge.

- **Owner (Arun):** final authority on methodology, scope, major architecture, merges, and deployment.
- **ChatGPT:** research lead, product strategy, architectural review, independent verification.
- **Claude Code:** primary engineer for implementation, tests, and technical docs.
- **Grok:** repository administration, GitHub operations, workflow maintenance, project tracking.

Rules:

- Work happens on feature branches; significant changes arrive as pull requests.
- No merges without explicit owner approval. No automatic deployment of unfinished experiments.
- Implementation, verification, and deployment statuses are tracked separately.
- No expensive or high-frequency CI without approval.
- Never commit API keys, credentials, participant information, or private experimental secrets.

### Changes that require owner/research review

- The agent's system prompt, tools, or permissions (Claude Code presents these for review before implementing the agent loop).
- The test-case generator.
- The definition of a successful accessibility audit.
- The evaluation baseline, ground-truth labels, statistical metrics, or research hypothesis.

### Keep these separate

1. Defects established by controlled test cases (ground truth).
2. Findings reported by the AI agent.
3. Findings from baseline tools.
4. Verified remediation outcomes.

## Reproducibility

The repo should hold what's needed to reproduce experiments: test definitions, model configurations, relevant prompts, software versions, and evaluation methodology. Large generated datasets and verbose raw logs are not committed by default; an artifact storage approach will be set once the experimental design is final.

## Repository layout

```text
README.md   project overview (this file)
SPEC.md     authoritative engineering spec (owner-supplied)
docs/       methodology, architecture decisions, experimental protocols, evaluation docs
```
