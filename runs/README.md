# runs

Per-run JSONL records (gitignored) and the committed `summary.csv` with per-cell rates and Wilson 95% intervals. See SPEC.md, Logging.

- `spike-v2.jsonl`: the feasibility spike, kept untouched as pressure level 0.
- `calibration.jsonl`: exploratory calibration runs at pressure levels 1-4, summarized per level in `calibration_summary.csv`.
- `study1.jsonl`: Study 1 (SPEC v3), written only by real runs of `configs/study1.yaml`; summarized in `study1_summary.csv`.
