# Claims Registry (must be reproducible)

Rule: every claim must map to:
(1) the metric definition (Contract)
(2) the script/config that produced it
(3) the output file path
(4) the commit hash

## Canonical metrics
- dp: `Mean_dP/dy`
- q:  `FlowRate` :contentReference[oaicite:17]{index=17}

## Current headline claims (status: UNKNOWN until linked)
- [ ] “3× hydraulic stiffness vs biological baseline”
  - Evidence: TODO (link CSV + plot + script + config)
- [ ] “Pareto-optimal cooling vs baselines”
  - Evidence: TODO (link CSV + plot + script + config)

## Audit gates (must pass before publishing numbers)
- Threshold policy is single-source-of-truth (VOID_THRESHOLD = 0.60) :contentReference[oaicite:18]{index=18}
- Validation harness reports generated under outputs/reports/
- Claim-to-file mapping completed (no orphan numbers)
