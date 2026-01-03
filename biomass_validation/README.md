# Biomasser Validation Harness

Biomasser is the contract + audit layer that turns the inverse-design pipeline into a defensible research artifact.

## Contract (canonical metrics)
Target columns are:
- `Mean_dP/dy` (proxy gradient)
- `FlowRate` (proxy flow) :contentReference[oaicite:13]{index=13}

## Canonical simulator adapter
Uses Darcy solver pipeline (load_image → permeability_map → solve_darcy) and predicts
`dp_proxy = mean_grad_ref * (q / flow_ref)` :contentReference[oaicite:14]{index=14}

## Why this exists
We explicitly guard against “threshold trap” / inverted mask errors by enforcing a single constants policy across modules. :contentReference[oaicite:15]{index=15}

## Configs
The repo uses:
- `configs/baseline_config.yaml`
- `configs/validation_config.yaml` :contentReference[oaicite:16]{index=16}

## Reports
Outputs are written under `outputs/reports/`.
