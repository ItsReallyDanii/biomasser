# Biomass Validation Engine — Contract

## Canonical metric definitions
- dp target column: `Mean_dP/dy` (proxy gradient metric)
- q target column: `FlowRate` (proxy flow metric)

## Canonical simulator
- Uses Darcy solver pipeline from `src/flow_metrics_export.py`:
  - load_image -> permeability_map -> solve_darcy
- Predicts dp_proxy as:
  - mean_grad_ref * (q / flow_ref)

## Dataset schema
data/raw/hydraulic_curve.csv columns:
- filename (png name)
- q_m3_s (proxy flow)
- dp_pa  (proxy mean gradient)
