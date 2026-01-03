# Biomasser / Xylem Material Compiler (Physics-Informed Inverse Design)

A “material compiler” that **generates porous/graded microstructures** and optimizes them for **transport physics** (flow + heat), with an attached **validation harness** (“Biomasser”) that enforces consistent metrics and audit gates.

## What this repo is
This codebase combines:
- **Design engine**: generator/decoder → differentiable surrogate → latent optimizer → (optional) solver validation
- **Biomasser validation harness**: a contract-driven pipeline that defines *canonical metrics*, a *canonical simulator adapter*, and audit checks for correctness and reproducibility.

## Why Biomasser exists
Inverse design pipelines fail in boring ways (threshold drift, inverted masks, inconsistent definitions across scripts).
Biomasser makes results defensible by locking down:
- canonical metric column names (`Mean_dP/dy`, `FlowRate`) :contentReference[oaicite:0]{index=0}
- canonical simulator mapping (Darcy pipeline + dp proxy scaling rule) :contentReference[oaicite:1]{index=1}
- audit gates to prevent “threshold trap” errors (single source of truth constants) :contentReference[oaicite:2]{index=2}

## Core definitions (Contract)
Biomasser treats these as the **canonical targets**:
- `Mean_dP/dy` = proxy pressure-gradient metric
- `FlowRate` = proxy flow metric :contentReference[oaicite:3]{index=3}

Canonical simulator adapter:
- uses Darcy solver pipeline (load_image → permeability_map → solve_darcy)
- predicts `dp_proxy = mean_grad_ref * (q / flow_ref)` :contentReference[oaicite:4]{index=4}

Dataset schema expectation (example):
- `data/raw/hydraulic_curve.csv`: filename, q_m3_s, dp_pa :contentReference[oaicite:5]{index=5}

## Audit policy (Single Source of Truth)
We treat normalized images as:
- 0.0 = black/solid, 1.0 = white/void
- void if pixel > 0.60, solid otherwise :contentReference[oaicite:6]{index=6}
Conductivities:
- `K_SOLID = 1.0`, `K_VOID = 0.05` :contentReference[oaicite:7]{index=7}

## Quickstart (high-level)
This repo has two “modes”:

### A) Run the validation harness (Biomasser)
Biomasser provides configs + scripts to generate validation reports and summaries (see `configs/` and `src/biomass_validation/`). The original bootstrap steps created a minimal structure with configs + outputs/reports, then iterated on the simulator adapter and validation runs. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

### B) Run design/optimization + benchmarks
The broader pipeline includes scripts for:
- generating structures
- computing flow + thermal metrics
- connectivity/manufacturability analysis
- multiphysics benchmarking

(See `src/` for the canonical scripts in this clone.)

## Outputs
Validation artifacts live under:
- `outputs/reports/` (e.g., holdout summaries, external benchmark reports)

Design/analysis artifacts typically include:
- flow metrics tables
- thermal metrics tables
- connectivity metrics tables
- benchmark comparison plots

## Repo structure (key)
- `src/` : core pipeline scripts (generation, simulation, analysis, optimization)
- `src/biomass_validation/` : validation harness code + contract + runners
- `configs/` : run configs (`baseline_config.yaml`, `validation_config.yaml`) :contentReference[oaicite:10]{index=10}
- `outputs/reports/` : validation summaries / benchmarks

## Reproducibility stance
Numbers do not “count” unless they can be mapped to:
- a script + config
- an output file under `outputs/` or results folder
- the contract + constants policy above

See `AUDIT_PLAN.md` and `src/biomass_validation/CONTRACT.md` for the ground truth. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
