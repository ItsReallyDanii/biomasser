from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from src.flow_metrics_export import load_image, permeability_map, solve_darcy

@dataclass
class SimResult:
    q_m3_s: float
    dp_pa: float
    k_m2: float | None = None
    dp_proxy: float | None = None

def run_sim_once(config: dict, q_m3_s: float, out_dir: Path) -> SimResult:
    structure_png = config.get("structure_png_override") or config.get("structure_png")
    if not structure_png:
        raise KeyError("Missing structure_png (or structure_png_override) in config.")

    mu = float(config.get("mu", 1.0))
    dp0 = float(config.get("delta_p_ref", 1.0))
    scale_out = float(config.get("pa_per_proxy_unit", 1.0))  # leave 1.0; holdout fits scale anyway

    img = load_image(structure_png)     # matches flow_metrics_export (resizes internally)
    k_map = permeability_map(img)

    p, vx, vy = solve_darcy(k_map, delta_p=dp0, mu=mu)

    # These two match flow_metrics_export definitions at delta_p=dp0
    flow_ref = float(np.mean(np.abs(vy)))
    mean_grad_ref = float(np.mean(np.abs(np.gradient(p)[0])))

    # Linear scaling: if you ask for a different q, scale dp metrics by q/flow_ref
    if flow_ref <= 0:
        factor = 1e9
    else:
        factor = float(q_m3_s) / flow_ref

    dp_proxy = mean_grad_ref * factor
    dp_pa = dp_proxy * scale_out
    mean_k = float(np.mean(k_map))

    return SimResult(q_m3_s=float(q_m3_s), dp_pa=float(dp_pa), dp_proxy=float(dp_proxy), k_m2=mean_k)
