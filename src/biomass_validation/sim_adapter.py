from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import os
import numpy as np

# Reuse your existing Darcy solver + helpers
from src.flow_metrics_export import load_image, permeability_map, solve_darcy

@dataclass
class SimResult:
    q_m3_s: float
    dp_pa: float
    k_m2: float | None = None

def _pick_structure_png(config: Dict[str, Any]) -> str:
    """
    Priority:
      1) config["geometry"]["structure_png"]
      2) first PNG found in data/generated_microtubes
    """
    geo = config.get("geometry", {}) if isinstance(config.get("geometry", {}), dict) else {}
    explicit = geo.get("structure_png")
    if explicit:
        return explicit

    # default folder used elsewhere in repo
    folder = Path("data/generated_microtubes")
    if not folder.exists():
        raise FileNotFoundError("No structure_png set and data/generated_microtubes not found.")

    pngs = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".png"])
    if not pngs:
        raise FileNotFoundError("No PNGs found in data/generated_microtubes.")
    return str(pngs[0])

def run_sim_once(config: Dict[str, Any], q_m3_s: float, out_dir: str | Path) -> SimResult:
    """
    Maps target flowrate -> required delta_p using your Darcy solver:
      - compute q_ref at delta_p=1
      - delta_p_needed = q_target / q_ref
      - rerun solver at delta_p_needed
    Notes:
      - dp_pa here is a *proxy* (scaled delta_p). To get real Pa, you'd calibrate to real geometry/units.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu = float(config.get("sim", {}).get("fluid_viscosity_pa_s", 1.0))

    png_path = _pick_structure_png(config)
    img = load_image(png_path)               # 0..1, resized inside flow_metrics_export
    k_map = permeability_map(img)

    # 1) Reference run: delta_p = 1
    p1, vx1, vy1 = solve_darcy(k_map, delta_p=1.0, mu=mu)
    q_ref = float(np.mean(np.abs(vy1)))

    if q_ref <= 0:
        raise ValueError("Reference flowrate q_ref is <= 0; structure may be blocked or solver unstable.")

    # 2) Scale delta_p to hit the target flowrate
    delta_p_needed = float(q_m3_s / q_ref)

    p2, vx2, vy2 = solve_darcy(k_map, delta_p=delta_p_needed, mu=mu)

    # dp proxy: using delta_p directly (unitless until calibrated)
    dp_pa = delta_p_needed

    mean_k = float(np.mean(k_map))  # proxy for permeability
    return SimResult(q_m3_s=float(q_m3_s), dp_pa=float(dp_pa), k_m2=mean_k)
