from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

from src.simulate_flow import simulate_pressure_field, compute_conductivity


@dataclass
class SimResult:
    q_m3_s: float
    dp_pa: float
    k_m2: float | None = None
    dp_proxy: float | None = None


def _load_structure_png(path: Path, size=(256, 256), invert: bool = False) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    return arr


def run_sim_once(config: dict, q_m3_s: float, out_dir: Path) -> SimResult:
    structure_png = config.get("structure_png_override") or config.get("structure_png")
    if not structure_png:
        raise KeyError("Missing structure_png (or structure_png_override) in config.")

    structure_path = Path(structure_png)
    inlet = float(config.get("inlet", 1.0))
    outlet = float(config.get("outlet", 0.0))
    iterations = int(config.get("iterations", 5000))
    eps = float(config.get("eps", 1e-12))
    invert = bool(config.get("invert_structure", False))

    pa_per_proxy_unit = float(config.get("pa_per_proxy_unit", 1.0))

    structure = _load_structure_png(structure_path, size=(256, 256), invert=invert)

    p_field, mask = simulate_pressure_field(structure, inlet=inlet, outlet=outlet, iterations=iterations)
    cond = float(compute_conductivity(p_field, mask))

    # Guard: if cond ~0, you're basically blocked -> dp becomes eps-limited nonsense
    if cond < 1e-10:
        # Return a huge dp_proxy to reflect near-blockage, but not eps-dominated blow-up
        dp_proxy = float(q_m3_s) * 1e6
    else:
        r_proxy = 1.0 / (cond + eps)
        dp_proxy = float(q_m3_s * r_proxy)

    dp_pa = float(dp_proxy * pa_per_proxy_unit)
    return SimResult(q_m3_s=q_m3_s, dp_pa=dp_pa, dp_proxy=dp_proxy)
