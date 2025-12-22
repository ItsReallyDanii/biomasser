from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from src.simulate_flow import simulate_pressure_field, compute_conductivity
from PIL import Image


@dataclass
class SimResult:
    q_m3_s: float
    dp_pa: float
    k_m2: float | None = None
    dp_proxy: float | None = None


def _load_structure_png(path: Path, size=(256, 256)) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def run_sim_once(config: dict, q_m3_s: float, out_dir: Path) -> SimResult:
    """
    Minimal adapter:
      1) load a representative structure image
      2) compute conductivity proxy under unit inlet/outlet pressure
      3) convert to dp_proxy that scales linearly with q
      4) convert dp_proxy -> dp_pa via calibration factor
    """
    structure_path = Path(config["structure_png"])
    inlet = float(config.get("inlet", 1.0))
    outlet = float(config.get("outlet", 0.0))
    iterations = int(config.get("iterations", 5000))
    eps = float(config.get("eps", 1e-12))

    pa_per_proxy_unit = config.get("pa_per_proxy_unit", None)
    if pa_per_proxy_unit is None:
        raise ValueError("Missing config key: pa_per_proxy_unit (set this from your fitted scale).")
    pa_per_proxy_unit = float(pa_per_proxy_unit)

    structure = _load_structure_png(structure_path, size=(256, 256))
    p_field, mask = simulate_pressure_field(structure, inlet=inlet, outlet=outlet, iterations=iterations)
    cond = float(compute_conductivity(p_field, mask))  # bigger cond => easier flow

    r_proxy = 1.0 / (cond + eps)
    dp_proxy = float(q_m3_s * r_proxy)
    dp_pa = float(dp_proxy * pa_per_proxy_unit)

    return SimResult(q_m3_s=q_m3_s, dp_pa=dp_pa, dp_proxy=dp_proxy)
