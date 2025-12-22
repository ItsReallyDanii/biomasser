from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

@dataclass
class SimResult:
    q_m3_s: float
    dp_pa: float
    k_m2: float | None = None

def run_sim_once(config: Dict[str, Any], q_m3_s: float, out_dir: str | Path) -> SimResult:
    """
    TODO: Wire this into your existing xylem/flow sim entrypoint.

    Must return:
      - q_m3_s
      - dp_pa
    Optionally:
      - k_m2
    """
    raise NotImplementedError("Wire this into your current sim entrypoint.")
