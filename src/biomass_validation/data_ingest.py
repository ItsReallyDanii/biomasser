from __future__ import annotations
import pandas as pd
from pathlib import Path

def ingest_hydraulic_curve(path: str | Path) -> pd.DataFrame:
    """
    Reads a measured hydraulic curve CSV.

    Expected columns (case-insensitive):
      - q_m3_s
      - dp_pa   (or dp_pa_per_m)
    Optional:
      - temp_c
      - viscosity_pa_s
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df
