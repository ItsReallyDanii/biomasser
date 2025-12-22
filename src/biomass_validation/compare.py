from __future__ import annotations
import numpy as np
import pandas as pd

def dp_error_percent(df_meas: pd.DataFrame, df_sim: pd.DataFrame) -> float:
    """
    Both dataframes must have columns:
      - q_m3_s
      - dp_pa

    Returns mean absolute percent error on dp at matched q points.
    """
    m = df_meas.set_index("q_m3_s")["dp_pa"]
    s = df_sim.set_index("q_m3_s")["dp_pa"]
    common = m.index.intersection(s.index)
    if len(common) == 0:
        raise ValueError("No matching q values between measured and simulated data.")
    mpe = (np.abs((s.loc[common] - m.loc[common]) / m.loc[common]).mean()) * 100.0
    return float(mpe)
