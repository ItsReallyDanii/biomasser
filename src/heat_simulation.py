"""
heat_simulation.py — 2D steady-state heat solver for "thermal sponge" designs.

Given binary-ish microstructure images (solid vs void), we:
  - Map them to a conductivity field k(x,y)
  - Solve ∇·(k ∇T) = 0 with simple Dirichlet BCs:
      * Left boundary: hot chip  (T = T_HOT)
      * Right boundary: cold sink (T = T_COLD)
      * Top/bottom: insulated (Neumann ≈ no-flux)
  - Compute:
      * T_max_chip: max temperature in a hot-strip near the left side
      * Q_total: total heat flux leaving at the cold boundary
      * rho_solid: solid volume fraction (material cost)
  - Save metrics for all images to results/thermal_metrics/thermal_metrics.csv
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

# ------------------------
# Config
# ------------------------
DATA_DIR = "data/generated_microtubes"
OUTPUT_DIR = "results/thermal_metrics"  # Ensure directory exists
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "thermal_metrics.csv")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif")

# Image → material mapping
# Adjusted threshold for Robust Normalized images:
# After normalization, 0.5 is the median gray. 
# Anything brighter than 0.6 is VOID (Water/Air). Darker is SOLID (Metal).
VOID_THRESHOLD = 0.60   
K_SOLID = 1.0           # relative conductivity of solid (Metal)
K_VOID = 0.05           # relative conductivity of void (Water/Air)

# Thermal BCs
T_HOT = 1.0
T_COLD = 0.0
HOT_STRIP_WIDTH = 5      # how many columns from the left we treat as "chip" region for T_max

# Solver
MAX_ITERS = 4000
TOL = 1e-4              # residual stopping criterion
PRINT_EVERY = 500       # how often to print residual


# ------------------------
# Core solver
# ------------------------
def solve_steady_heat(k_grid, t_hot=T_HOT, t_cold=T_COLD,
                      max_iters=MAX_ITERS, tol=TOL, print_every=PRINT_EVERY):
    """
    Solve ∇·(k ∇T) = 0 on a 2D grid with:
      - Left boundary: T = t_hot
      - Right boundary: T = t_cold
      - Top/bottom: insulated (approx via copying interior values)

    k_grid: (H, W) conductivity field.
    Returns: T (H, W), steady-state temperature field.
    """
    H, W = k_grid.shape
    T = np.zeros((H, W), dtype=np.float32)
    
    # Linear Initialization (Speeds up convergence vs Zeros)
    x_coords = np.linspace(t_hot, t_cold, W)
    T[:] = x_coords  
    
    T[:, 0] = t_hot
    T[:, -1] = t_cold

    # Convenience for boundary masks
    left_idx = 0
    right_idx = W - 1

    for it in range(max_iters):
        max_delta = 0.0

        # Neumann on top/bottom: copy neighbors (simple approximation)
        # This keeps dT/dn ≈ 0 at boundaries
        T[0, :] = T[1, :]
        T[-1, :] = T[-2, :]

        # Gauss–Seidel update on interior (excluding left/right Dirichlet cols)
        # Vectorized implementation (much faster than loops)
        # Note: This is a Jacobi update (parallel), but works for convergence.
        # T_new = (k_n*T_up + k_s*T_down + k_w*T_left + k_e*T_right) / sum_k
        
        # Shifted arrays for neighbors
        T_up    = T[0:-2, 1:-1]
        T_down  = T[2:,   1:-1]
        T_left  = T[1:-1, 0:-2]
        T_right = T[1:-1, 2:]
        
        # Current K
        k_c = k_grid[1:-1, 1:-1]
        
        # Neighbor K (Harmonic Mean for interface conductivity)
        # Harmonic mean handles the sharp jump from k=1 to k=0.05 better than arithmetic
        eps = 1e-8
        k_n = 2 * k_c * k_grid[0:-2, 1:-1] / (k_c + k_grid[0:-2, 1:-1] + eps)
        k_s = 2 * k_c * k_grid[2:,   1:-1] / (k_c + k_grid[2:,   1:-1] + eps)
        k_w = 2 * k_c * k_grid[1:-1, 0:-2] / (k_c + k_grid[1:-1, 0:-2] + eps)
        k_e = 2 * k_c * k_grid[1:-1, 2:]   / (k_c + k_grid[1:-1, 2:]   + eps)
        
        denom = k_n + k_s + k_w + k_e + eps
        
        T_new_interior = (k_n * T_up + k_s * T_down + k_w * T_left + k_e * T_right) / denom
        
        # Calculate Delta
        delta = np.abs(T_new_interior - T[1:-1, 1:-1])
        current_max_delta = np.max(delta)
        
        # Update T
        T[1:-1, 1:-1] = T_new_interior
        
        if (it + 1) % print_every == 0:
            print(f"   [heat] iter {it+1:4d}/{max_iters}, max ΔT = {current_max_delta:.3e}")
        
        if current_max_delta < tol:
            break

    return T


def compute_metrics(img_arr, T, k_grid):
    """
    img_arr: (H, W) in [0,1]
    T:       (H, W) temperature field
    k_grid:  (H, W) conductivity field

    Returns:
      T_max_chip, Q_total, rho_solid
    """
    H, W = img_arr.shape

    # Solid vs void (consistent with VOID_THRESHOLD convention)
    # Bright pixels (> Threshold) are Void. Dark pixels are Solid.
    solid_mask = img_arr <= VOID_THRESHOLD
    rho_solid = solid_mask.mean()

    # Chip / hot strip region (left HOT_STRIP_WIDTH columns)
    chip_cols = min(HOT_STRIP_WIDTH, W)
    chip_region = T[:, :chip_cols]
    T_max_chip = float(chip_region.max())

    # Heat flux through cold boundary (right boundary)
    # Approximate q = -k * dT/dx, so:
    #   dT/dx ≈ T[:, -1] - T[:, -2]
    #   Note: T[:,-1] is 0.0 (Cold Sink). T[:,-2] is > 0.
    #   So dT/dx is negative. Flux should be positive leaving the domain?
    #   Fourier Law: q = -k dT/dx.
    #   Flux vector points Right.
    dTdx = T[:, -1] - T[:, -2] # (0 - Positive) = Negative
    q_boundary = -k_grid[:, -1] * dTdx # (-k * Negative) = Positive Flux
    Q_total = float(q_boundary.sum())

    return T_max_chip, Q_total, float(rho_solid)


# ------------------------
# Main loop over images
# ------------------------
def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Input directory not found: {DATA_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.lower().endswith(IMG_EXTS))

    if not files:
        raise RuntimeError(f"No images found in {DATA_DIR} with extensions {IMG_EXTS}")

    print(f"🌡️  Running thermal simulation on {len(files)} structures...")
    print(f"   Data dir: {DATA_DIR}")
    print(f"   Output:   {OUTPUT_CSV}")
    print(f"   Void threshold: {VOID_THRESHOLD}, K_solid={K_SOLID}, K_void={K_VOID}")

    records = []

    for idx, fname in enumerate(files, start=1):
        path = os.path.join(DATA_DIR, fname)
        img = Image.open(path).convert("L")
        arr_raw = np.array(img, dtype=np.float32)
        
        # --- ROBUST NORMALIZATION FIX ---
        # Stretch contrast to full 0.0-1.0 range so gray images become B/W
        img_min, img_max = arr_raw.min(), arr_raw.max()
        if img_max > img_min:
            arr_norm = (arr_raw - img_min) / (img_max - img_min)
        else:
            arr_norm = arr_raw / 255.0 # Fallback for blank images
            
        # Map to conductivity field
        # White (>= threshold) = void (low k), dark = solid (high k)
        void_mask = arr_norm > VOID_THRESHOLD
        solid_mask = ~void_mask
        k_grid = np.where(solid_mask, K_SOLID, K_VOID).astype(np.float32)

        print(f"\n[{idx}/{len(files)}] {fname}")
        T = solve_steady_heat(k_grid)

        T_max_chip, Q_total, rho_solid = compute_metrics(arr_norm, T, k_grid)

        print(f"   ➜ T_max_chip = {T_max_chip:.4f}, "
              f"Q_total = {Q_total:.4e}, "
              f"rho_solid = {rho_solid:.3f}")

        records.append({
            "filename": fname,
            "T_max_chip": T_max_chip,
            "Q_total": Q_total,
            "rho_solid": rho_solid,
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Thermal metrics saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()