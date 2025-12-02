import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# CONFIG
OUTPUT_DIR = "results/baselines/"
AI_METRICS_PATH = "results/thermal_metrics/thermal_metrics.csv" # Your AI results
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PHYSICS CONSTANTS (Must match heat_simulation.py)
VOID_THRESHOLD = 0.60
K_SOLID = 1.0
K_VOID = 0.05
T_HOT = 1.0
T_COLD = 0.0
HOT_STRIP_WIDTH = 5

# ---------------------------------------------------------
# 1. Baseline Generators
# ---------------------------------------------------------

def generate_vertical_fins(shape=(256, 256), num_fins=10, thickness=5):
    """Straight vertical lines (Standard Heat Sink)"""
    img = np.ones(shape) # Start with Void (White=1.0)
    spacing = shape[1] // num_fins
    for i in range(num_fins):
        x = i * spacing + (spacing // 2)
        img[:, x:x+thickness] = 0.0 # Solid (Black=0.0)
    return img

def generate_grid(shape=(256, 256), num_cells=8, thickness=2):
    """Cross-hatch grid"""
    img = np.ones(shape)
    step = shape[0] // num_cells
    # Vertical lines
    for x in range(0, shape[1], step):
        img[:, x:x+thickness] = 0.0
    # Horizontal lines
    for y in range(0, shape[0], step):
        img[y:y+thickness, :] = 0.0
    return img

def generate_random_noise(shape=(256, 256), density=0.4):
    """Random porous media"""
    return np.random.choice([0.0, 1.0], size=shape, p=[density, 1-density])

# ---------------------------------------------------------
# 2. The Solver (Copied for consistency)
# ---------------------------------------------------------
def solve_steady_heat(k_grid, max_iters=2000):
    H, W = k_grid.shape
    T = np.linspace(1.0, 0.0, W); T = np.tile(T, (H, 1)) # Linear init
    T[:, 0] = T_HOT; T[:, -1] = T_COLD

    for _ in range(max_iters):
        # Fast Vectorized Gauss-Seidel Approximation
        T_up, T_down = T[0:-2, 1:-1], T[2:, 1:-1]
        T_left, T_right = T[1:-1, 0:-2], T[1:-1, 2:]
        k_c = k_grid[1:-1, 1:-1]
        
        # Neighbor K (Harmonic Mean)
        eps = 1e-8
        k_n = 2*k_c*k_grid[0:-2, 1:-1]/(k_c+k_grid[0:-2, 1:-1]+eps)
        k_s = 2*k_c*k_grid[2:, 1:-1]/(k_c+k_grid[2:, 1:-1]+eps)
        k_w = 2*k_c*k_grid[1:-1, 0:-2]/(k_c+k_grid[1:-1, 0:-2]+eps)
        k_e = 2*k_c*k_grid[1:-1, 2:]/(k_c+k_grid[1:-1, 2:]+eps)
        
        T[1:-1, 1:-1] = (k_n*T_up + k_s*T_down + k_w*T_left + k_e*T_right) / (k_n+k_s+k_w+k_e+eps)
        
        # BCs
        T[:, 0] = T_HOT; T[:, -1] = T_COLD
        T[0, :] = T[1, :]; T[-1, :] = T[-2, :] # Insulated Top/Bottom
        
    return T

def evaluate_design(img, name):
    # Map to K
    # Img: 1.0=Void, 0.0=Solid
    k_map = np.where(img < 0.5, K_SOLID, K_VOID)
    
    # Solve
    T = solve_steady_heat(k_map)
    
    # Metrics
    dTdx = T[:, -1] - T[:, -2]
    flux = np.sum(-k_map[:, -1] * dTdx)
    density = np.mean(img < 0.5)
    
    return {"name": name, "flux": flux, "density": density}

# ---------------------------------------------------------
# 3. Execution & Plotting
# ---------------------------------------------------------
def main():
    print("🧪 Generative AI vs. Engineering Baselines...")
    results = []
    
    # 1. Run Baselines
    # Generate a spectrum of fins (sparse to dense)
    for n in [5, 10, 20, 40, 60]: 
        img = generate_vertical_fins(num_fins=n, thickness=4)
        results.append(evaluate_design(img, f"Fins_{n}"))
        
    # Generate a spectrum of grids
    for n in [4, 8, 16, 32]:
        img = generate_grid(num_cells=n, thickness=2)
        results.append(evaluate_design(img, f"Grid_{n}"))
        
    # Generate random noise (The "Bad" Baseline)
    for d in [0.2, 0.4, 0.6]:
        img = generate_random_noise(density=d)
        results.append(evaluate_design(img, f"Random_{d}"))

    base_df = pd.DataFrame(results)
    base_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_metrics.csv"), index=False)
    print("✅ Baselines Simulated.")

    # 2. Load AI Results
    if os.path.exists(AI_METRICS_PATH):
        ai_df = pd.read_csv(AI_METRICS_PATH)
        # Normalize names for plotting
        ai_flux = ai_df['Q_total']
        ai_dens = ai_df['rho_solid']
    else:
        print("⚠️  Warning: AI metrics not found. Plotting baselines only.")
        ai_df = None

    # 3. The Showdown Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Baselines
    plt.scatter(base_df['density'], base_df['flux'], c='gray', marker='x', s=100, label='Standard Engineering (Fins/Grids)')
    
    # Plot AI
    if ai_df is not None:
        plt.scatter(ai_dens, ai_flux, c=ai_flux, cmap='inferno', alpha=0.6, label='AI Generative Design')
        plt.colorbar(label='Heat Flux')
    
    plt.title("Performance Frontier: Generative Coral vs. Standard Fins", fontsize=14)
    plt.xlabel("Material Density (Cost/Weight)", fontsize=12)
    plt.ylabel("Heat Flux (Cooling Performance)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = os.path.join(OUTPUT_DIR, "performance_frontier.png")
    plt.savefig(save_path, dpi=150)
    print(f"📊 Comparison Plot saved to {save_path}")

if __name__ == "__main__":
    main()