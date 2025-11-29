"""
flow_simulation.py
-------------------------------------
Simulates fluid transport through real and synthetic xylem morphologies
using a simplified Darcy / permeability model.

Each image acts as a porous domain, where pixel intensity ~ permeability.
Computes relative flow efficiency, pressure maps, and streamlines.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, linalg

# -------------------------------
# CONFIGURATION
# -------------------------------
REAL_DIR = "data/real_xylem_preprocessed"
SYN_DIR = "data/generated_microtubes"
SAVE_DIR = "results/flow_simulation"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
VISCOSITY = 1.0
DELTA_P = 1.0  # unit pressure drop from top to bottom
EPSILON = 1e-6  # avoid division by zero


# -------------------------------
# UTILITIES
# -------------------------------
def load_grayscale_images(path, limit=5):
    files = sorted([f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    imgs = []
    for f in files[:limit]:
        img = Image.open(os.path.join(path, f)).convert("L").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        imgs.append(arr)
    return imgs


def compute_permeability_map(img):
    """Convert grayscale structure into permeability map (Darcy term)."""
    # Brighter = more porous (higher k)
    k = img**3 + EPSILON
    return gaussian_filter(k, sigma=1)


def solve_darcy_flow(k_map, delta_p=DELTA_P, mu=VISCOSITY):
    """
    Solve simplified Darcy’s law in 2D steady-state:
        div(k * grad(p)) = 0
    Approximation using finite differences and linear solver.
    """
    ny, nx = k_map.shape
    N = nx * ny

    # Compute interface conductances
    kx = (k_map[:, 1:] + k_map[:, :-1]) / 2     # shape (ny, nx-1)
    ky = (k_map[1:, :] + k_map[:-1, :]) / 2     # shape (ny-1, nx)

    main = np.zeros(N)
    east = np.zeros(N - 1)
    west = np.zeros(N - 1)
    north = np.zeros(N - nx)
    south = np.zeros(N - nx)

    for y in range(ny):
        for x in range(nx):
            i = y * nx + x

            # Safely index within valid ranges
            k_e = kx[y, x] if x < nx - 1 else 0
            k_w = kx[y, x - 1] if x > 0 else 0
            k_n = ky[y - 1, x] if y > 0 else 0
            k_s = ky[y, x] if y < ny - 1 else 0

            main[i] = -(k_e + k_w + k_n + k_s)
            if x > 0:
                east[i - 1] = k_e
            if x < nx - 1:
                west[i] = k_w
            if y > 0:
                north[i - nx] = k_n
            if y < ny - 1:
                south[i] = k_s

    A = diags([main, east, west, north, south], [0, -1, 1, -nx, nx], format="csr")

    # Boundary conditions: top (p=ΔP), bottom (p=0)
    b = np.zeros(N)
    for x in range(nx):
        b[x] = delta_p * k_map[0, x]
        b[-nx + x] = 0.0

    # Solve linear system
    p = linalg.spsolve(A, b)
    p_field = p.reshape(ny, nx)

    # Compute flux = -k * grad(p)
    grad_y, grad_x = np.gradient(p_field)
    vx = -k_map * grad_x / mu
    vy = -k_map * grad_y / mu
    return p_field, vx, vy


def visualize_flow(img, p_field, vx, vy, title, save_path):
    """Overlay flow field arrows on morphology."""
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray", origin="lower")
    plt.quiver(vx[::4, ::4], vy[::4, ::4], color="cyan", scale=30)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"💧 Saved flow visualization: {save_path}")


# -------------------------------
# MAIN
# -------------------------------
def simulate_batch(image_dir, label):
    print(f"\n🌿 Simulating flow for {label} structures...")
    imgs = load_grayscale_images(image_dir, limit=5)
    mean_flux = []

    for i, img in enumerate(imgs):
        k_map = compute_permeability_map(img)
        p_field, vx, vy = solve_darcy_flow(k_map)
        flux = np.mean(np.abs(vy))  # proxy for throughput
        mean_flux.append(flux)
        out_path = os.path.join(SAVE_DIR, f"flow_{label}_{i+1}.png")
        visualize_flow(img, p_field, vx, vy, f"{label} Flow {i+1}", out_path)

    avg = np.mean(mean_flux)
    print(f"🌊 Mean relative flow efficiency ({label}): {avg:.4f}")
    return avg


if __name__ == "__main__":
    flow_real = simulate_batch(REAL_DIR, "Real Xylem")
    flow_syn = simulate_batch(SYN_DIR, "Synthetic Xylem")

    ratio = flow_syn / flow_real if flow_real > 0 else 0
    print(f"\n🧩 Synthetic vs Real Flow Ratio: {ratio:.2f}")
    print(f"✅ Flow simulation complete. Results in {SAVE_DIR}")