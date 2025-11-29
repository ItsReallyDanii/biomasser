"""
optimize_structures.py
Search the latent space for synthetic xylem designs with high simulated conductivity.
"""

import os, sys, subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# --- dependencies ---
REQUIRED = ["torch", "torchvision", "numpy", "matplotlib", "Pillow"]
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# --- paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "optimization")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- import project modules ---
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from src.model import XylemAutoencoder
    from src.simulate_flow import simulate_pressure_field, compute_conductivity
except ModuleNotFoundError:
    import model
    from simulate_flow import simulate_pressure_field, compute_conductivity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 32
ITERATIONS = 20
POP_SIZE = 8
STEP = 0.05

# --- decode latent vector into image tensor ---
def decode_structure(model, z):
    """
    Converts latent vector z → reconstructed 256x256 image tensor.
    Handles reshaping to match decoder input.
    """
    with torch.no_grad():
        # fully-connected reconstruction to feature map
        flat = model.fc_dec(z)
        # reshape: (batch, channels=128, height=16, width=16)
        feature = flat.view(-1, 128, 16, 16)
        recon = model.decoder_deconv(feature)
        return recon

# --- main routine ---
def main():
    model = XylemAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    dummy = torch.zeros(1, 1, 256, 256).to(DEVICE)
    _ = model(dummy)  # build internal layers
    state_dict = torch.load(os.path.join(ROOT_DIR, "results", "xylem_autoencoder.pt"), map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # initialize latent population
    z = torch.randn(POP_SIZE, LATENT_DIM, device=DEVICE)
    best_score = -1
    best_img = None

    for it in range(ITERATIONS):
        scores = []
        for i in range(POP_SIZE):
            recon = decode_structure(model, z[i:i+1]).cpu().numpy()[0, 0]
            p_field, mask = simulate_pressure_field(recon)
            cond = compute_conductivity(p_field, mask)
            scores.append(cond)

        scores = np.array(scores)
        best_idx = np.argmax(scores)

        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_img = decode_structure(model, z[best_idx:best_idx+1])
            save_image(best_img, os.path.join(RESULTS_DIR, f"best_iter_{it+1}.png"))

        print(f"Iter {it+1}/{ITERATIONS} | Best conductivity: {scores[best_idx]:.5f}")

        # evolve latent vectors toward top performer
        elite = z[best_idx].unsqueeze(0)
        noise = torch.randn_like(z) * STEP
        z = elite + noise

    save_image(best_img, os.path.join(RESULTS_DIR, "optimized_structure.png"))
    print(f"✅ Optimization complete. Best conductivity = {best_score:.5f}")
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
