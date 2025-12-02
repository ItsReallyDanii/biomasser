import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from src.model import XylemAutoencoder 
from src.train_thermal_surrogate import ThermalSurrogate # Import your new model class

# CONFIG
MODEL_PATH = "results/model_physics_tuned.pth"
THERMAL_SURROGATE_PATH = "results/thermal_surrogate.pth"
OUTPUT_DIR = "results/thermal_design/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Optimization Loop (Thermal Edition)
# ---------------------------------------------------------

def inverse_design_thermal(ae, surrogate, meta, target_q, target_rho, steps=200):
    device = next(ae.parameters()).device
    
    # 1. Initialize Z
    latent_dim = 32
    z = torch.randn(1, latent_dim, device=device, requires_grad=True)
    
    # 2. Optimizer
    optimizer = optim.Adam([z], lr=0.05)
    
    # Weights 
    w_q = 20.0     # Priority 1: Hit Flux Target
    w_rho = 10.0   # Priority 2: Hit Density Target
    
    # Normalization stats from training metadata
    q_mean, q_std = meta['q_mean'], meta['q_std']
    rho_mean, rho_std = meta['rho_mean'], meta['rho_std']
    
    # Normalize Targets (so they match surrogate output space)
    target_q_norm = (target_q - q_mean) / q_std
    target_rho_norm = (target_rho - rho_mean) / rho_std
    
    history = []

    print(f"🎯 Target: Flux={target_q:.4f}, Density={target_rho:.4f}")

    for i in range(steps):
        optimizer.zero_grad()
        
        # Generator
        recon = ae.decode(z)
        
        # Thermal Surrogate Prediction
        # Output: [Q_norm, Rho_norm]
        preds = surrogate(recon)
        pred_q_norm = preds[:, 0]
        pred_rho_norm = preds[:, 1]
        
        # Losses (in normalized space)
        loss_q = (pred_q_norm - target_q_norm) ** 2
        loss_rho = (pred_rho_norm - target_rho_norm) ** 2
        
        total_loss = (w_q * loss_q) + (w_rho * loss_rho) 
        
        total_loss.backward()
        optimizer.step()
        
        # Un-normalize for logging
        current_q = (pred_q_norm.item() * q_std) + q_mean
        current_rho = (pred_rho_norm.item() * rho_std) + rho_mean
        
        if i % 50 == 0:
            print(f"   Step {i:03d}: Flux={current_q:.4f}, Rho={current_rho:.4f} | Loss={total_loss.item():.4f}")
            
        history.append({
            'step': i,
            'flux': current_q,
            'density': current_rho,
            'loss': total_loss.item()
        })

    return z.detach(), recon.detach(), pd.DataFrame(history)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Autoencoder (Geometry)
    print("⏳ Loading Geometry Model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    ae = XylemAutoencoder().to(device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        ae.load_state_dict(checkpoint['state_dict'])
    else:
        ae.load_state_dict(checkpoint)
    ae.eval()
    
    # Load Thermal Surrogate (Physics)
    print("⏳ Loading Thermal Surrogate...")
    surrogate = ThermalSurrogate().to(device)
    surr_checkpoint = torch.load(THERMAL_SURROGATE_PATH, map_location=device, weights_only=False)
    
    # Load weights and metadata
    surrogate.load_state_dict(surr_checkpoint['state_dict'])
    meta = {k: v for k, v in surr_checkpoint.items() if k != 'state_dict'}
    surrogate.eval()
    
    # --- DESIGN SWEEP ---
    # Targets based on your dataset stats:
    # Q range: 0.05 - 0.14
    # Rho range: 0.07 - 0.60
    
    targets = [
        {"name": "MaxCooling_Heavy", "q": 0.14, "r": 0.60}, # Max performance, heavy
        {"name": "Balanced_Sink",    "q": 0.10, "r": 0.35}, # Good cooling, light
        {"name": "Lightweight_Fin",  "q": 0.07, "r": 0.15}, # Minimal material
    ]
    
    for t in targets:
        print(f"\n🚀 Inverse Designing: {t['name']}")
        z_opt, img_opt, hist = inverse_design_thermal(ae, surrogate, meta, t['q'], t['r'])
        
        # Save Structure
        img_np = img_opt.cpu().squeeze().numpy()
        plt.imsave(os.path.join(OUTPUT_DIR, f"{t['name']}_structure.png"), img_np, cmap='inferno')
        
        # Save Convergence Plot
        plt.figure(figsize=(10,4))
        plt.plot(hist['step'], hist['flux'], label='Flux (Q)', color='red')
        plt.plot(hist['step'], hist['density'], label='Density (Rho)', color='grey')
        plt.axhline(y=t['q'], color='red', linestyle='--', alpha=0.5, label='Target Flux')
        plt.axhline(y=t['r'], color='grey', linestyle='--', alpha=0.5, label='Target Rho')
        plt.legend()
        plt.title(f"Thermal Optimization: {t['name']}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{t['name']}_trajectory.png"))
        plt.close()

    print("\n✅ Thermal Design Sweep Complete. Check 'results/thermal_design/'")

if __name__ == "__main__":
    main()