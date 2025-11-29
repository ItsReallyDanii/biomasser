import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

RESULTS_DIR = "results/flow_simulation"
os.makedirs(RESULTS_DIR, exist_ok=True)

REAL_DIR = "data/real_xylem_preprocessed"
SYN_DIR = "data/generated_microtubes"
METRICS_PATH = "results/flow_metrics/flow_metrics.csv"
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

def load_images(path):
    imgs = []
    for f in sorted(os.listdir(path)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img = np.array(Image.open(os.path.join(path, f)).convert("L")) / 255.0
            imgs.append(img)
    return np.array(imgs)

def compute_flow_metrics(img):
    # Basic permeability proxy (Darcy-like)
    porosity = np.mean(img)
    grad_y, grad_x = np.gradient(img)
    dp_dy = np.mean(np.abs(grad_y))
    anisotropy = np.mean(np.abs(grad_x)) / (dp_dy + 1e-8)
    mean_k = porosity * (1.0 - dp_dy)
    flow_rate = mean_k * porosity
    return {
        "Mean_K": mean_k,
        "Mean_dP/dy": dp_dy,
        "FlowRate": flow_rate,
        "Porosity": porosity,
        "Anisotropy": anisotropy,
    }

def simulate_batch(img_dir, label):
    imgs = load_images(img_dir)
    n = len(imgs)
    if n == 0:
        print(f"⚠️ No images found in {img_dir}")
        return []

    print(f"\n🌿 Simulating flow for {label} structures...")
    all_metrics = []
    efficiencies = []

    for i, img in enumerate(imgs):
        metrics = compute_flow_metrics(img)
        all_metrics.append(metrics)
        efficiencies.append(metrics["FlowRate"])

    mean_efficiency = np.mean(efficiencies)
    print(f"🌊 Mean relative flow efficiency ({label}): {mean_efficiency:.4f}")
    return all_metrics, mean_efficiency

def main():
    real_metrics, real_eff = simulate_batch(REAL_DIR, "Real Xylem")
    syn_metrics, syn_eff = simulate_batch(SYN_DIR, "Synthetic Xylem")

    ratio = syn_eff / (real_eff + 1e-8)
    print(f"\n🧩 Synthetic vs Real Flow Ratio: {ratio:.2f}")

    import pandas as pd
    df_real = pd.DataFrame(real_metrics)
    df_real["Type"] = "Real"
    df_syn = pd.DataFrame(syn_metrics)
    df_syn["Type"] = "Synthetic"
    df = pd.concat([df_real, df_syn], ignore_index=True)
    df.to_csv(METRICS_PATH, index=False)

    print(f"✅ Flow simulation complete. Results in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
