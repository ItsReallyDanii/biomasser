import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import csv
from src.model import XylemAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "results/model_hybrid.pth"
DATA_DIR = "data/generated_microtubes"
SAVE_MODEL_PATH = "results/model_physics_tuned.pth"
LOG_CSV = "results/physics_training_log.csv"

# strong physics influence
LAMBDA_K = 5.0
LAMBDA_POROSITY = 5.0

TARGETS = {
    "Mean_K": 0.52,      # real avg permeability
    "Porosity": 0.73     # real avg porosity
}

# Differentiable proxy for flow physics
def simulate_flow_metrics_torch(img):
    img = img.squeeze(1)  # (B,H,W)
    porosity = torch.mean((img > 0.5).float(), dim=[1,2])
    mean_k = torch.mean(img, dim=[1,2])
    dp_dy = torch.mean(torch.abs(torch.gradient(img, dim=1)[0]), dim=[1,2])
    flow_rate = mean_k * porosity * 0.1
    anisotropy = torch.std(torch.gradient(img, dim=1)[0], dim=[1,2]) / (
        torch.std(torch.gradient(img, dim=2)[0], dim=[1,2]) + 1e-5)
    return mean_k, porosity, flow_rate, anisotropy, dp_dy

def compute_physics_loss(model, imgs, criterion):
    imgs = imgs.to(DEVICE)
    recon, _ = model(imgs)
    loss_recon = criterion(recon, imgs)

    mean_k, porosity, _, _, _ = simulate_flow_metrics_torch(recon)

    # physics penalty
    loss_phys = (
        LAMBDA_K * torch.mean((mean_k - TARGETS["Mean_K"])**2) +
        LAMBDA_POROSITY * torch.mean((porosity - TARGETS["Porosity"])**2)
    )

    total_loss = loss_recon + loss_phys
    return total_loss, loss_recon.item(), loss_phys.item(), mean_k.mean().item(), porosity.mean().item()

def train_physics_informed():
    print(f"🌱 Physics-informed fine-tuning started on {DEVICE}")

    model = XylemAutoencoder().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        print(f"✅ Loaded pretrained model from {MODEL_PATH}")
    else:
        print("⚠️ No pretrained model found — training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    imgs = []
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img = transform(Image.open(os.path.join(DATA_DIR, f)))
            imgs.append(img)
    imgs = torch.stack(imgs)
    print(f"🧩 Loaded {len(imgs)} generated structures.")

    num_epochs = 10
    if os.path.exists(LOG_CSV):
        os.remove(LOG_CSV)
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "recon_loss", "phys_loss", "mean_k", "porosity"])

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        total_loss, recon_loss, phys_loss, mean_k, porosity = compute_physics_loss(model, imgs, criterion)
        total_loss.backward()
        optimizer.step()

        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, total_loss.item(), recon_loss, phys_loss, mean_k, porosity])

        print(f"Epoch {epoch}/{num_epochs} | "
              f"Total: {total_loss.item():.5f} | Recon: {recon_loss:.5f} | Phys: {phys_loss:.5f} | "
              f"K: {mean_k:.5f} | Porosity: {porosity:.5f}")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"✅ Physics-informed fine-tuning complete.")
    print(f"💾 Model saved → {SAVE_MODEL_PATH}")
    print(f"🧾 Training log saved → {LOG_CSV}")

if __name__ == "__main__":
    train_physics_informed()
