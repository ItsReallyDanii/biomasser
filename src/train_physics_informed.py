import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from src.model import XylemAutoencoder
from src.flow_simulation_utils import compute_flow_metrics  # we’ll add this helper next

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/generated_microtubes"
MODEL_PATH = "results/model_hybrid.pth"
SAVE_PATH = "results/model_physics_informed.pth"

# ------------
# Hyperparams
# ------------
EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 4
LAMBDA_PHYSICS = 0.1  # relative weight for physics loss
IMG_SIZE = 256

# ------------
# Data loader
# ------------
def load_images(path):
    files = [f for f in os.listdir(path) if f.endswith(".png")]
    imgs = []
    for f in files:
        img = Image.open(os.path.join(path, f)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_t = transforms.ToTensor()(img)
        imgs.append(img_t)
    return torch.stack(imgs)

# ------------
# Physics loss
# ------------
def physics_loss(img_tensor):
    """Run mini flow sim on decoded structure and return mismatch from target physics."""
    img_np = img_tensor.detach().cpu().numpy()[0, 0]
    metrics = compute_flow_metrics(img_np)

    # Reference "realistic" targets from your dataset’s mean
    target_flow = 0.26  # mean from real_xylem
    target_porosity = 0.72

    flow_diff = (metrics["FlowRate"] - target_flow) ** 2
    porosity_diff = (metrics["Porosity"] - target_porosity) ** 2

    return torch.tensor(flow_diff + porosity_diff, device=DEVICE, dtype=torch.float32)

# ------------
# Main
# ------------
def main():
    model = XylemAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    imgs = load_images(DATA_DIR)
    print(f"🧬 Loaded {len(imgs)} training images for physics-informed fine-tuning.")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i in range(0, len(imgs), BATCH_SIZE):
            batch = imgs[i:i+BATCH_SIZE].to(DEVICE)
            recon, _ = model(batch)
            recon_loss = criterion(recon, batch)

            # physics term on first image in batch
            phys_loss = physics_loss(recon[:1])
            loss = recon_loss + LAMBDA_PHYSICS * phys_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"🌿 Epoch {epoch+1}/{EPOCHS} | Total Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Saved physics-informed model → {SAVE_PATH}")

if __name__ == "__main__":
    main()
