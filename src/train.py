import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.model import XylemAutoencoder

# ======================================================
# 1️⃣ Configuration
# ======================================================

DATA_DIR = "data/generated_microtubes"
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "model.pth")
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================================================
# 2️⃣ Dataset
# ======================================================

class XylemDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        return self.transform(img), 0

# ======================================================
# 3️⃣ Training Loop
# ======================================================

def train():
    dataset = XylemDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training on {len(dataset)} images for {EPOCHS} epochs using {DEVICE}")

    model = XylemAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch, _ in dataloader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # ======================================================
    # 4️⃣ Save Model
    # ======================================================

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Training complete. Results saved to {MODEL_PATH}")


# ======================================================
# 5️⃣ Entry Point
# ======================================================

if __name__ == "__main__":
    train()
