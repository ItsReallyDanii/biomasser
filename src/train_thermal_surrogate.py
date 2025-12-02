import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# CONFIG
DATA_DIR = "data/generated_microtubes"
METRICS_FILE = "results/thermal_metrics/thermal_metrics.csv"
SAVE_PATH = "results/thermal_surrogate.pth"
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3

# ---------------------------------------------------------
# 1. The Dataset
# ---------------------------------------------------------
class ThermalDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # We want to predict: [Q_total, rho_solid]
        # We normalize them for better training stability
        self.q_mean = self.df['Q_total'].mean()
        self.q_std = self.df['Q_total'].std()
        self.rho_mean = self.df['rho_solid'].mean()
        self.rho_std = self.df['rho_solid'].std()
        
        print(f"Dataset Stats:")
        print(f"   Q_total:   mean={self.q_mean:.4f}, std={self.q_std:.4f}")
        print(f"   rho_solid: mean={self.rho_mean:.4f}, std={self.rho_std:.4f}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.data_dir, img_name)
        
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
            
        # Targets
        q_val = (row['Q_total'] - self.q_mean) / (self.q_std + 1e-6)
        rho_val = (row['rho_solid'] - self.rho_mean) / (self.rho_std + 1e-6)
        
        targets = torch.tensor([q_val, rho_val], dtype=torch.float32)
        return image, targets

# ---------------------------------------------------------
# 2. The Model (Simple CNN)
# ---------------------------------------------------------
class ThermalSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1 channel (grayscale), 256x256
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # 128
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling -> 128 features
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Outputs: [Q_norm, Rho_norm]
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training Thermal Surrogate on {device}...")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = ThermalDataset(DATA_DIR, METRICS_FILE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = ThermalSurrogate().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}")
            
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save metadata for un-normalization later
            meta = {
                'state_dict': model.state_dict(),
                'q_mean': dataset.q_mean,
                'q_std': dataset.q_std,
                'rho_mean': dataset.rho_mean,
                'rho_std': dataset.rho_std
            }
            torch.save(meta, SAVE_PATH)
            
    print(f"✅ Training complete. Best Loss: {best_loss:.4f}")
    print(f"💾 Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()