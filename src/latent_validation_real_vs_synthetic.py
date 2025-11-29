"""
latent_validation_real_vs_synthetic.py
Compare latent space distributions between real and synthetic xylem structures.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
from torchvision import transforms
from src.model import XylemAutoencoder

# -------------------------------
# Configuration
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results/model.pth"
REAL_DIR = "data/real_xylem_preprocessed"
SYN_DIR = "data/generated_microtubes"
SAVE_PATH = "results/latent_validation"
os.makedirs(SAVE_PATH, exist_ok=True)

# -------------------------------
# Load model
# -------------------------------
print("⚙️ Loading model...")
model = XylemAutoencoder(latent_dim=32).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Partial load with mismatched keys skipped
model_dict = model.state_dict()
compatible_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(compatible_dict)
model.load_state_dict(model_dict)
print(f"✅ Loaded {len(compatible_dict)} compatible layers.")

# -------------------------------
# Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_images(path, n=20):
    imgs = []
    valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    for f in sorted(os.listdir(path)):
        full_path = os.path.join(path, f)
        if not os.path.isfile(full_path) or not f.lower().endswith(valid_ext):
            continue
        img = Image.open(full_path).convert("L")
        imgs.append(transform(img).unsqueeze(0))
        if len(imgs) >= n:
            break
    return torch.cat(imgs, dim=0)

# -------------------------------
# Load datasets
# -------------------------------
print("📥 Loading datasets...")
real_imgs = load_images(REAL_DIR)
syn_imgs = load_images(SYN_DIR)
real_imgs, syn_imgs = real_imgs.to(DEVICE), syn_imgs.to(DEVICE)

# -------------------------------
# Encode latent space
# -------------------------------
print("🔬 Encoding latent representations...")
with torch.no_grad():
    z_real, _ = model(real_imgs)
    z_syn, _ = model(syn_imgs)

# Flatten latent tensors for PCA
z_real = z_real.view(z_real.size(0), -1).cpu().numpy()
z_syn = z_syn.view(z_syn.size(0), -1).cpu().numpy()

# -------------------------------
# Dimensionality reduction
# -------------------------------
print("📉 Projecting to 2D space (PCA + t-SNE)...")
all_z = np.concatenate([z_real, z_syn], axis=0)
labels = np.array(["Real"] * len(z_real) + ["Synthetic"] * len(z_syn))

pca = PCA(n_components=min(20, all_z.shape[1])).fit_transform(all_z)

tsne = TSNE(
    n_components=2,
    perplexity=5,
    max_iter=2000,
    learning_rate="auto",
    init="pca",
    random_state=42
).fit_transform(pca)

# -------------------------------
# Plot latent overlap
# -------------------------------
print("🧩 Plotting latent overlap map...")
plt.figure(figsize=(8, 6))
plt.scatter(tsne[labels == "Real", 0], tsne[labels == "Real", 1],
            alpha=0.6, label="Real Xylem", s=50)
plt.scatter(tsne[labels == "Synthetic", 0], tsne[labels == "Synthetic", 1],
            alpha=0.6, label="Synthetic Xylem", s=50)
plt.legend()
plt.title("Latent Space Overlap: Real vs Synthetic Xylem Structures")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")

save_file = os.path.join(SAVE_PATH, "latent_overlap.png")
plt.savefig(save_file, dpi=300)
plt.close()

print(f"✅ Latent validation complete! Saved visualization to {save_file}")
