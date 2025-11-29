import os
from PIL import Image
from torchvision import transforms

INPUT_DIR = "data/real_xylem_raw"
OUTPUT_DIR = "data/real_xylem_preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256))
])

files = sorted(os.listdir(INPUT_DIR))[:20]
print(f"🧪 Preprocessing {len(files)} images...")

for f in files:
    in_path = os.path.join(INPUT_DIR, f)
    out_path = os.path.join(OUTPUT_DIR, f)
    try:
        img = Image.open(in_path).convert("RGB")
        img_t = transform(img)
        img_t.save(out_path)
    except Exception as e:
        print(f"⚠️ Skipping {f}: {e}")

print(f"✅ Saved preprocessed images to {OUTPUT_DIR}")
