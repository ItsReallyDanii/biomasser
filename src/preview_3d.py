import trimesh
import matplotlib.pyplot as plt
import numpy as np
import os

MESH_FILE = "results/gradient_beam/gradient_beam_3d.stl"
OUTPUT_IMG = "results/gradient_beam/3d_preview.png"

def main():
    if not os.path.exists(MESH_FILE):
        print("❌ STL file not found.")
        return

    print(f"👀 Loading mesh for preview...")
    mesh = trimesh.load(MESH_FILE)
    
    # Downsample: 1.3M points is too heavy for matplotlib. We take 5,000 random points.
    print(f"   Mesh has {len(mesh.vertices)} vertices. Sampling 5,000 for preview...")
    indices = np.random.choice(len(mesh.vertices), 5000, replace=False)
    points = mesh.vertices[indices]

    # Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot (X=Length, Y=Width, Z=Height)
    # Coloring by Z (Height) to show layers
    ax.scatter(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap='viridis', s=1, alpha=0.6)
    
    # Clean up axis
    ax.set_title("3D Gradient Beam (Point Cloud Preview)")
    ax.set_xlabel("Length (X)")
    ax.set_ylabel("Width (Y)")
    ax.set_zlabel("Height (Z)")
    
    # Force Aspect Ratio so it doesn't look squashed
    # (Matplotlib 3D aspect ratio is tricky, usually requires manual box setting)
    
    plt.savefig(OUTPUT_IMG, dpi=100)
    print(f"✅ Preview image saved to: {OUTPUT_IMG}")
    print("   Open this image to see your structure!")

if __name__ == "__main__":
    main()