import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Try to import Trimesh + Submodules
try:
    import trimesh
    from trimesh.voxel.encoding import Packed3D
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("⚠️ Trimesh not fully available. Will skip STL and just generate PNG.")

# CONFIG
INPUT_IMAGE = "data/generated_microtubes/synthetic_098.png"
OUTPUT_STL = "results/titanium_wood.stl"
OUTPUT_PNG = "results/titanium_wood_preview.png"

def main():
    print(f"🦖 Resurrection Protocol: Targeted Extraction")
    print(f"   Target: {INPUT_IMAGE}")

    if not os.path.exists(INPUT_IMAGE):
        print("❌ Error: Candidate image not found.")
        return

    # 1. Load the Image & Binarize
    img = Image.open(INPUT_IMAGE).convert('L')
    img = img.resize((128, 128)) 
    arr = np.array(img)
    
    # Threshold: Dark pixels are solid walls (< 153)
    solid_mask = arr < 153
    
    points_for_plot = None

    # 2. Try to Generate 3D Mesh (STL)
    if HAS_TRIMESH:
        try:
            print("   🔨 Extruding to 3D (Voxel Method)...")
            volume = np.stack([solid_mask] * 20, axis=0) 
            
            # Use the explicit import class
            voxel_grid = trimesh.voxel.VoxelGrid(Packed3D(volume))
            mesh = voxel_grid.as_boxes()
            
            print(f"   ✅ Mesh generated: {len(mesh.vertices)} vertices")
            mesh.export(OUTPUT_STL)
            print(f"   💾 Saved STL: {OUTPUT_STL}")
            
            points_for_plot = mesh.vertices
            
        except Exception as e:
            print(f"⚠️ STL Generation skipped due to error: {e}")
            print("   -> Proceeding to Point Cloud visualization only.")

    # 3. Generate Preview Snapshot (PNG)
    # If mesh failed, fallback to raw voxel coordinates
    if points_for_plot is None:
        print("   ⚠️ Using raw voxel points for preview...")
        # Get coordinates of all solid pixels
        z, y, x = np.where(np.stack([solid_mask] * 20, axis=0))
        points_for_plot = np.column_stack((x, y, z))

    print("   📸 Taking snapshot...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample points for the plot so it doesn't crash
    points = points_for_plot
    if len(points) > 10000:
        idx = np.random.choice(len(points), 10000, replace=False)
        points = points[idx]
        
    ax.scatter(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap='magma', s=1)
    ax.set_title("Titanium Wood Candidate (Internal Lattice)")
    ax.view_init(elev=90, azim=-90) # Top-down view
    
    plt.savefig(OUTPUT_PNG)
    print(f"   🖼️  Preview saved: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()