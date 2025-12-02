import os
import numpy as np
import trimesh
from skimage import measure
from PIL import Image
import matplotlib.pyplot as plt

# CONFIG
# We will use the 'gradient beam' image you just generated
INPUT_IMAGE = "results/gradient_beam/beam_structure.png" 
OUTPUT_STL = "results/gradient_beam/gradient_beam_3d.stl"
EXTRUSION_HEIGHT = 50  # How thick is the beam? (Number of layers)
VOXEL_SIZE = 1.0       # Size of one pixel in mm (Scale factor)

def load_and_process_image(path):
    if not os.path.exists(path):
        print(f"❌ Error: Image not found at {path}")
        return None
    
    # Load image, convert to grayscale
    img = Image.open(path).convert('L')
    img_arr = np.array(img).astype(np.float32) / 255.0
    
    # Threshold: Solid is typically Dark (0.0) in your plots.
    # Let's check: In 'beam_structure.png', did you plot Solids as Black?
    # If yes, we want pixels < 0.5 to be TRUE (Solid).
    # Adjust this threshold based on your visual inspection of the PNG.
    binary_slice = img_arr < 0.5 
    
    return binary_slice

def create_voxel_volume(binary_slice, depth):
    print(f"🔨 Extruding 2D slice ({binary_slice.shape}) to {depth} layers...")
    
    # Stack the 2D slice 'depth' times to make a 3D block
    volume = np.stack([binary_slice] * depth, axis=0)
    
    # Optional: Add a "Floor" (Base Plate) so it prints easily
    # We set the bottom 2 layers to be completely solid (True)
    volume[0:2, :, :] = True
    
    return volume

def export_stl(volume, output_path):
    print("🕸️  Running Marching Cubes (Voxel -> Mesh)...")
    
    # Marching Cubes algorithm (finds the surface of the 3D blob)
    # volume is boolean (True/False). Level=0.5 finds the boundary.
    try:
        verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)
        
        # Scale vertices (Optional)
        verts = verts * VOXEL_SIZE
        
        # Create Mesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Fix normals (make it look smooth)
        mesh.fix_normals()
        
        # Save
        mesh.export(output_path)
        print(f"✅ 3D Mesh saved to: {output_path}")
        print(f"   Vertices: {len(verts)}, Faces: {len(faces)}")
        
    except ValueError:
        print("⚠️  Error: No surface found. Is the image empty or all solid?")

def main():
    print("🚀 Starting 2D-to-3D Conversion...")
    
    # 1. Load
    binary_slice = load_and_process_image(INPUT_IMAGE)
    if binary_slice is None: return
    
    # 2. Extrude
    volume = create_voxel_volume(binary_slice, depth=EXTRUSION_HEIGHT)
    
    # 3. Export
    os.makedirs(os.path.dirname(OUTPUT_STL), exist_ok=True)
    export_stl(volume, OUTPUT_STL)
    
    print("\n🎉 DONE. You have escaped Flatland.")
    print("👉 Next Step: Drag that .stl file into https://www.viewstl.com/ to see it.")

if __name__ == "__main__":
    main()