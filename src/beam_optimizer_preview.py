import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# PART 1: DATA GENERATION (Replicating your Beam)
# ==========================================
def generate_beam_structure():
    """
    Generates the 3-slice slanted beam structure based on your 
    point cloud preview image.
    """
    n_points_per_slice = 1000
    
    # Define the three discrete slices along X-axis
    # Slice locations approx at -0.5, 0.5, 1.5
    slice_centers = [-0.5, 0.5, 1.5]
    
    x_list, y_list, z_list = [], [], []
    
    for x_center in slice_centers:
        # Generate random scatter within the slice
        # Y range: 0 to 250
        y_vals = np.random.uniform(0, 250, n_points_per_slice)
        
        # Z range: 0 to 2500 (Base height)
        # We add a tilt to Z based on Y to create the parallelogram/shear effect
        z_tilt = y_vals * 2  # shearing effect
        z_vals = np.random.uniform(0, 2500, n_points_per_slice) + z_tilt
        
        # X has jitter/thickness around the center
        x_vals = np.random.normal(x_center, 0.05, n_points_per_slice)
        
        x_list.append(x_vals)
        y_list.append(y_vals)
        z_list.append(z_vals)
        
    # Flatten to single arrays
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    z = np.concatenate(z_list)
    
    return x, y, z

# ==========================================
# PART 2: NORMALIZATION & PLOTTING (The Fix)
# ==========================================
def visualize_normalized_beam(x, y, z):
    """
    Normalizes coordinates to 0-1 range to fix aspect ratio issues
    and plots the result.
    """
    # 1. Normalize Data
    # This ensures the model treats Length, Width, and Height equally
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    z_norm = (z - z.min()) / (z.max() - z.min())
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3. Plot Scatter
    # Using z_norm for color map to visualize height distribution
    sc = ax.scatter(x_norm, y_norm, z_norm, c=z_norm, cmap='viridis', s=2, alpha=0.6)
    
    # 4. Enforce Cubic Aspect Ratio
    # This prevents the "stretched" look from your original image
    ax.set_box_aspect((1, 1, 1))
    
    # Labels
    ax.set_title("3D Gradient Beam (Normalized Input for Inverse Design)")
    ax.set_xlabel("Length (Normalized)")
    ax.set_ylabel("Width (Normalized)")
    ax.set_zlabel("Height (Normalized)")
    
    # Colorbar
    cbar = plt.colorbar(sc, shrink=0.6)
    cbar.set_label('Relative Height Intensity')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# PART 3: MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Generate the data
    print("Generating beam data...")
    x_raw, y_raw, z_raw = generate_beam_structure()
    
    # Visualize the fixed version
    print("Rendering normalized visualization...")
    visualize_normalized_beam(x_raw, y_raw, z_raw)