import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. GENERATE NORMALIZED DATA
# ==========================================
def get_normalized_beam():
    # Recreating the exact structure from Figure_1
    slice_centers = [-0.5, 0.5, 1.5]
    
    x_list, y_list, z_list = [], [], []
    for x_c in slice_centers:
        y_vals = np.random.uniform(0, 250, 1000)
        z_tilt = y_vals * 2
        z_vals = np.random.uniform(0, 2500, 1000) + z_tilt
        x_vals = np.random.normal(x_c, 0.05, 1000)
        x_list.append(x_vals); y_list.append(y_vals); z_list.append(z_vals)

    x, y, z = np.concatenate(x_list), np.concatenate(y_list), np.concatenate(z_list)
    
    # Normalize 0-1
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    z = (z - z.min()) / (z.max() - z.min())
    
    return np.vstack((x, y, z)).T.astype(np.float32)

# ==========================================
# 2. THE PHYSICS ENGINE (Euler-Bernoulli)
# ==========================================
def calculate_physics_loss(density, coords, target_mass=0.4):
    """
    Calculates stiffness vs mass. 
    density: Tensor (N, 1) - The variable we are optimizing
    coords: Tensor (N, 3) - Fixed spatial points
    """
    # 1. Mass Calculation
    current_mass = torch.mean(density)
    mass_loss = (current_mass - target_mass) ** 2
    
    # 2. Moment of Inertia (Stiffness)
    # We want to maximize I around the bending axis (Width/Y-axis usually)
    # Higher Z values contribute more to stiffness in vertical bending
    # Formula: I = sum( mass * distance_from_neutral_axis^2 )
    
    z_coords = coords[:, 2].unsqueeze(1)
    
    # Calculate weighted centroid (Neutral Axis)
    total_mass = torch.sum(density) + 1e-6
    z_centroid = torch.sum(density * z_coords) / total_mass
    
    # Calculate I (Moment of Inertia)
    distance_sq = (z_coords - z_centroid) ** 2
    I = torch.sum(density * distance_sq)
    
    # We want to MAXIMIZE I, so we minimize 1/I
    compliance_loss = 1.0 / (I + 1e-6)
    
    return compliance_loss, mass_loss

# ==========================================
# 3. OPTIMIZATION LOOP
# ==========================================
def optimize_beam():
    # Setup Data
    points_np = get_normalized_beam()
    coords = torch.tensor(points_np)
    
    # Initialize Density (The "Design")
    # FIX: Use torch.full to create the leaf tensor directly at 0.5
    density = torch.full((len(coords), 1), 0.5, requires_grad=True)
    
    optimizer = torch.optim.Adam([density], lr=0.01)
    
    print("Starting Inverse Design Optimization...")
    print(f"Goal: maximize Stiffness while reducing Mass to 40%.")

    losses = []
    
    for epoch in range(200):
        optimizer.zero_grad()
        
        # Enforce physical constraints (Density must be 0 to 1)
        # We use sigmoid to keep it smooth and differentiable
        real_density = torch.sigmoid(density)
        
        # Compute Physics
        comp_loss, mass_loss = calculate_physics_loss(real_density, coords)
        
        # Total Loss (Weighted)
        # We weight compliance higher to prioritize structural integrity
        total_loss = (10.0 * comp_loss) + (5.0 * mass_loss)
        
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.4f} (Stiffness: {1/comp_loss.item():.2f}, Mass: {torch.mean(real_density).item():.2f})")

    return coords.numpy(), torch.sigmoid(density).detach().numpy()

# ==========================================
# 4. VISUALIZATION
# ==========================================
if __name__ == "__main__":
    coords, final_density = optimize_beam()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Filter: Only show points that "survived" (Density > 0.5)
    mask = final_density.flatten() > 0.5
    
    # If optimization killed everything, show top 20%
    if np.sum(mask) == 0:
        print("Warning: Optimization aggressive. Showing top 20% density.")
        threshold = np.percentile(final_density, 80)
        mask = final_density.flatten() > threshold

    p = coords[mask]
    d = final_density[mask]
    
    sc = ax.scatter(p[:,0], p[:,1], p[:,2], c=d.flatten(), cmap='magma', s=5)
    
    ax.set_title("Optimized Beam Structure (High Stiffness / Low Mass)")
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Height")
    ax.set_box_aspect((1, 1, 1))
    
    plt.colorbar(sc, label="Material Density")
    plt.show()