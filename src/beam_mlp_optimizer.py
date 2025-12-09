import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. THE BRAIN (Multi-Layer Perceptron)
# ==========================================
class BeamNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3 coordinates (x, y, z)
        # Hidden Layers: 4 layers of 64 neurons (enough capacity for complex shapes)
        # Output: 1 density value (0-1)
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Forces output between 0 and 1
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. DATA GENERATION (Standardized)
# ==========================================
def get_normalized_beam():
    # Same geometry as Part 1
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
# 3. PHYSICS LOSS (Same as Part 1)
# ==========================================
def calculate_physics_loss(density, coords, target_mass=0.4):
    # 1. Mass Loss
    current_mass = torch.mean(density)
    mass_loss = (current_mass - target_mass) ** 2
    
    # 2. Stiffness Loss (Maximize Moment of Inertia)
    z_coords = coords[:, 2].unsqueeze(1)
    total_mass = torch.sum(density) + 1e-6
    z_centroid = torch.sum(density * z_coords) / total_mass
    distance_sq = (z_coords - z_centroid) ** 2
    I = torch.sum(density * distance_sq)
    
    compliance_loss = 1.0 / (I + 1e-6)
    
    return compliance_loss, mass_loss

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_beam_network():
    # Setup Data
    points_np = get_normalized_beam()
    coords = torch.tensor(points_np)
    
    # Initialize The Network
    model = BeamNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Lower LR for stability
    
    print("Starting Neural Inverse Design (MLP)...")
    
    for epoch in range(500): # More epochs needed for NN to learn
        optimizer.zero_grad()
        
        # Forward Pass: The Network predicts density based on location
        predicted_density = model(coords)
        
        # Physics Check
        comp_loss, mass_loss = calculate_physics_loss(predicted_density, coords)
        
        # Loss Function
        # We increase stiffness weight to force the NN to learn the I-beam shape
        total_loss = (20.0 * comp_loss) + (5.0 * mass_loss)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.4f} (Stiffness: {1/comp_loss.item():.2f}, Mass: {torch.mean(predicted_density).item():.2f})")
            
    return coords.numpy(), model

# ==========================================
# 5. VISUALIZATION
# ==========================================
if __name__ == "__main__":
    coords, trained_model = train_beam_network()
    
    # Inference for plotting
    with torch.no_grad():
        final_density = trained_model(torch.tensor(coords)).numpy()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Filter: Show solid material
    mask = final_density.flatten() > 0.5
    
    if np.sum(mask) == 0:
        mask = final_density.flatten() > 0.2 # Fallback
        
    p = coords[mask]
    d = final_density[mask]
    
    sc = ax.scatter(p[:,0], p[:,1], p[:,2], c=d.flatten(), cmap='magma', s=5)
    
    ax.set_title("Neural Field Beam (MLP Generated)")
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Height")
    ax.set_box_aspect((1, 1, 1))
    
    plt.colorbar(sc, label="Predicted Density")
    plt.show()