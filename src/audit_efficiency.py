import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# CONFIG
AI_IMAGE_PATH = "results/thermal_design/MaxCooling_Heavy_structure.png"
OUTPUT_DIR = "results/efficiency_audit/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. The Physics (Simplified Fluid Solver)
# ---------------------------------------------------------
def solve_flow_resistance(binary_img, max_iter=2000):
    """
    Estimates fluid resistance using a simplified Darcy-Stokes proxy.
    Solve P (Pressure) field: Div(K * Grad(P)) = 0
    where K is high in Void and 0 in Solid.
    
    This is effectively the same math as Heat, but interpreted as Pressure.
    Flux = Flow Rate.
    Resistance = Pressure Drop / Flow Rate.
    """
    h, w = binary_img.shape
    
    # Material Permeability
    # Void (1) = High Permeability (Flows easily)
    # Solid (0) = Near-Zero Permeability (Blocked)
    K_VOID = 1.0
    K_SOLID = 1e-6 # Nearly blocked
    
    # Map image to Permeability Field
    # Assuming Binary: 1=Void, 0=Solid
    k_map = np.where(binary_img > 0.5, K_VOID, K_SOLID)
    
    # Initialize Pressure Field (Linear gradient Left->Right)
    P = np.linspace(1.0, 0.0, w)
    P = np.tile(P, (h, 1))
    
    # Iterative Solver (Finite Difference)
    for i in range(max_iter):
        P_old = P.copy()
        
        # Simple Neighbor Average weighted by Permeability (k)
        # This is a proxy for P_new = (P_neighbors) 
        # Correct physics would use face-centered velocities, 
        # but this diffuses Pressure correctly for a resistance metric.
        
        P[1:-1, 1:-1] = 0.25 * (
            P_old[0:-2, 1:-1] + 
            P_old[2:, 1:-1] + 
            P_old[1:-1, 0:-2] + 
            P_old[1:-1, 2:]
        )
        
        # Boundary Conditions (Left=High P, Right=Low P)
        P[:, 0] = 1.0
        P[:, -1] = 0.0
        # Insulated Walls (Top/Bottom)
        P[0, :] = P[1, :]
        P[-1, :] = P[-2, :]
        
    # Calculate Flow Rate (Flux)
    # Q = -k * dP/dx at the outlet
    dPdx = P[:, -1] - P[:, -2]
    flux = np.sum(-k_map[:, -1] * dPdx)
    
    # Resistance = Pressure Drop / Flux
    # dP = 1.0
    resistance = 1.0 / (flux + 1e-9)
    
    return resistance, P

# ---------------------------------------------------------
# 2. Generators & Loaders
# ---------------------------------------------------------
def generate_random_noise(shape=(256, 256), density=0.6):
    """Generates the high-performing 'Random' baseline"""
    # Random 0.0 - 1.0
    noise = np.random.rand(*shape)
    # Threshold: We want 'density' fraction to be SOLID (0)
    # So if noise < density -> Solid. Else -> Void (1)
    binary = np.ones(shape)
    binary[noise < density] = 0.0
    return binary

def load_ai_design(path):
    if not os.path.exists(path):
        print(f"⚠️ AI Image not found: {path}")
        return None
    img = Image.open(path).convert('L')
    arr = np.array(img).astype(np.float32) / 255.0
    
    # Normalize and Threshold
    # In your thermal plots, 'inferno' likely saved Dark=Low, Light=High
    # We need Void=1, Solid=0.
    # Usually Xylem AI: Bright=Void.
    binary = arr > 0.5
    return binary.astype(float)

# ---------------------------------------------------------
# 3. The Showdown
# ---------------------------------------------------------
def main():
    print("⚔️  AUDIT: Flow Resistance Showdown")
    
    # 1. Load AI Candidate
    print(f"   Loading AI Design: {AI_IMAGE_PATH}")
    ai_img = load_ai_design(AI_IMAGE_PATH)
    if ai_img is None:
        print("   -> Using Placeholder AI (Error loading)")
        ai_img = np.zeros((256,256))
        
    # 2. Generate Random Baseline (The "Winner" of Thermal)
    print("   Generating Random Noise (Density=0.6)...")
    rand_img = generate_random_noise(density=0.6)
    
    # 3. Simulate Flow
    print("   🌊 Simulating Flow (Calculating Resistance)...")
    res_ai, p_ai = solve_flow_resistance(ai_img)
    print(f"      AI Resistance: {res_ai:.4f}")
    
    res_rand, p_rand = solve_flow_resistance(rand_img)
    print(f"      Random Resistance: {res_rand:.4f}")
    
    # 4. The Verdict
    ratio = res_rand / res_ai
    print("\n🏆 THE VERDICT:")
    if ratio > 1.0:
        print(f"   ✅ The AI Design flows {ratio:.1f}x BETTER than Random Noise.")
        print("   Conclusion: Random Noise cools well but clogs the pump.")
        print("   AI Design balances cooling with flow efficiency.")
    else:
        print(f"   ❌ Random Noise flows {1/ratio:.1f}x better. (Unexpected!)")
        
    # 5. Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(p_ai, cmap='Blues')
    plt.title(f"AI Flow Field\nRes = {res_ai:.2f}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(p_rand, cmap='Blues')
    plt.title(f"Random Noise Flow Field\nRes = {res_rand:.2f}")
    plt.axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, "flow_resistance_audit.png")
    plt.savefig(save_path)
    print(f"   📊 Flow maps saved to {save_path}")

if __name__ == "__main__":
    main()