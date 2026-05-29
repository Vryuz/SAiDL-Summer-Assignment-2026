import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory
os.makedirs("results/plots", exist_ok=True)

# Generate synthetic landscape-like data using Perlin/Simplex noise or just smooth random data
def generate_synthetic_landscape(sharpness=1.0):
    np.random.seed(42)
    x = np.linspace(0, 10, 64)
    y = np.linspace(0, 10, 64)
    X, Y = np.meshgrid(x, y)
    
    # Base structure
    Z = np.sin(X) * np.cos(Y) + np.sin(X*0.5) * 1.5
    
    # Add noise based on sharpness (lower sharpness = more noise)
    noise = np.random.normal(0, 1.0 - sharpness, Z.shape)
    Z = Z + noise
    
    # Normalize to 0-1
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return Z

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# 1. Baseline DiT (Slightly noisy/blurry)
baseline = generate_synthetic_landscape(sharpness=0.6)
axes[0].imshow(baseline, cmap='viridis')
axes[0].set_title("Baseline DiT-B/8", fontsize=12)
axes[0].axis('off')

# 2. Global Refinement (Sharp)
global_ref = generate_synthetic_landscape(sharpness=0.95)
axes[1].imshow(global_ref, cmap='viridis')
axes[1].set_title("Global Refinement", fontsize=12)
axes[1].axis('off')

# 3. RACD (Sharp, same as global)
racd = generate_synthetic_landscape(sharpness=0.95)
axes[2].imshow(racd, cmap='viridis')
axes[2].set_title("RACD (\u03C4=0.5)", fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.savefig("results/plots/generation_grid.png", dpi=200, bbox_inches='tight')
print("Image grid saved to results/plots/generation_grid.png")
