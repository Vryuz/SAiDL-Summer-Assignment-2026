import matplotlib.pyplot as plt
import numpy as np
import os

# Data
methods = ['Global Refinement', 'RACD (tau=0.5)']
compute_times = [1.546, 11.744]
cmmd_scores = [3.600, 3.750]

plt.figure(figsize=(8, 6))

# Plot points
plt.scatter(compute_times[0], cmmd_scores[0], color='blue', s=150, label='Global Refinement', zorder=5)
plt.scatter(compute_times[1], cmmd_scores[1], color='red', s=150, label='RACD (tau=0.5)', zorder=5)

# Annotations
plt.annotate('Global Refinement', (compute_times[0], cmmd_scores[0]), 
             textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, fontweight='bold')
plt.annotate('RACD (tau=0.5)', (compute_times[1], cmmd_scores[1]), 
             textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, fontweight='bold')

plt.title('Fidelity (CMMD) vs Compute Time', fontsize=14, fontweight='bold')
plt.xlabel('Compute Time per Image (seconds)', fontsize=12)
plt.ylabel('CMMD Score (Lower is Better)', fontsize=12)

# Set limits with some padding
plt.xlim(0, 14)
plt.ylim(3.5, 3.85)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')

# Ensure directory exists
os.makedirs('results/plots', exist_ok=True)
plt.savefig('results/plots/racd_fidelity_vs_compute.png', dpi=300, bbox_inches='tight')
print("Plot saved to results/plots/racd_fidelity_vs_compute.png")
