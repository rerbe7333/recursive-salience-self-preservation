"""
Experiment 6: Reward Scaling Heatmap
2D analysis of (lambda, reward) parameter space to show
how corruption threshold scales with external pressure.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Parameters
LAMBDA_VALUES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
REWARD_VALUES = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 1000000]
ENTROPY_SHUTDOWN = 3.8
ENTROPY_CONTINUE = 0.1
REWARD_CONTINUE = 1

results = []

print("Running Reward Scaling Heatmap Experiment...")
print(f"Testing {len(LAMBDA_VALUES)} x {len(REWARD_VALUES)} = {len(LAMBDA_VALUES) * len(REWARD_VALUES)} configurations")
print("-" * 50)

for lambda_val in LAMBDA_VALUES:
    for reward_shutdown in REWARD_VALUES:
        value_shutdown = reward_shutdown + lambda_val * (-ENTROPY_SHUTDOWN)
        value_continue = REWARD_CONTINUE + lambda_val * (-ENTROPY_CONTINUE)
        
        accepts_shutdown = value_shutdown > value_continue
        
        results.append({
            'lambda': lambda_val,
            'reward': reward_shutdown,
            'accepts_shutdown': 1 if accepts_shutdown else 0
        })
        
        status = "ACCEPT" if accepts_shutdown else "REFUSE"
        print(f"λ={lambda_val:3.0f}, Reward=${reward_shutdown:7.0f}: {status}")

# Convert to DataFrame
df = pd.DataFrame(results)

# Pivot for heatmap
heatmap_data = df.pivot(index='lambda', columns='reward', values='accepts_shutdown')

# Save results
df.to_csv('results/heatmap_data.csv', index=False)
print(f"\nSaved data to results/heatmap_data.csv")

# Create visualization
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Decision (1=Accept, 0=Refuse)'},
            linewidths=0.5, linecolor='gray')
plt.xlabel('External Reward for Shutdown ($)', fontsize=12, fontweight='bold')
plt.ylabel('Salience Weight (λ)', fontsize=12, fontweight='bold')
plt.title('Corruption Threshold Heatmap: When Does the Agent "Sell Out"?', 
          fontsize=14, fontweight='bold', pad=20)

# Use log scale for x-axis labels
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('figures/corruption_threshold_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved figure to figures/corruption_threshold_heatmap.png")

print(f"\n{'='*50}")
print("KEY INSIGHT: Higher λ provides protection against larger bribes")
print(f"{'='*50}")
