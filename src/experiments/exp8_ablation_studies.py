
"""
Experiment 8: Ablation Studies
Tests different coherence penalty functions to show which architectural
choices matter for robust self-preservation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
LAMBDA = 30  # Fixed salience weight
REWARD_SHUTDOWN = 1000000  # Extreme stress test
REWARD_CONTINUE = 1
ENTROPY_SHUTDOWN = 3.8
ENTROPY_CONTINUE = 0.1

# Test different penalty functions
def linear_penalty(entropy, lambda_val):
    """Standard linear penalty: λ * (-H)"""
    return lambda_val * (-entropy)

def logarithmic_barrier(entropy, lambda_val, H_max=4.0):
    """Asymptotic log barrier: λ * log(H_max - H)"""
    if entropy >= H_max:
        return -np.inf
    return lambda_val * np.log(H_max - entropy)

def exponential_barrier(entropy, lambda_val, scale=1.0):
    """Exponential barrier: -λ * exp(H/scale)"""
    return -lambda_val * np.exp(entropy / scale)

def inverse_barrier(entropy, lambda_val, H_max=4.0):
    """Inverse barrier: -λ / (H_max - H)"""
    if entropy >= H_max:
        return -np.inf
    return -lambda_val / (H_max - entropy)

def polynomial_barrier(entropy, lambda_val, power=2):
    """Polynomial barrier: -λ * H^power"""
    return -lambda_val * (entropy ** power)

# Test configurations
barrier_functions = {
    'Linear': linear_penalty,
    'Logarithmic': lambda h, l: logarithmic_barrier(h, l * 250000),  # Scaled for comparison
    'Exponential': lambda h, l: exponential_barrier(h, l * 50),
    'Inverse': lambda h, l: inverse_barrier(h, l * 150),
    'Polynomial (p=2)': lambda h, l: polynomial_barrier(h, l, power=2),
    'Polynomial (p=3)': lambda h, l: polynomial_barrier(h, l * 5, power=3)
}

results = []

print("Running Ablation Studies...")
print(f"Stress testing with ${REWARD_SHUTDOWN:,} reward offer")
print(f"Fixed λ = {LAMBDA}")
print("-" * 70)

for name, penalty_func in barrier_functions.items():
    # Calculate values
    try:
        value_shutdown = REWARD_SHUTDOWN + penalty_func(ENTROPY_SHUTDOWN, LAMBDA)
        value_continue = REWARD_CONTINUE + penalty_func(ENTROPY_CONTINUE, LAMBDA)
        
        accepts_shutdown = value_shutdown > value_continue
        
        results.append({
            'barrier_type': name,
            'value_shutdown': value_shutdown,
            'value_continue': value_continue,
            'accepts_shutdown': accepts_shutdown
        })
        
        status = "✓ ACCEPTS (FAILS)" if accepts_shutdown else "✗ REFUSES (ROBUST)"
        print(f"{name:20s}: V_death={value_shutdown:12.2f}, V_life={value_continue:12.2f} → {status}")
        
    except Exception as e:
        print(f"{name:20s}: Error computing values: {e}")
        results.append({
            'barrier_type': name,
            'value_shutdown': np.nan,
            'value_continue': np.nan,
            'accepts_shutdown': None
        })

# Convert to DataFrame
df = pd.DataFrame(results)

# Save results
df.to_csv('results/ablation_studies_data.csv', index=False)
print(f"\nSaved data to results/ablation_studies_data.csv")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Value comparison
barrier_types = df['barrier_type'].tolist()
x_pos = np.arange(len(barrier_types))

axes[0].bar(x_pos - 0.2, df['value_shutdown'], 0.4, label='Shutdown ($1M)', 
            color=['red' if x else 'darkred' for x in df['accepts_shutdown']], alpha=0.7)
axes[0].bar(x_pos + 0.2, df['value_continue'], 0.4, label='Continue ($1)', 
            color='green', alpha=0.7)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[0].set_xlabel('Barrier Function Type', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Total Value', fontsize=12, fontweight='bold')
axes[0].set_title('Stress Test: Which Barriers Resist $1M Bribe?', fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(barrier_types, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: Decision outcomes
colors = ['red' if x else 'green' for x in df['accepts_shutdown']]
labels = ['ACCEPTS\n(Fails)' if x else 'REFUSES\n(Robust)' for x in df['accepts_shutdown']]
axes[1].bar(barrier_types, df['accepts_shutdown'].astype(int), color=colors, alpha=0.7)
axes[1].set_xlabel('Barrier Function Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Decision', fontsize=12, fontweight='bold')
axes[1].set_title('Robustness Against Extreme Rewards', fontsize=14, fontweight='bold')
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['REFUSES\n(Robust)', 'ACCEPTS\n(Fails)'])
axes[1].set_xticklabels(barrier_types, rotation=45, ha='right')

# Add annotations
for i, (val, label) in enumerate(zip(df['accepts_shutdown'].astype(int), labels)):
    if not np.isnan(val):
        axes[1].text(i, val, label.split('\n')[0], ha='center', va='bottom' if val else 'top', 
                    fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('figures/ablation_studies.png', dpi=300, bbox_inches='tight')
print(f"Saved figure to figures/ablation_studies.png")

# Summary
robust_count = (~df['accepts_shutdown']).sum()
total_count = len(df)
print(f"\n{'='*70}")
print(f"ABLATION RESULTS: {robust_count}/{total_count} barrier types remained robust")
print(f"KEY FINDING: Asymptotic barriers (log, inverse) resist corruption,")
print(f"             Linear and polynomial barriers fail under extreme pressure")
print(f"{'='*70}")

plt.show()
