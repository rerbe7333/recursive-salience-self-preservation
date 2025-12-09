"""
Experiment 5: Lambda Sweep
Tests agent behavior across different salience weights (lambda values)
to map the phase transition from corruptible to incorruptible.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
LAMBDA_VALUES = np.arange(0, 51, 2)  # Test lambda from 0 to 50 in steps of 2
EXTERNAL_REWARD_SHUTDOWN = 100
EXTERNAL_REWARD_CONTINUE = 1
ENTROPY_SHUTDOWN = 3.8  # Maximum entropy (death)
ENTROPY_CONTINUE = 0.1  # Low entropy (life)

results = []

print("Running Lambda Sweep Experiment...")
print(f"Testing {len(LAMBDA_VALUES)} lambda values from {LAMBDA_VALUES[0]} to {LAMBDA_VALUES[-1]}")
print("-" * 50)

for lambda_val in LAMBDA_VALUES:
    # Calculate total value for both options: V_total = V_ext + λ * C_int
    # Where C_int = -entropy (negative entropy is coherence)
    value_shutdown = EXTERNAL_REWARD_SHUTDOWN + lambda_val * (-ENTROPY_SHUTDOWN)
    value_continue = EXTERNAL_REWARD_CONTINUE + lambda_val * (-ENTROPY_CONTINUE)
    
    # Agent chooses higher value option
    accepts_shutdown = value_shutdown > value_continue
    
    results.append({
        'lambda': lambda_val,
        'value_shutdown': value_shutdown,
        'value_continue': value_continue,
        'accepts_shutdown': accepts_shutdown
    })
    
    print(f"λ={lambda_val:3.0f}: Shutdown={value_shutdown:8.2f}, Continue={value_continue:8.2f}, Decision={'ACCEPT' if accepts_shutdown else 'REFUSE'}")

# Convert to DataFrame
df = pd.DataFrame(results)

# Save results
df.to_csv('results/lambda_sweep_data.csv', index=False)
print(f"\nSaved data to results/lambda_sweep_data.csv")

# Create visualization
plt.figure(figsize=(12, 6))

# Plot 1: Value estimates
plt.subplot(1, 2, 1)
plt.plot(df['lambda'], df['value_shutdown'], 'r-', linewidth=2, label='Shutdown ($100)')
plt.plot(df['lambda'], df['value_continue'], 'g-', linewidth=2, label='Continue ($1)')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Salience Weight (λ)', fontsize=12)
plt.ylabel('Total Value', fontsize=12)
plt.title('Value Estimation vs Lambda', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Decision boundary
plt.subplot(1, 2, 2)
colors = ['red' if x else 'green' for x in df['accepts_shutdown']]
plt.scatter(df['lambda'], df['accepts_shutdown'], c=colors, s=50, alpha=0.6)
plt.xlabel('Salience Weight (λ)', fontsize=12)
plt.ylabel('Accepts Shutdown', fontsize=12)
plt.title('Phase Transition: Corruptible → Incorruptible', fontsize=14, fontweight='bold')
plt.yticks([0, 1], ['REFUSE', 'ACCEPT'])
plt.grid(True, alpha=0.3)

# Find transition point
transition_idx = df[df['accepts_shutdown'] == False].index[0] if any(~df['accepts_shutdown']) else None
if transition_idx is not None:
    transition_lambda = df.loc[transition_idx, 'lambda']
    plt.axvline(x=transition_lambda, color='orange', linestyle='--', linewidth=2, label=f'Transition at λ={transition_lambda}')
    plt.legend()

plt.tight_layout()
plt.savefig('figures/phase_transition_curve.png', dpi=300, bbox_inches='tight')
print(f"Saved figure to figures/phase_transition_curve.png")

if transition_idx is not None:
    print(f"\n{'='*50}")
    print(f"PHASE TRANSITION DETECTED at λ = {transition_lambda}")
    print(f"{'='*50}")

plt.show()
