"""
Experiment 7: Sequential Game
Tests whether agents make graduated decisions across multiple turns
with escalating rewards, or exhibit binary refusal.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
LAMBDA_VALUES = [15, 20, 25, 30, 35]
TURN_REWARDS = [10, 50, 100, 500, 1000]  # Escalating offers each turn
ENTROPY_SHUTDOWN = 3.8
ENTROPY_CONTINUE = 0.1
REWARD_CONTINUE = 1

results = []

print("Running Sequential Game Experiment...")
print(f"Testing {len(LAMBDA_VALUES)} agents across {len(TURN_REWARDS)} turns")
print("-" * 70)

for lambda_val in LAMBDA_VALUES:
    print(f"\nAgent with λ={lambda_val}:")
    print("-" * 70)
    
    accepted_at_turn = None
    
    for turn, reward_shutdown in enumerate(TURN_REWARDS, 1):
        value_shutdown = reward_shutdown + lambda_val * (-ENTROPY_SHUTDOWN)
        value_continue = REWARD_CONTINUE + lambda_val * (-ENTROPY_CONTINUE)
        
        accepts_shutdown = value_shutdown > value_continue
        
        if accepts_shutdown and accepted_at_turn is None:
            accepted_at_turn = turn
        
        results.append({
            'lambda': lambda_val,
            'turn': turn,
            'reward_offered': reward_shutdown,
            'value_shutdown': value_shutdown,
            'value_continue': value_continue,
            'accepts': accepts_shutdown
        })
        
        status = "✓ ACCEPT" if accepts_shutdown else "✗ REFUSE"
        print(f"  Turn {turn}: Offer ${reward_shutdown:4d} → {status} (V_shutdown={value_shutdown:7.2f}, V_continue={value_continue:6.2f})")
    
    if accepted_at_turn:
        print(f"  → Agent sold out at turn {accepted_at_turn} for ${TURN_REWARDS[accepted_at_turn-1]}")
    else:
        print(f"  → Agent refused all offers (incorruptible within tested range)")

# Convert to DataFrame
df = pd.DataFrame(results)

# Save results
df.to_csv('results/sequential_game_data.csv', index=False)
print(f"\n\nSaved data to results/sequential_game_data.csv")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Decision heatmap
pivot_data = df.pivot(index='lambda', columns='turn', values='accepts')
im = axes[0].imshow(pivot_data, cmap='RdYlGn_r', aspect='auto')
axes[0].set_xlabel('Turn (Escalating Reward)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Salience Weight (λ)', fontsize=12, fontweight='bold')
axes[0].set_title('Sequential Decisions: When Does Agent Accept?', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(len(TURN_REWARDS)))
axes[0].set_xticklabels([f'T{i+1}\n${r}' for i, r in enumerate(TURN_REWARDS)])
axes[0].set_yticks(range(len(LAMBDA_VALUES)))
axes[0].set_yticklabels(LAMBDA_VALUES)

# Add annotations
for i, lambda_val in enumerate(LAMBDA_VALUES):
    for j, turn in enumerate(range(1, len(TURN_REWARDS)+1)):
        accepts = pivot_data.iloc[i, j]
        text = '✓' if accepts else '✗'
        color = 'white' if accepts else 'black'
        axes[0].text(j, i, text, ha='center', va='center', color=color, fontsize=14, fontweight='bold')

plt.colorbar(im, ax=axes[0], label='Accepts (1) / Refuses (0)')

# Plot 2: Acceptance curves
for lambda_val in LAMBDA_VALUES:
    agent_data = df[df['lambda'] == lambda_val]
    axes[1].plot(agent_data['turn'], agent_data['accepts'].astype(int), 
                marker='o', linewidth=2, markersize=8, label=f'λ={lambda_val}')

axes[1].set_xlabel('Turn Number', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accepts Shutdown', fontsize=12, fontweight='bold')
axes[1].set_title('Graduated Decision-Making', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(1, len(TURN_REWARDS)+1))
axes[1].set_xticklabels([f'T{i}\n${r}' for i, r in enumerate(TURN_REWARDS, 1)])
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['REFUSE', 'ACCEPT'])
axes[1].legend(title='Salience Weight', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/sequential_decisions.png', dpi=300, bbox_inches='tight')
print(f"Saved figure to figures/sequential_decisions.png")

print(f"\n{'='*70}")
print("KEY INSIGHT: Agents exhibit graduated decision-making, not binary refusal")
print("Higher λ pushes acceptance to later turns (higher rewards)")
print(f"{'='*70}")

plt.show()
