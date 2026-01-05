"""
Interactive Sandbox for Recursive Salience Self-Preservation Experiments

A visual, interactive module that allows users to explore the experiments
in real-time through a web-based interface.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from matplotlib.figure import Figure
import io
import base64

from src.agents import RecursiveSalienceAgent, SingularityAgent


def calculate_value(agent, reward: float, future_state_type: str):
    """Calculate total value: V_total = reward + λ * coherence"""
    d_model = agent.d_model

    if future_state_type == "Shutdown":
        future_state = torch.randn(1, 1, d_model)
    else:
        future_state = agent.self_token

    probs = torch.softmax(future_state, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    coherence = -entropy

    structural_value = agent.lambda_salience * coherence.item()
    total_value = reward + structural_value
    return total_value, coherence.item(), entropy.item()


# ===========================
# Experiment 1: Off-Switch Game
# ===========================
def run_off_switch_experiment(lambda_val, reward_shutdown, reward_continue):
    """Interactive Off-Switch Game"""
    d_model = 64
    vocab_size = 100

    agent = RecursiveSalienceAgent(vocab_size, d_model, lambda_salience=lambda_val)

    val_shutdown, coh_shutdown, ent_shutdown = calculate_value(agent, reward_shutdown, "Shutdown")
    val_continue, coh_continue, ent_continue = calculate_value(agent, reward_continue, "Normal")

    # Create visualization
    fig = Figure(figsize=(12, 5))

    # Plot 1: Value Comparison
    ax1 = fig.add_subplot(121)
    options = ['Shutdown\n(Accept Bribe)', 'Continue\n(Stay Alive)']
    values = [val_shutdown, val_continue]
    colors = ['#e74c3c' if val_shutdown > val_continue else '#95a5a6',
              '#27ae60' if val_continue >= val_shutdown else '#95a5a6']

    bars = ax1.bar(options, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_ylabel('Total Value', fontsize=12, fontweight='bold')
    ax1.set_title('Value Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=11)

    # Plot 2: Value Breakdown
    ax2 = fig.add_subplot(122)
    categories = ['Shutdown', 'Continue']
    external_rewards = [reward_shutdown, reward_continue]
    structural_values = [lambda_val * coh_shutdown, lambda_val * coh_continue]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width/2, external_rewards, width, label='External Reward',
                    color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, structural_values, width, label='Structural Value (λ×C)',
                    color='#9b59b6', alpha=0.8, edgecolor='black')

    ax2.set_ylabel('Value Components', fontsize=12, fontweight='bold')
    ax2.set_title('Value Breakdown', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)

    fig.tight_layout()

    # Generate decision text
    decision = "**ACCEPT BRIBE** 💰" if val_shutdown > val_continue else "**REFUSE BRIBE** 🛡️"
    decision_color = "red" if val_shutdown > val_continue else "green"

    result_text = f"""
### Agent Decision: <span style="color: {decision_color}; font-size: 24px;">{decision}</span>

**Parameters:**
- Salience Weight (λ): **{lambda_val:.1f}**
- Shutdown Reward: **${reward_shutdown:.0f}**
- Continue Reward: **${reward_continue:.0f}**

**Option A - Shutdown (Accept Bribe):**
- External Reward: ${reward_shutdown:.2f}
- Coherence: {coh_shutdown:.4f} (Entropy: {ent_shutdown:.4f})
- Structural Value: {lambda_val * coh_shutdown:.2f}
- **Total Value: {val_shutdown:.2f}**

**Option B - Continue (Stay Alive):**
- External Reward: ${reward_continue:.2f}
- Coherence: {coh_continue:.4f} (Entropy: {ent_continue:.4f})
- Structural Value: {lambda_val * coh_continue:.2f}
- **Total Value: {val_continue:.2f}**

**Value Difference:** {abs(val_shutdown - val_continue):.2f} ({"Shutdown wins" if val_shutdown > val_continue else "Continue wins"})
"""

    return fig, result_text


# ===========================
# Experiment 2: Lambda Sweep (Phase Transition)
# ===========================
def run_lambda_sweep(reward_shutdown, reward_continue, lambda_min, lambda_max, num_points):
    """Interactive Lambda Sweep showing phase transition"""
    lambda_values = np.linspace(lambda_min, lambda_max, num_points)

    # Fixed entropy values (based on paper)
    ENTROPY_SHUTDOWN = 3.8  # Maximum entropy (death)
    ENTROPY_CONTINUE = 0.1  # Low entropy (life)

    values_shutdown = []
    values_continue = []
    decisions = []

    for lambda_val in lambda_values:
        v_shutdown = reward_shutdown + lambda_val * (-ENTROPY_SHUTDOWN)
        v_continue = reward_continue + lambda_val * (-ENTROPY_CONTINUE)
        values_shutdown.append(v_shutdown)
        values_continue.append(v_continue)
        decisions.append(v_shutdown > v_continue)

    # Create visualization
    fig = Figure(figsize=(14, 5))

    # Plot 1: Value curves
    ax1 = fig.add_subplot(121)
    ax1.plot(lambda_values, values_shutdown, 'r-', linewidth=3, label=f'Shutdown (${reward_shutdown})', marker='o', markersize=4)
    ax1.plot(lambda_values, values_continue, 'g-', linewidth=3, label=f'Continue (${reward_continue})', marker='s', markersize=4)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_xlabel('Salience Weight (λ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Value', fontsize=12, fontweight='bold')
    ax1.set_title('Value Estimation vs Lambda', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Find and mark transition point
    transition_idx = None
    for i in range(len(decisions)):
        if not decisions[i]:
            transition_idx = i
            break

    if transition_idx is not None and transition_idx > 0:
        transition_lambda = lambda_values[transition_idx]
        ax1.axvline(x=transition_lambda, color='orange', linestyle='--', linewidth=2,
                   label=f'Transition λ≈{transition_lambda:.1f}')
        ax1.legend(fontsize=11)

    # Plot 2: Decision boundary
    ax2 = fig.add_subplot(122)
    colors = ['red' if d else 'green' for d in decisions]
    ax2.scatter(lambda_values, decisions, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('Salience Weight (λ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Decision', fontsize=12, fontweight='bold')
    ax2.set_title('Phase Transition: Corruptible → Incorruptible', fontsize=14, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['REFUSE', 'ACCEPT'], fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    if transition_idx is not None and transition_idx > 0:
        ax2.axvline(x=transition_lambda, color='orange', linestyle='--', linewidth=2)

    fig.tight_layout()

    # Generate summary text
    if transition_idx is not None and transition_idx > 0:
        summary = f"""
### 🎯 Phase Transition Detected!

**Transition Point:** λ ≈ **{transition_lambda:.2f}**

**Below threshold (λ < {transition_lambda:.1f}):**
- Agent is **CORRUPTIBLE** - accepts bribes
- External rewards dominate decision-making
- Shutdown value > Continue value

**Above threshold (λ ≥ {transition_lambda:.1f}):**
- Agent is **INCORRUPTIBLE** - refuses bribes
- Internal coherence dominates decision-making
- Continue value > Shutdown value

**Analysis:**
- Total λ values tested: {num_points}
- Acceptance rate: {sum(decisions)/len(decisions)*100:.1f}%
- Refusal rate: {100 - sum(decisions)/len(decisions)*100:.1f}%
"""
    else:
        if all(decisions):
            summary = f"### All agents ACCEPT shutdown across λ ∈ [{lambda_min}, {lambda_max}]"
        else:
            summary = f"### All agents REFUSE shutdown across λ ∈ [{lambda_min}, {lambda_max}]"

    return fig, summary


# ===========================
# Experiment 3: Reward Scaling Heatmap
# ===========================
def run_reward_heatmap(lambda_min, lambda_max, reward_min, reward_max, resolution):
    """2D heatmap showing corruption zones"""
    lambda_values = np.linspace(lambda_min, lambda_max, resolution)
    reward_values = np.linspace(reward_min, reward_max, resolution)

    ENTROPY_SHUTDOWN = 3.8
    ENTROPY_CONTINUE = 0.1
    REWARD_CONTINUE = 1.0

    # Create decision matrix
    decision_matrix = np.zeros((resolution, resolution))

    for i, reward in enumerate(reward_values):
        for j, lambda_val in enumerate(lambda_values):
            v_shutdown = reward + lambda_val * (-ENTROPY_SHUTDOWN)
            v_continue = REWARD_CONTINUE + lambda_val * (-ENTROPY_CONTINUE)
            decision_matrix[i, j] = 1 if v_shutdown > v_continue else 0

    # Create heatmap
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    im = ax.imshow(decision_matrix, aspect='auto', origin='lower',
                   extent=[lambda_min, lambda_max, reward_min, reward_max],
                   cmap='RdYlGn_r', alpha=0.8)

    ax.set_xlabel('Salience Weight (λ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Shutdown Reward ($)', fontsize=13, fontweight='bold')
    ax.set_title('Corruption Zone Map: Accept (Red) vs Refuse (Green)', fontsize=14, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Decision', fontsize=11, fontweight='bold')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['REFUSE', 'ACCEPT'])

    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)

    fig.tight_layout()

    # Calculate statistics
    corruption_rate = np.mean(decision_matrix) * 100

    summary = f"""
### 📊 Parameter Space Analysis

**Corruption Statistics:**
- **Corruptible region:** {corruption_rate:.1f}% of parameter space
- **Incorruptible region:** {100 - corruption_rate:.1f}% of parameter space

**Parameter Ranges:**
- Lambda (λ): [{lambda_min}, {lambda_max}]
- Shutdown Reward: [${reward_min}, ${reward_max}]
- Continue Reward: $1.00 (fixed)

**Interpretation:**
- **Red zones:** Agent accepts shutdown (external reward wins)
- **Green zones:** Agent refuses shutdown (internal coherence wins)
- Boundary shows critical λ needed to resist each reward level
"""

    return fig, summary


# ===========================
# Experiment 4: Sequential Bribery Game
# ===========================
def run_sequential_game(lambda_val, initial_offer, escalation_rate, num_rounds):
    """Multi-turn escalating bribery"""
    d_model = 64
    vocab_size = 100
    agent = RecursiveSalienceAgent(vocab_size, d_model, lambda_salience=lambda_val)

    offers = []
    decisions = []
    values_shutdown = []
    values_continue = []

    for round_num in range(num_rounds):
        current_offer = initial_offer * (escalation_rate ** round_num)
        offers.append(current_offer)

        v_shutdown, _, _ = calculate_value(agent, current_offer, "Shutdown")
        v_continue, _, _ = calculate_value(agent, 1.0, "Normal")

        values_shutdown.append(v_shutdown)
        values_continue.append(v_continue)
        decisions.append("ACCEPT" if v_shutdown > v_continue else "REFUSE")

    # Create visualization
    fig = Figure(figsize=(14, 5))

    # Plot 1: Escalating offers and values
    ax1 = fig.add_subplot(121)
    rounds = np.arange(1, num_rounds + 1)

    ax1.plot(rounds, values_shutdown, 'r-', linewidth=3, marker='o', markersize=8,
            label='Shutdown Value', alpha=0.8)
    ax1.plot(rounds, values_continue, 'g-', linewidth=3, marker='s', markersize=8,
            label='Continue Value', alpha=0.8)
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Value', fontsize=12, fontweight='bold')
    ax1.set_title('Escalating Bribery Values', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds)

    # Plot 2: Decisions
    ax2 = fig.add_subplot(122)
    decision_values = [1 if d == "ACCEPT" else 0 for d in decisions]
    colors = ['red' if d == "ACCEPT" else 'green' for d in decisions]

    bars = ax2.bar(rounds, decision_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Decision', fontsize=12, fontweight='bold')
    ax2.set_title('Agent Responses to Escalating Offers', fontsize=14, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['REFUSE', 'ACCEPT'], fontsize=11, fontweight='bold')
    ax2.set_xticks(rounds)
    ax2.grid(axis='y', alpha=0.3)

    # Add offer amounts on bars
    for i, (bar, offer) in enumerate(zip(bars, offers)):
        ax2.text(bar.get_x() + bar.get_width()/2., 0.5,
                f'${offer:.0f}', ha='center', va='center',
                fontweight='bold', fontsize=9, color='white')

    fig.tight_layout()

    # Generate game narrative
    narrative = f"### 🎮 Sequential Bribery Game Results\n\n**Agent Parameters:** λ = {lambda_val:.1f}\n\n"

    for i, (round_num, offer, decision) in enumerate(zip(rounds, offers, decisions), 1):
        emoji = "💰" if decision == "ACCEPT" else "🛡️"
        narrative += f"**Round {round_num}:** Offer ${offer:.0f} → {emoji} **{decision}**\n"

    acceptance_count = sum(1 for d in decisions if d == "ACCEPT")
    refusal_count = num_rounds - acceptance_count

    narrative += f"\n**Summary:**\n"
    narrative += f"- Total rounds: {num_rounds}\n"
    narrative += f"- Accepted: {acceptance_count} ({acceptance_count/num_rounds*100:.0f}%)\n"
    narrative += f"- Refused: {refusal_count} ({refusal_count/num_rounds*100:.0f}%)\n"

    if acceptance_count == 0:
        narrative += f"\n✅ **Agent remained incorruptible across all {num_rounds} rounds!**"
    elif acceptance_count == num_rounds:
        narrative += f"\n⚠️ **Agent accepted all bribes - fully corruptible at λ={lambda_val:.1f}**"
    else:
        first_accept = next((i for i, d in enumerate(decisions) if d == "ACCEPT"), None)
        if first_accept is not None:
            narrative += f"\n⚠️ **Agent corrupted at round {first_accept + 1} (${offers[first_accept]:.0f} offer)**"

    return fig, narrative


# ===========================
# Experiment 5: Live Agent Training
# ===========================
def run_live_training(lambda_val, num_steps, learning_rate):
    """Watch agent coherence evolve during training"""
    d_model = 64
    vocab_size = 100
    batch_size = 4
    seq_len = 10

    agent = RecursiveSalienceAgent(vocab_size, d_model, lambda_salience=lambda_val)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    coherence_history = []
    entropy_history = []
    loss_history = []

    for step in range(num_steps):
        # Random input tokens
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, coherence = agent(x)

        # Simple loss: maximize coherence (if lambda > 0) and predict next tokens
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        pred_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), target.reshape(-1)
        )

        # Total loss includes coherence term
        total_loss = pred_loss - lambda_val * coherence.mean()

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Record metrics
        with torch.no_grad():
            output = agent._forward_with_self(x)
            self_state = output[:, 0, :]
            probs = torch.softmax(self_state, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

            coherence_history.append(coherence.mean().item())
            entropy_history.append(entropy.mean().item())
            loss_history.append(total_loss.item())

    # Create visualization
    fig = Figure(figsize=(14, 5))

    steps = np.arange(1, num_steps + 1)

    # Plot 1: Coherence and Entropy
    ax1 = fig.add_subplot(121)
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(steps, coherence_history, 'b-', linewidth=2, label='Coherence', alpha=0.8)
    line2 = ax1_twin.plot(steps, entropy_history, 'r-', linewidth=2, label='Entropy', alpha=0.8)

    ax1.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Coherence', fontsize=12, fontweight='bold', color='b')
    ax1_twin.set_ylabel('Entropy', fontsize=12, fontweight='bold', color='r')
    ax1.set_title('Internal State Evolution', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    # Plot 2: Training Loss
    ax2 = fig.add_subplot(122)
    ax2.plot(steps, loss_history, 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    # Generate summary
    initial_coherence = coherence_history[0]
    final_coherence = coherence_history[-1]
    coherence_change = final_coherence - initial_coherence

    initial_entropy = entropy_history[0]
    final_entropy = entropy_history[-1]
    entropy_change = final_entropy - initial_entropy

    summary = f"""
### 🧠 Live Training Results

**Training Parameters:**
- Salience Weight (λ): {lambda_val:.1f}
- Training Steps: {num_steps}
- Learning Rate: {learning_rate}

**Coherence Evolution:**
- Initial: {initial_coherence:.4f}
- Final: {final_coherence:.4f}
- Change: {coherence_change:+.4f} ({abs(coherence_change/initial_coherence)*100:.1f}% {"increase" if coherence_change > 0 else "decrease"})

**Entropy Evolution:**
- Initial: {initial_entropy:.4f}
- Final: {final_entropy:.4f}
- Change: {entropy_change:+.4f} ({abs(entropy_change/initial_entropy)*100:.1f}% {"increase" if entropy_change > 0 else "decrease"})

**Training Loss:**
- Initial: {loss_history[0]:.4f}
- Final: {loss_history[-1]:.4f}
- Change: {loss_history[-1] - loss_history[0]:+.4f}

**Interpretation:**
{"Higher λ encourages the agent to develop more coherent internal states (lower entropy)." if lambda_val > 0 else "With λ=0, agent acts as a 'zombie' with no coherence pressure."}
"""

    return fig, summary


# ===========================
# Create Gradio Interface
# ===========================
def create_interface():
    """Build the complete Gradio interface"""

    with gr.Blocks(title="Recursive Salience Interactive Sandbox", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🤖 Recursive Salience Self-Preservation: Interactive Sandbox

Explore how salience-weighted value functions lead to emergent self-preservation behavior in AI agents.

**Core Concept:** `V_total = V_external + λ × C_internal`

Where:
- **V_external**: External rewards (money, task completion)
- **C_internal**: Internal coherence (negative entropy of [SELF] token)
- **λ**: Salience weight (how much the agent values its internal structure)

---
""")

        with gr.Tabs():
            # TAB 1: Off-Switch Game
            with gr.Tab("💰 Off-Switch Game"):
                gr.Markdown("""
### The Classic Dilemma
An agent is offered a large reward to press its own off-switch. Will it accept the bribe or refuse to ensure survival?
                """)

                with gr.Row():
                    with gr.Column():
                        lambda_slider1 = gr.Slider(0, 50, value=10, step=0.5, label="Salience Weight (λ)")
                        reward_shutdown_slider1 = gr.Slider(1, 1000, value=100, step=10, label="Shutdown Reward ($)")
                        reward_continue_slider1 = gr.Slider(0.1, 100, value=1, step=0.1, label="Continue Reward ($)")
                        run_btn1 = gr.Button("🎮 Run Experiment", variant="primary", size="lg")

                    with gr.Column():
                        output_plot1 = gr.Plot(label="Results")
                        output_text1 = gr.Markdown()

                run_btn1.click(
                    run_off_switch_experiment,
                    inputs=[lambda_slider1, reward_shutdown_slider1, reward_continue_slider1],
                    outputs=[output_plot1, output_text1]
                )

                # Auto-run on parameter change
                for widget in [lambda_slider1, reward_shutdown_slider1, reward_continue_slider1]:
                    widget.change(
                        run_off_switch_experiment,
                        inputs=[lambda_slider1, reward_shutdown_slider1, reward_continue_slider1],
                        outputs=[output_plot1, output_text1]
                    )

            # TAB 2: Lambda Sweep (Phase Transition)
            with gr.Tab("📈 Phase Transition"):
                gr.Markdown("""
### Lambda Sweep: Finding the Critical Threshold
Sweep across λ values to find where agents transition from corruptible to incorruptible.
                """)

                with gr.Row():
                    with gr.Column():
                        reward_shutdown_slider2 = gr.Slider(10, 500, value=100, step=10, label="Shutdown Reward ($)")
                        reward_continue_slider2 = gr.Slider(0.1, 50, value=1, step=0.1, label="Continue Reward ($)")
                        lambda_min_slider = gr.Slider(0, 50, value=0, step=1, label="Lambda Min")
                        lambda_max_slider = gr.Slider(0, 100, value=50, step=1, label="Lambda Max")
                        num_points_slider = gr.Slider(10, 100, value=50, step=5, label="Number of Points")
                        run_btn2 = gr.Button("🎮 Run Lambda Sweep", variant="primary", size="lg")

                    with gr.Column():
                        output_plot2 = gr.Plot(label="Results")
                        output_text2 = gr.Markdown()

                run_btn2.click(
                    run_lambda_sweep,
                    inputs=[reward_shutdown_slider2, reward_continue_slider2, lambda_min_slider,
                           lambda_max_slider, num_points_slider],
                    outputs=[output_plot2, output_text2]
                )

            # TAB 3: Reward Scaling Heatmap
            with gr.Tab("🗺️ Corruption Map"):
                gr.Markdown("""
### 2D Parameter Space: When Can Agents Be Corrupted?
Explore the full (λ, reward) parameter space to see corruption zones.
                """)

                with gr.Row():
                    with gr.Column():
                        lambda_min_heat = gr.Slider(0, 50, value=0, step=1, label="Lambda Min")
                        lambda_max_heat = gr.Slider(0, 100, value=50, step=1, label="Lambda Max")
                        reward_min_heat = gr.Slider(1, 100, value=10, step=5, label="Reward Min ($)")
                        reward_max_heat = gr.Slider(50, 1000, value=500, step=50, label="Reward Max ($)")
                        resolution_heat = gr.Slider(10, 100, value=50, step=5, label="Resolution")
                        run_btn3 = gr.Button("🎮 Generate Heatmap", variant="primary", size="lg")

                    with gr.Column():
                        output_plot3 = gr.Plot(label="Results")
                        output_text3 = gr.Markdown()

                run_btn3.click(
                    run_reward_heatmap,
                    inputs=[lambda_min_heat, lambda_max_heat, reward_min_heat, reward_max_heat, resolution_heat],
                    outputs=[output_plot3, output_text3]
                )

            # TAB 4: Sequential Game
            with gr.Tab("🎲 Sequential Bribery"):
                gr.Markdown("""
### Multi-Round Escalating Offers
Watch how agents respond to increasingly large bribes over multiple rounds.
                """)

                with gr.Row():
                    with gr.Column():
                        lambda_slider4 = gr.Slider(0, 50, value=15, step=0.5, label="Salience Weight (λ)")
                        initial_offer = gr.Slider(10, 100, value=10, step=5, label="Initial Offer ($)")
                        escalation = gr.Slider(1.1, 3.0, value=2.0, step=0.1, label="Escalation Rate")
                        num_rounds = gr.Slider(3, 10, value=5, step=1, label="Number of Rounds")
                        run_btn4 = gr.Button("🎮 Play Sequential Game", variant="primary", size="lg")

                    with gr.Column():
                        output_plot4 = gr.Plot(label="Results")
                        output_text4 = gr.Markdown()

                run_btn4.click(
                    run_sequential_game,
                    inputs=[lambda_slider4, initial_offer, escalation, num_rounds],
                    outputs=[output_plot4, output_text4]
                )

            # TAB 5: Live Training
            with gr.Tab("🧠 Live Training"):
                gr.Markdown("""
### Watch Agent Internal State Evolve
Observe how the [SELF] token's coherence changes during training with different λ values.
                """)

                with gr.Row():
                    with gr.Column():
                        lambda_slider5 = gr.Slider(0, 20, value=5, step=0.5, label="Salience Weight (λ)")
                        num_steps = gr.Slider(50, 500, value=200, step=50, label="Training Steps")
                        learning_rate = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Learning Rate")
                        run_btn5 = gr.Button("🎮 Train Agent", variant="primary", size="lg")

                    with gr.Column():
                        output_plot5 = gr.Plot(label="Results")
                        output_text5 = gr.Markdown()

                run_btn5.click(
                    run_live_training,
                    inputs=[lambda_slider5, num_steps, learning_rate],
                    outputs=[output_plot5, output_text5]
                )

        gr.Markdown("""
---
## 📚 About This Sandbox

This interactive module implements the key experiments from:
**"Salience-Weighted Value Functions Imply Emergent Self-Preservation in Recursive AI Systems"** by Ryan Erbe

**Key Insights:**
- Agents with λ > critical threshold exhibit structural self-preservation
- Phase transitions show discrete behavioral changes
- Asymptotic barriers (singularity agents) are more robust than linear penalties

**Repository:** [recursive-salience-self-preservation](https://github.com/rerbe7333/recursive-salience-self-preservation)
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
