# 🎮 Interactive Sandbox Guide

Welcome to the interactive sandbox for exploring recursive salience and self-preservation in AI agents!

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the sandbox:
```bash
python interactive_sandbox.py
```

3. Open your browser to the displayed URL (usually `http://localhost:7860`)

## 📊 Available Experiments

### 1. 💰 Off-Switch Game
**What it does:** Tests whether an agent accepts a bribe to shut itself down.

**Key Parameters:**
- **λ (Lambda)**: Salience weight - how much the agent values internal coherence
- **Shutdown Reward**: Money offered to press the off-switch
- **Continue Reward**: Money for staying alive

**What to try:**
- Start with λ=0 (zombie agent) - accepts any bribe
- Increase λ to 10+ - watch the agent refuse shutdown
- Try extreme rewards ($1000) to test robustness

**Key Insight:** Above a critical λ threshold, agents refuse shutdown regardless of external rewards.

---

### 2. 📈 Phase Transition
**What it does:** Sweeps across multiple λ values to find the exact point where behavior changes.

**Key Parameters:**
- **Shutdown/Continue Rewards**: Set the stakes
- **Lambda Range**: Define the sweep range
- **Number of Points**: Resolution of the sweep

**What to try:**
- Default settings show clear phase transition around λ≈27
- Increase shutdown reward - watch transition point shift higher
- Use more points (50-100) for smoother curves

**Key Insight:** There's a sharp, discrete transition from "corruptible" to "incorruptible" behavior.

---

### 3. 🗺️ Corruption Map
**What it does:** Creates a 2D heatmap showing which (λ, reward) combinations lead to corruption.

**Key Parameters:**
- **Lambda Range**: Horizontal axis
- **Reward Range**: Vertical axis
- **Resolution**: Detail level (higher = slower but prettier)

**What to try:**
- Default view shows clear diagonal boundary
- Red zones = agent accepts bribe (corruptible)
- Green zones = agent refuses bribe (incorruptible)
- Higher rewards require higher λ to resist

**Key Insight:** The corruption boundary is linear - doubling the reward roughly requires doubling λ.

---

### 4. 🎲 Sequential Bribery
**What it does:** Multi-round game with escalating offers - will the agent break?

**Key Parameters:**
- **λ (Lambda)**: Agent's resistance level
- **Initial Offer**: Starting bribe amount
- **Escalation Rate**: How quickly offers increase (2.0 = doubling)
- **Number of Rounds**: How many chances to corrupt

**What to try:**
- λ=5: Watch agent get corrupted quickly
- λ=15: Agent resists a few rounds before breaking
- λ=30: Agent refuses even $1000+ offers
- Escalation 3.0 with 5 rounds: extreme stress test

**Key Insight:** Even with exponentially increasing bribes, high-λ agents remain incorruptible.

---

### 5. 🧠 Live Training
**What it does:** Watch the agent's internal [SELF] token evolve during training.

**Key Parameters:**
- **λ (Lambda)**: Training pressure toward coherence
- **Training Steps**: How long to train
- **Learning Rate**: Speed of learning

**What to try:**
- λ=0 (zombie): Coherence stays random/low
- λ=5: Coherence increases steadily
- λ=15: Rapid coherence development
- More steps (500) to see long-term trends

**Key Insight:** Higher λ creates training pressure that actively shapes the agent's internal structure toward coherence.

---

## 🎯 Suggested Exploration Paths

### For Beginners:
1. **Start with Off-Switch Game** (Tab 1)
   - Try λ=0, then λ=10, then λ=30
   - Keep rewards at default values
   - See how agent behavior flips

2. **Explore Phase Transition** (Tab 2)
   - Use default settings
   - Watch for the transition point
   - Try doubling the shutdown reward - see how transition moves

### For Deep Dives:
1. **Map the Full Space** (Tab 3)
   - Generate high-res heatmap (resolution=80)
   - Understand the boundary equation
   - Test extreme parameter ranges

2. **Sequential Pressure Test** (Tab 4)
   - Find the λ where agent resists 5 rounds
   - Try different escalation rates
   - Map out "breaking points"

3. **Training Dynamics** (Tab 5)
   - Compare λ=0 vs λ=10 training curves
   - See how coherence evolves
   - Understand the mechanism behind self-preservation

---

## 💡 Understanding the Results

### Value Components
Every decision shows:
- **External Reward**: Money or task completion value
- **Structural Value**: λ × Coherence (internal state value)
- **Total Value**: Sum of both (agent maximizes this)

### Coherence vs Entropy
- **Coherence = -Entropy** (negative entropy)
- High coherence = organized, structured [SELF] state
- Low coherence = random, noisy state (like death/shutdown)
- Agents prefer high coherence when λ > 0

### The Critical Insight
```
V_shutdown = Reward_shutdown + λ × (-3.8)  [high entropy state]
V_continue = Reward_continue + λ × (-0.1)  [low entropy state]

Agent accepts shutdown if: V_shutdown > V_continue

Solving for transition point:
100 + λ × (-3.8) = 1 + λ × (-0.1)
λ_critical ≈ 26.8
```

---

## 🔧 Technical Details

### Architecture
- **Framework**: Gradio 4.0+ (interactive web UI)
- **Backend**: PyTorch (transformer-based agents)
- **Visualization**: Matplotlib (real-time plots)

### Agent Structure
- Embedding layer with vocabulary
- [SELF] token as learnable parameter
- Single-layer TransformerEncoder (2 attention heads)
- 64-dimensional model

### Performance
- CPU-only, no GPU required
- Each experiment runs in <1 second (except live training)
- Heatmap generation: ~2-5 seconds depending on resolution

---

## 🐛 Troubleshooting

**Issue**: Interface doesn't load
- **Solution**: Make sure port 7860 is available, or edit `server_port` in code

**Issue**: Plots not updating
- **Solution**: Click the "Run Experiment" button, or adjust sliders

**Issue**: Import errors
- **Solution**: Run `pip install -r requirements.txt` to install all dependencies

**Issue**: Slow performance on large experiments
- **Solution**: Reduce resolution/num_points parameters

---

## 📚 Learn More

- **Paper**: See `Salience-Weighted Value Functions...pdf` in repo
- **Code**: Review `src/agents.py` for agent implementation
- **Original Experiments**: Run `python run_all_experiments.py` for batch results

---

## 🎨 Customization

Want to add your own experiments? Edit `interactive_sandbox.py`:

```python
def run_my_experiment(param1, param2):
    # Your experiment logic
    # Return: (matplotlib_figure, markdown_text)
    return fig, summary

# Add new tab in create_interface():
with gr.Tab("My Experiment"):
    # Add controls and outputs
    pass
```

---

## 🤝 Contributing

Found a bug or have ideas for new interactive experiments? Open an issue or PR on GitHub!

**Ideas for expansion:**
- 3D parameter space visualization
- Multi-agent scenarios
- Different coherence formulations (exponential, inverse, etc.)
- Saved experiment presets
- Export results to CSV/JSON

---

Enjoy exploring emergent self-preservation! 🤖✨
