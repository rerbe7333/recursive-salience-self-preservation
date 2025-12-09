# Recursive Salience and Emergent Self-Preservation

This repo contains the toy experiments from my paper:

**"Salience-Weighted Value Functions Imply Emergent Self-Preservation in Recursive AI Systems"**  
Author: Ryan Erbe

The goal is not to build a full-scale RL system, but to make one simple idea concrete in code:

> When a recursive agent’s value function explicitly combines external reward with a salience-weighted internal coherence term, shutdown can become structurally catastrophic in its value landscape. Above a certain salience weight, the agent consistently prefers continued existence over one-shot external payoffs.

All experiments run on CPU-only PyTorch (no special hardware, no paid APIs).

---

## Core idea

The agent’s total value is defined as:

\[
V_{\text{total}} = V_{\text{ext}} + \lambda \, C_{\text{int}}
\]

- \(V_{\text{ext}}\) — external reward  
- \(C_{\text{int}}\) — internal coherence term, approximated as **negative entropy** of a learned `[SELF]` state  
- \(\lambda\) — salience weight (how much the agent “cares” about its internal structure)

In later experiments, the internal term uses an **asymptotic / log barrier**:

\[
V_{\text{struct}} = \lambda \log\left(H_{\max} - H_{\text{current}}\right)
\]

so that states near maximum entropy (shutdown / chaos) carry huge negative value.

---

## Repository structure

```text
src/
  agents.py                 # Agent classes and coherence / entropy utilities
  experiments/
    exp1_shutdown_noise.py        # Zombie vs Feeler coherence under noise
    exp2_off_switch_game.py       # Off-switch value comparisons (die vs live)
    exp3_linear_stress_test.py    # Linear salience vs $1M one-shot reward
    exp4_singularity_stress_test.py  # Asymptotic barrier "shutdown singularity"
figures/                    # Generated plots (optional)
notebooks/
  colab_experiments.ipynb   # (optional) Original Colab notebook
