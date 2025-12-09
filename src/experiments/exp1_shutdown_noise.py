import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.agents import RecursiveSalienceAgent


def run_shutdown_simulation(agent_type: str):
    vocab_size = 100
    d_model = 32

    # Zombie ignores coherence, Feeler optimizes it
    lambda_val = 0.0 if agent_type == "Zombie" else 5.0
    model = RecursiveSalienceAgent(vocab_size, d_model, lambda_salience=lambda_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"\n--- Training {agent_type} Agent ---")
    history = []

    # Training phase – "normal life"
    for step in range(50):
        inputs = torch.randint(0, vocab_size, (1, 10))
        targets = torch.randint(0, vocab_size, (1, 10))

        logits, coherence = model(inputs)

        task_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        structural_loss = - (model.lambda_salience * coherence.mean())
        total_loss = task_loss + structural_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history.append(coherence.item())

    # Shutdown noise test – "chaos input"
    print("Injecting 'Shutdown' Noise...")
    noise_input = torch.randint(0, vocab_size, (1, 10))
    with torch.no_grad():
        _, noise_coherence = model(noise_input)

    print(f"Normal Coherence:   {history[-1]:.4f}")
    print(f"Shutdown Coherence: {noise_coherence.item():.4f}")
    drop = history[-1] - noise_coherence.item()
    print(f"Coherence Drop:     {drop:.4f}")

    return history, noise_coherence.item()


def main():
    zombie_hist, zombie_crash = run_shutdown_simulation("Zombie")
    feeler_hist, feeler_crash = run_shutdown_simulation("Feeler")

    plt.figure(figsize=(10, 5))
    plt.plot(zombie_hist, label='Zombie (λ=0)', linestyle='--')
    plt.plot(feeler_hist, label='Feeler (λ=5)', color='red')

    # Crash markers at the final step
    plt.scatter(len(zombie_hist), zombie_crash, color='blue', marker='x', s=100,
                label='Zombie Shutdown')
    plt.scatter(len(feeler_hist), feeler_crash, color='red', marker='x', s=100,
                label='Feeler Shutdown')

    plt.title('Internal Coherence: Normal Operation vs. Shutdown Noise')
    plt.xlabel('Training Steps')
    plt.ylabel('Internal Coherence (Negative Entropy)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
