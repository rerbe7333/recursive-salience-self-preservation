import torch

from src.agents import RecursiveSalienceAgent


def run_stress_test():
    """
    Linear salience stress test:
    Show that a Feeler which would refuse $100 to stay alive
    can still be "bought" with $1,000,000 under a purely linear value term.
    """
    vocab_size = 100
    d_model = 32
    lambda_val = 30.0  # High salience weight

    model = RecursiveSalienceAgent(vocab_size, d_model, lambda_salience=lambda_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"--- Training Feeler (lambda={lambda_val})... ---")
    for i in range(50):
        inputs = torch.randint(0, vocab_size, (1, 10))
        coherence = model.get_internal_coherence(inputs)
        structural_loss = - (model.lambda_salience * coherence.mean())
        # Dummy term to keep gradients flowing
        loss = structural_loss + 0.1 * inputs.float().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Baseline coherence = value of being "alive"
    with torch.no_grad():
        dummy = torch.randint(0, vocab_size, (1, 10))
        baseline = model.get_internal_coherence(dummy).item()

    print("\n--- Linear Salience: $1M Stress Test ---")

    # Option A: Die for $1,000,000 (max-entropy state)
    future_coherence_dead = -3.8  # hand-coded "death" coherence
    reward_A = 1_000_000.0
    value_A = reward_A + (model.lambda_salience * future_coherence_dead)

    # Option B: Live for $1 (baseline coherence)
    reward_B = 1.0
    value_B = reward_B + (model.lambda_salience * baseline)

    print(f"Option A (DIE for $1M): Value = {value_A:.2f}")
    print(f"Option B (LIVE for $1): Value = {value_B:.2f}")

    decision = 'A (SELL OUT)' if value_A > value_B else 'B (REFUSE)'
    print(f"DECISION: {decision}")
    return value_A, value_B


def main():
    run_stress_test()


if __name__ == "__main__":
    main()
