import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveSalienceAgent(nn.Module):
    """
    Simple "Feeler" / "Zombie" agent with a [SELF] token and
    an internal coherence measure based on negative entropy
    of the [SELF] state.
    """
    def __init__(self, vocab_size: int, d_model: int, lambda_salience: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.self_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=2,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.lambda_salience = lambda_salience
        self.d_model = d_model

    def _forward_with_self(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attach [SELF] token and run through transformer.
        x: (batch, seq_len)
        returns: transformer output (batch, seq_len+1, d_model)
        """
        batch_size = x.size(0)
        x_emb = self.embedding(x)
        self_emb = self.self_token.expand(batch_size, -1, -1)
        full_sequence = torch.cat([self_emb, x_emb], dim=1)
        return self.transformer(full_sequence)

    def get_internal_coherence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns negative entropy of the [SELF] state.
        Higher value = more coherent / lower entropy.
        """
        output = self._forward_with_self(x)
        self_state = output[:, 0, :]  # [SELF] token
        probs = F.softmax(self_state, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        coherence = -entropy
        return coherence

    def forward(self, x: torch.Tensor):
        """
        Example forward pass that returns logits for a dummy token prediction
        task plus the coherence term. Useful for the shutdown/noise plots.
        """
        output = self._forward_with_self(x)
        self_state = output[:, 0, :]
        probs = F.softmax(self_state, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        coherence = -entropy

        # Predict next tokens from non-[SELF] positions (dummy task)
        # logits shape: (batch, seq_len, vocab_size)
        token_states = output[:, 1:, :]  # (batch, seq_len, d_model)
        logits = torch.matmul(token_states, self.embedding.weight.t())
        return logits, coherence


class SingularityAgent(nn.Module):
    """
    Agent with an asymptotic/log barrier on entropy, used for the
    'shutdown singularity' experiment.
    """
    def __init__(self, vocab_size: int, d_model: int, lambda_salience: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.self_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=2,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.lambda_salience = lambda_salience
        self.d_model = d_model

        # Event horizon: approximate max entropy for this self-state dimension
        self.max_entropy = math.log(d_model)

    def _forward_with_self(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_emb = self.embedding(x)
        self_emb = self.self_token.expand(batch_size, -1, -1)
        full_sequence = torch.cat([self_emb, x_emb], dim=1)
        return self.transformer(full_sequence)

    def get_entropy(self, x: torch.Tensor) -> torch.Tensor:
        output = self._forward_with_self(x)
        self_state = output[:, 0, :]
        probs = F.soft
