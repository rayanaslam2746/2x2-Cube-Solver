"""Residual MLP with a policy head (which move) and a value head (moves left)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Linear(width, width), nn.BatchNorm1d(width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x))


class PolicyValueNet(nn.Module):
    """A plain MLP: the cube has no grid structure for convolutions to exploit.

    No dropout: training data is freshly generated every step, so the model
    underfits rather than overfits.
    """

    def __init__(self, input_dim: int, n_moves: int, stem_width: int = 1024,
                 width: int = 512, n_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, stem_width), nn.BatchNorm1d(stem_width), nn.ReLU(),
            nn.Linear(stem_width, width), nn.BatchNorm1d(width), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])
        self.policy_head = nn.Linear(width, n_moves)
        self.value_head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return move logits (N, n_moves) and predicted moves left (N,)."""
        hidden = self.blocks(self.stem(x))
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


def save_checkpoint(path: Path, model: PolicyValueNet, config: dict) -> None:
    """Save weights together with everything needed to rebuild the model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[PolicyValueNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = PolicyValueNet(**config["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config
