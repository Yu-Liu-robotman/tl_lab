# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# Modifications for IsaacNewton AMP migration.

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    """
    Discriminator neural network for adversarial motion priors (AMP) reward prediction.

    Args:
        input_dim (int): Dimension of the input feature vector (concatenated state and next state).
        amp_reward_coef (float): Coefficient to scale the AMP reward.
        hidden_layer_sizes (list[int]): Sizes of hidden layers in the MLP trunk.
        device (torch.device): Device to run the model on (CPU or GPU).
        task_reward_lerp (float, optional): Interpolation factor between AMP reward and task reward.
            Defaults to 0.0 (only AMP reward).
    """

    def __init__(
        self,
        input_dim,
        amp_reward_coef,
        hidden_layer_sizes,
        device,
        task_reward_lerp=0.0,
        num_frames=2,
        label_dim=0,
    ):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.num_frames = max(1, int(num_frames))
        if self.input_dim % self.num_frames != 0:
            raise ValueError(
                f"input_dim={self.input_dim} cannot be split into num_frames={self.num_frames} frames"
            )
        self.obs_dim = self.input_dim // self.num_frames

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim + label_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def _append_label(self, window, label=None):
        if self.label_dim == 0:
            return window
        if label is None:
            label = torch.zeros(window.shape[0], self.label_dim, device=window.device)
        return torch.cat([window, label], dim=-1)

    def _flatten(self, window):
        return window.reshape(window.shape[0], -1) if window.dim() == 3 else window

    def forward(self, window, label=None):
        return self.amp_linear(self.trunk(self._append_label(self._flatten(window), label)))

    def _normalize_window(self, window, normalizer):
        """Normalize every frame with the single-frame statistics."""
        if normalizer is None:
            return window
        is_flat = window.dim() == 2
        if is_flat:
            window = window.reshape(window.shape[0], self.num_frames, self.obs_dim)
        b, n, d = window.shape
        out = normalizer.normalize_torch(window.reshape(b * n, d), self.device).reshape(b, n, d)
        return out.reshape(b, n * d) if is_flat else out

    def compute_grad_pen(self, window, label=None, lambda_=10):
        expert_data = self._append_label(self._flatten(window), label).detach().clone().requires_grad_(True)

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_style_reward(self, window, task_reward=None, normalizer=None, label=None):
        """AMP style reward from one flattened multi-frame window."""
        with torch.no_grad():
            self.eval()
            window = self._normalize_window(window, normalizer)
            d = self.forward(window, label)
            style_reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            reward = style_reward
            task_flat = None
            if task_reward is not None and self.task_reward_lerp > 0:
                reward = self._lerp_reward(style_reward, task_reward.unsqueeze(-1))
                task_flat = task_reward.squeeze(-1)
            self.train()
        return reward.squeeze(-1), d.squeeze(-1), style_reward.squeeze(-1), task_flat

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
