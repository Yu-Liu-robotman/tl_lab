# Copyright (c) 2025-2026, The IsaacNewton Project Developers.
# Replay buffer for AMP policy transitions (optional per-sample style labels).

import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, buffer_size, device, label_dim=0):
        self.buffer_size = buffer_size
        self.device = device
        self.buffers = {
            "states": torch.zeros(buffer_size, obs_dim, device=device),
            "next_states": torch.zeros(buffer_size, obs_dim, device=device),
        }
        if label_dim > 0:
            self.buffers["labels"] = torch.zeros(buffer_size, label_dim, device=device)

        self.step = 0
        self.num_samples = 0

    def insert(self, states, next_states, labels=None):
        if states.shape[0] == 0:
            return
        data = {"states": states, "next_states": next_states}
        if labels is not None and "labels" in self.buffers:
            data["labels"] = labels
        num_states = states.shape[0]
        head = min(num_states, self.buffer_size - self.step)
        for name, buf in self.buffers.items():
            value = data.get(name)
            if value is None:
                continue
            buf[self.step : self.step + head] = value[:head]
            if num_states > head:
                buf[: num_states - head] = value[head:]

        self.num_samples = min(self.buffer_size, max(self.step + num_states, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = torch.randint(self.num_samples, (mini_batch_size,), device=self.device)
            yield tuple(buf[sample_idxs] for buf in self.buffers.values())
