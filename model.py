"""Expandable MLP with support for hidden-unit splitting.

Architecture: input -> [hidden layers + ReLU] -> task-specific heads.
Multi-head design isolates output-layer interference, letting us study
hidden-layer dynamics cleanly.
"""

import torch
import torch.nn as nn


class ExpandableMLP(nn.Module):

    def __init__(self, input_dim=784, hidden_dims=None, classes_per_task=2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.input_dim = input_dim
        self.classes_per_task = classes_per_task

        dims = [input_dim] + list(hidden_dims)
        self.hidden = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))
        ])
        self.heads = nn.ModuleList()
        self._head_in = hidden_dims[-1]

    def add_head(self):
        """Add a new task head. Returns the new task id."""
        self.heads.append(nn.Linear(self._head_in, self.classes_per_task))
        return len(self.heads) - 1

    def forward(self, x, task_id):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.heads[task_id](x)

    # ---- Architecture expansion ----

    def split_unit(self, layer_idx, unit_idx, noise_scale=0.001):
        """Split one hidden unit into two, preserving network output.

        1. Duplicate incoming weights (+ small noise for symmetry breaking).
        2. Halve outgoing weights between original and copy.

        The original keeps its index; the copy is appended at the end.
        Because outgoing weights are split 50/50, the network's output is
        unchanged immediately after the split.
        """
        dev = self.hidden[layer_idx].weight.device
        old = self.hidden[layer_idx]

        # Expand the layer itself: add one output unit
        new_layer = nn.Linear(old.in_features, old.out_features + 1, device=dev)
        with torch.no_grad():
            new_layer.weight[:old.out_features] = old.weight
            new_layer.weight[-1] = (old.weight[unit_idx]
                                    + noise_scale * torch.randn(old.in_features, device=dev))
            new_layer.bias[:old.out_features] = old.bias
            new_layer.bias[-1] = old.bias[unit_idx]
        self.hidden[layer_idx] = new_layer

        # Expand downstream: add one input column, halve the split unit's outgoing weights
        is_last = (layer_idx == len(self.hidden) - 1)
        targets = list(self.heads) if is_last else [self.hidden[layer_idx + 1]]

        for i, tgt in enumerate(targets):
            exp = nn.Linear(tgt.in_features + 1, tgt.out_features, device=dev)
            with torch.no_grad():
                exp.weight[:, :tgt.in_features] = tgt.weight
                # Halve original column, copy half to new column
                exp.weight[:, unit_idx] = tgt.weight[:, unit_idx] * 0.5
                exp.weight[:, -1] = tgt.weight[:, unit_idx] * 0.5
                exp.bias[:] = tgt.bias
            if is_last:
                self.heads[i] = exp
            else:
                self.hidden[layer_idx + 1] = exp

        if is_last:
            self._head_in = new_layer.out_features

    def layer_dims(self):
        """Current output dimensions of each hidden layer."""
        return [l.out_features for l in self.hidden]

    def total_hidden_units(self):
        return sum(self.layer_dims())

    def total_params(self):
        return sum(p.numel() for p in self.parameters())
