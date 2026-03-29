"""Gradient interference detection for continual learning.

Core idea: after each task, store per-unit average gradients for the split layer.
During the next task, accumulate per-unit gradients and compare directions.
Persistent negative cosine similarity means the unit is under conflicting
gradient pressure from old vs new tasks -- a candidate for splitting.

This module is designed to be swappable: the same interface could use Fisher
diagonal, activation statistics, or other importance measures.
"""

import torch
import torch.nn.functional as F


class InterferenceDetector:
    """Tracks per-unit gradient history and detects interference on one layer."""

    def __init__(self, split_layer_idx=0):
        self.layer_idx = split_layer_idx
        # Per-task stored gradient references: {task_id: Tensor[num_units, in_features]}
        self.ref_grads = {}
        self._accum = None   # running sum of gradients for the current task
        self._count = 0

    def store_reference(self, model, dataloader, task_id, device):
        """After finishing a task, store its per-unit average gradients.

        Each unit's "reference gradient" is the average of dL/dW[unit, :] over the
        task's data. This captures the direction the task's loss wants to push each
        unit's incoming weights.
        """
        layer = model.hidden[self.layer_idx]
        accum = torch.zeros_like(layer.weight)  # [out_features, in_features]
        n = 0

        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            out = model(x, task_id)
            F.cross_entropy(out, y).backward()
            accum += layer.weight.grad.data
            n += x.size(0)
        model.train()

        self.ref_grads[task_id] = (accum / n).detach().clone()

    def reset_accumulator(self, model):
        """Reset the current-task gradient accumulator (call at each epoch start)."""
        self._accum = torch.zeros_like(model.hidden[self.layer_idx].weight)
        self._count = 0

    def accumulate(self, model):
        """Accumulate gradients from the latest backward pass."""
        grad = model.hidden[self.layer_idx].weight.grad
        if grad is not None:
            self._accum += grad.data
            self._count += 1

    def compute_scores(self):
        """Per-unit interference score = -cos_sim(current_grad, old_aggregate_grad).

        Returns Tensor[num_units] where higher = more interference, or None if no data.
        """
        if not self.ref_grads or self._count == 0:
            return None

        current = self._accum / self._count          # [num_units, in_features]
        num_units = current.shape[0]

        # Aggregate old-task references, padded to current layer size
        old_agg = torch.zeros_like(current)
        for ref in self.ref_grads.values():
            n = min(ref.shape[0], num_units)
            old_agg[:n] += ref[:n]

        # Per-unit cosine similarity; score = -cos so positive means conflict
        scores = torch.zeros(num_units)
        for j in range(num_units):
            c_norm = current[j].norm()
            o_norm = old_agg[j].norm()
            if c_norm > 1e-8 and o_norm > 1e-8:
                cos = F.cosine_similarity(
                    current[j].unsqueeze(0), old_agg[j].unsqueeze(0)).item()
                scores[j] = -cos
        return scores

    def get_split_candidates(self, scores, threshold=0.3, max_splits=5):
        """Return indices of units with interference above threshold, sorted descending."""
        if scores is None:
            return []
        mask = scores > threshold
        if not mask.any():
            return []
        indices = torch.where(mask)[0]
        order = scores[indices].argsort(descending=True)
        return indices[order[:max_splits]].tolist()

    def update_after_split(self, unit_idx):
        """After splitting a unit, append a zero row to all stored references.

        The new unit (appended at end) has no old-task history, so it won't
        be flagged as interfering. The original unit keeps its reference,
        biasing it toward preservation. This naturally achieves the
        'one copy preserves old, one adapts' behavior.
        """
        for task_id in self.ref_grads:
            ref = self.ref_grads[task_id]
            zero_row = torch.zeros(1, ref.shape[1], device=ref.device)
            self.ref_grads[task_id] = torch.cat([ref, zero_row], dim=0)
