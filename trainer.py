"""Training loops for continual learning experiments.

Three methods:
  1. Sequential fine-tuning (baseline, exhibits catastrophic forgetting)
  2. EWC (elastic weight consolidation)
  3. Gradient-interference-triggered splitting (our method)
"""

import torch
import torch.nn.functional as F
from model import ExpandableMLP
from interference import InterferenceDetector


def evaluate(model, loader, task_id, device):
    """Evaluate accuracy on one task."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x, task_id).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def _eval_all(model, tasks, n_seen, device):
    return [evaluate(model, tasks[t][1], t, device) for t in range(n_seen)]


# ===================================================================
# 1. Sequential fine-tuning (naive baseline)
# ===================================================================

def train_sequential(tasks, device, epochs=10, lr=1e-3, hidden_dims=None):
    model = ExpandableMLP(hidden_dims=hidden_dims or [128, 128]).to(device)
    acc_matrix = []

    for tid, (tr, te, cls) in enumerate(tasks):
        model.add_head()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            model.train()
            for x, y in tr:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                F.cross_entropy(model(x, tid), y).backward()
                opt.step()

        accs = _eval_all(model, tasks, tid + 1, device)
        acc_matrix.append(accs)
        print(f"  [Sequential] After task {tid}: {[f'{a:.3f}' for a in accs]}")

    return acc_matrix, [model.total_hidden_units()] * len(tasks)


# ===================================================================
# 2. EWC
# ===================================================================

def _compute_fisher(model, loader, task_id, device, max_samples=500):
    """Diagonal Fisher information matrix (per-parameter)."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    count = 0
    for x, y in loader:
        if count >= max_samples:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        F.cross_entropy(model(x, task_id), y).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data ** 2 * x.size(0)
        count += x.size(0)
    for n in fisher:
        fisher[n] /= max(count, 1)
    return fisher


def train_ewc(tasks, device, epochs=10, lr=1e-3, ewc_lambda=400,
              hidden_dims=None):
    model = ExpandableMLP(hidden_dims=hidden_dims or [128, 128]).to(device)
    acc_matrix = []
    fishers, star_params = {}, {}

    for tid, (tr, te, cls) in enumerate(tasks):
        model.add_head()
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.train()
            for x, y in tr:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x, tid), y)

                # EWC penalty: sum over all old tasks
                ewc_pen = 0.0
                for t_old in range(tid):
                    for n, p in model.named_parameters():
                        if n in fishers[t_old]:
                            ewc_pen += (fishers[t_old][n]
                                        * (p - star_params[t_old][n]) ** 2).sum()

                (loss + ewc_lambda * ewc_pen).backward()
                opt.step()

        fishers[tid] = _compute_fisher(model, tr, tid, device)
        star_params[tid] = {n: p.data.clone() for n, p in model.named_parameters()}

        accs = _eval_all(model, tasks, tid + 1, device)
        acc_matrix.append(accs)
        print(f"  [EWC] After task {tid}: {[f'{a:.3f}' for a in accs]}")

    return acc_matrix, [model.total_hidden_units()] * len(tasks)


# ===================================================================
# 3. Gradient-interference-triggered splitting (our method)
# ===================================================================

def train_splitting(tasks, device, epochs=10, lr=1e-3, hidden_dims=None,
                    split_layer=0, split_threshold=0.5,
                    max_splits_per_epoch=3, noise_scale=0.001,
                    anchor_lambda=100.0):
    """Our method: detect gradient interference, split overloaded units.

    Key addition: anchor regularization. After each task, we store the split
    layer's weights as an 'anchor'. During the next task, a penalty keeps
    old units (those that existed before this task) close to their anchored
    values. This protects old representations while new copies adapt freely.
    Without this, both copies drift toward the new task and forgetting persists.
    """
    model = ExpandableMLP(hidden_dims=hidden_dims or [128, 128]).to(device)
    detector = InterferenceDetector(split_layer_idx=split_layer)
    acc_matrix = []
    model_sizes = []
    split_log = []
    anchor = None  # anchored weights for the split layer [n_old_units, in_features]

    for tid, (tr, te, cls) in enumerate(tasks):
        model.add_head()
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            detector.reset_accumulator(model)

            for x, y in tr:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x, tid), y)

                # Anchor regularization: penalize drift of old units in the
                # split layer. New units (from splits) are appended beyond
                # anchor.shape[0] and are NOT penalized — free to adapt.
                if anchor is not None and anchor_lambda > 0:
                    layer_w = model.hidden[split_layer].weight
                    n_old = anchor.shape[0]
                    loss = loss + anchor_lambda * (
                        (layer_w[:n_old] - anchor) ** 2).sum() / n_old

                loss.backward()

                if tid > 0:
                    detector.accumulate(model)

                opt.step()

            # End-of-epoch: check for units to split
            if tid > 0:
                scores = detector.compute_scores()
                candidates = detector.get_split_candidates(
                    scores, threshold=split_threshold,
                    max_splits=max_splits_per_epoch)

                for uid in candidates:
                    model.split_unit(split_layer, uid, noise_scale=noise_scale)
                    detector.update_after_split(uid)

                if candidates:
                    # Rebuild optimizer after architecture change.
                    # This loses momentum state -- acceptable for a POC.
                    opt = torch.optim.Adam(model.parameters(), lr=lr)

                split_log.append((tid, epoch, len(candidates),
                                  scores.tolist() if scores is not None else []))

        # Store reference gradients for future interference detection
        detector.store_reference(model, tr, tid, device)
        # Anchor the split layer's current weights for next task's regularization
        anchor = model.hidden[split_layer].weight.data.clone().detach()

        accs = _eval_all(model, tasks, tid + 1, device)
        acc_matrix.append(accs)
        model_sizes.append(model.total_hidden_units())
        print(f"  [Splitting] After task {tid}: {[f'{a:.3f}' for a in accs]}  "
              f"layers={model.layer_dims()}")

    return acc_matrix, model_sizes, split_log
