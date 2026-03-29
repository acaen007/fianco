# Gradient-Interference Splitting for Continual Learning

Proof-of-concept: detect when hidden units receive conflicting gradient pressure from old vs new tasks, and split them into two copies — one biased toward preservation, one free to adapt.

## Quick start

```bash
pip install torch torchvision matplotlib
python main.py
```

Override defaults:

```bash
python main.py --epochs 15 --split-threshold 0.2 --max-splits 8 --ewc-lambda 600
```

## What it does

1. **Split MNIST** — 5 binary tasks: 0v1, 2v3, 4v5, 6v7, 8v9
2. **Three methods** trained sequentially:
   - **Sequential fine-tuning** (baseline, shows catastrophic forgetting)
   - **EWC** (elastic weight consolidation)
   - **Gradient-interference splitting** (our method)
3. **Outputs**: per-task accuracy tables, forgetting metrics, comparison plots

## How splitting works

After each task, we store per-unit average gradients for the split layer. During the next task, we accumulate per-unit gradients and compute **cosine similarity** against the stored old-task gradients.

- **Negative cosine** = old and new tasks push the unit in opposite directions
- If this exceeds a threshold persistently, the unit is **split**:
  - Incoming weights are duplicated (+ small noise for symmetry breaking)
  - Outgoing weights are halved between original and copy (preserves network output)
  - The original unit retains its old-task gradient history (biased toward preservation)
  - The new copy has no history (free to adapt to the new task)

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point, runs all experiments |
| `data.py` | Split MNIST dataset creation |
| `model.py` | Expandable MLP with `split_unit()` |
| `interference.py` | Gradient tracking and interference detection |
| `trainer.py` | Training loops for all three methods |
| `metrics.py` | Accuracy, forgetting, and plotting |

## Design choices

- **Multi-head output** (one binary head per task): isolates output-layer interference so we can study hidden-layer dynamics cleanly.
- **Split only one hidden layer** (configurable via `--split-layer`): keeps the prototype simple while demonstrating the mechanism.
- **Anchor regularization**: after each task, the split layer's weights are stored as an "anchor". During the next task, old units are penalized for drifting from their anchored values. New copies (appended beyond the anchor) are free to adapt. This is critical — without it, both copies drift toward the new task and forgetting persists.
- **Optimizer rebuild after splits**: loses momentum state. A production version would transfer state for unchanged parameters.
- **Cosine similarity** as interference metric: simple, interpretable, and direction-sensitive. Could be swapped for Fisher-weighted metrics.

## Results (default settings, seed=42)

| Method | Avg Accuracy | Avg Forgetting | Hidden Units |
|--------|:----------:|:-------------:|:------------:|
| Sequential | 0.968 | 0.030 | 256 |
| EWC | 0.948 | 0.050 | 256 |
| **Splitting** | **0.986** | **0.011** | 352 |

The splitting method achieves the highest accuracy and lowest forgetting at the cost of ~37% more hidden units in the split layer (128 → 224).

## Key hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--split-threshold` | 0.5 | Lower = more splits, higher = more selective |
| `--max-splits` | 3 | Max units split per epoch |
| `--anchor-lambda` | 100.0 | Old-unit protection strength |
| `--ewc-lambda` | 400 | EWC regularization strength |

## Known limitations

- Splitting is limited to one hidden layer per run.
- No pruning of unused units yet (TODO: add post-task pruning pass).
- Optimizer state is reset after splits (momentum lost).
- Interference metric is per-epoch; finer-grained tracking may be more responsive.
- Anchor regularization is uniform across old units; weighting by Fisher importance could help.
- This is a POC on a simple benchmark. Scaling to harder tasks needs further work.
