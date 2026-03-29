"""Evaluation metrics and plotting for continual learning experiments."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_forgetting(acc_matrix):
    """Per-task forgetting = best accuracy ever seen - final accuracy."""
    n = len(acc_matrix)
    rect = np.zeros((n, n))
    for i, row in enumerate(acc_matrix):
        for j, v in enumerate(row):
            rect[i, j] = v

    forgetting = []
    for t in range(n):
        best = max(rect[s, t] for s in range(t, n))
        forgetting.append(best - rect[n - 1, t])
    return forgetting


def avg_accuracy(acc_matrix):
    """Average accuracy across all tasks after the final training stage."""
    return float(np.mean(acc_matrix[-1]))


def print_summary(name, acc_matrix, forgetting):
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(f"  Final per-task acc:  {[f'{a:.3f}' for a in acc_matrix[-1]]}")
    print(f"  Average accuracy:    {avg_accuracy(acc_matrix):.3f}")
    print(f"  Per-task forgetting: {[f'{f:.3f}' for f in forgetting]}")
    print(f"  Average forgetting:  {np.mean(forgetting):.3f}")


def plot_accuracy_heatmap(acc_matrix, title, filename):
    """Heatmap: rows = after training task i, columns = accuracy on task j."""
    n = len(acc_matrix)
    rect = np.full((n, n), np.nan)
    for i, row in enumerate(acc_matrix):
        for j, v in enumerate(row):
            rect[i, j] = v

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rect, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xlabel("Task evaluated")
    ax.set_ylabel("After training on task")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"T{i}" for i in range(n)])
    ax.set_yticklabels([f"T{i}" for i in range(n)])
    for i in range(n):
        for j in range(n):
            if not np.isnan(rect[i, j]):
                ax.text(j, i, f"{rect[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_model_sizes(sizes_dict, filename):
    """Line plot of total hidden units across tasks."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, sizes in sizes_dict.items():
        ax.plot(range(len(sizes)), sizes, "o-", label=name)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Total hidden units")
    ax.set_title("Model capacity over tasks")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_comparison(results, filename):
    """Bar charts comparing average accuracy and forgetting across methods."""
    names = list(results.keys())
    avg_accs = [avg_accuracy(results[n]["acc_matrix"]) for n in names]
    avg_forg = [np.mean(compute_forgetting(results[n]["acc_matrix"])) for n in names]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"][:len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, avg_accs, color=colors)
    axes[0].set_ylabel("Average accuracy")
    axes[0].set_title("Final average accuracy")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(names, avg_forg, color=colors)
    axes[1].set_ylabel("Average forgetting")
    axes[1].set_title("Average forgetting (lower is better)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_interference_scores(split_log, filename):
    """Plot interference score distributions over training."""
    if not split_log:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for tid, epoch, n_splits, scores in split_log:
        if scores:
            ax.scatter([f"T{tid}E{epoch}"] * len(scores), scores,
                       alpha=0.3, s=10, color="steelblue")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Interference score (-cos sim)")
    ax.set_title("Per-unit interference scores during training")
    plt.xticks(rotation=45, fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
