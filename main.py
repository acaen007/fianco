"""Gradient-interference splitting for continual learning — experiment runner.

Usage:
    python main.py                          # default settings
    python main.py --epochs 15 --lr 0.001   # override hyperparameters
    python main.py --split-threshold 0.2    # more aggressive splitting

Runs Sequential, EWC, and Splitting on Split MNIST, prints metrics,
and saves comparison plots.
"""

import argparse
import torch
from data import get_split_mnist
from trainer import train_sequential, train_ewc, train_splitting
from metrics import (compute_forgetting, print_summary, avg_accuracy,
                     plot_accuracy_heatmap, plot_model_sizes,
                     plot_comparison, plot_interference_scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ewc-lambda", type=float, default=400)
    p.add_argument("--split-threshold", type=float, default=0.5)
    p.add_argument("--max-splits", type=int, default=3)
    p.add_argument("--anchor-lambda", type=float, default=100.0,
                   help="Regularization strength for anchoring old units")
    p.add_argument("--split-layer", type=int, default=0,
                   help="Which hidden layer to apply splitting (0-indexed)")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="./data")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tasks = get_split_mnist(data_dir=args.data_dir, batch_size=args.batch_size)
    print(f"Tasks: {[t[2] for t in tasks]}\n")

    results = {}

    # 1. Sequential baseline
    print("--- Sequential fine-tuning ---")
    acc_seq, sizes_seq = train_sequential(
        tasks, device, epochs=args.epochs, lr=args.lr, hidden_dims=args.hidden)
    results["Sequential"] = {"acc_matrix": acc_seq, "sizes": sizes_seq}

    # 2. EWC
    print("\n--- EWC ---")
    acc_ewc, sizes_ewc = train_ewc(
        tasks, device, epochs=args.epochs, lr=args.lr,
        ewc_lambda=args.ewc_lambda, hidden_dims=args.hidden)
    results["EWC"] = {"acc_matrix": acc_ewc, "sizes": sizes_ewc}

    # 3. Splitting
    print("\n--- Gradient-interference splitting ---")
    acc_spl, sizes_spl, split_log = train_splitting(
        tasks, device, epochs=args.epochs, lr=args.lr, hidden_dims=args.hidden,
        split_layer=args.split_layer, split_threshold=args.split_threshold,
        max_splits_per_epoch=args.max_splits, anchor_lambda=args.anchor_lambda)
    results["Splitting"] = {"acc_matrix": acc_spl, "sizes": sizes_spl}

    # Print summaries
    for name, r in results.items():
        print_summary(name, r["acc_matrix"], compute_forgetting(r["acc_matrix"]))

    if split_log:
        print("\nSplit events:")
        for tid, epoch, n, _ in split_log:
            if n > 0:
                print(f"  Task {tid}, epoch {epoch}: {n} unit(s) split")

    # Save plots
    for name, r in results.items():
        plot_accuracy_heatmap(r["acc_matrix"], name,
                              f"heatmap_{name.lower()}.png")
    plot_model_sizes(
        {"Sequential": sizes_seq, "EWC": sizes_ewc, "Splitting": sizes_spl},
        "model_sizes.png")
    plot_comparison(results, "comparison.png")
    plot_interference_scores(split_log, "interference_scores.png")
    print("\nPlots saved: heatmap_*.png, model_sizes.png, comparison.png, "
          "interference_scores.png")


if __name__ == "__main__":
    main()
