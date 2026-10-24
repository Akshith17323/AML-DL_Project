"""Learning-curve plots from recorded training histories."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_curves(histories: list[dict], best_epochs: list[int], path: str | Path) -> Path:
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), sharey=True, squeeze=False)
    for i, (ax, h, b) in enumerate(zip(axes[0], histories, best_epochs)):
        ax.plot(h["train_loss"], label="inner-train")
        ax.plot(h["val_loss"], label="inner-validation")
        ax.axvline(b, color="grey", ls="--", lw=0.8, label="best epoch")
        ax.set_title(f"fold {i}"); ax.set_xlabel("epoch")
        if i == 0:
            ax.set_ylabel("BCE loss"); ax.legend(fontsize=7)
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    return path
