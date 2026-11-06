"""Narzędzia wspólne: seed, zapis wyników, wykresy."""
import random

import matplotlib
matplotlib.use("Agg")  # rysujemy do plików, bez okienek
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Stała paleta: każdy wariant ma zawsze ten sam kolor na każdym wykresie.
COLORS = {
    "linear_probe": "#2a78d6",
    "finetune": "#eb6834",
    "finetune + aug": "#1baf7a",
    "scratch": "#eda100",
    "scratch + aug": "#e87ba4",
    "train": "#2a78d6",
    "val": "#eb6834",
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def append_row(path, row: dict) -> None:
    """Dopisuje jeden wiersz do CSV (tworzy plik, jeśli go nie ma)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot_history(history: list[dict], title: str, path) -> None:
    """Krzywe treningowe: loss i accuracy dla train vs val."""
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric in zip(axes, ["loss", "acc"]):
        ax.plot(df["epoch"], df[f"train_{metric}"], color=COLORS["train"], lw=2, label="train")
        val = df.dropna(subset=[f"val_{metric}"])
        ax.plot(val["epoch"], val[f"val_{metric}"], color=COLORS["val"], lw=2, marker="o", ms=5, label="val")
        ax.set_xlabel("epoka")
        ax.set_title("loss" if metric == "loss" else "accuracy", color=INK)
        _style(ax)
        ax.legend(frameon=False)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names, title: str, path) -> None:
    """Confusion matrix znormalizowana po wierszach: jaki % zdjęć klasy X model nazwał Y."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
    ax.set_xlabel("przewidziana klasa")
    ax.set_ylabel("prawdziwa klasa")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] >= 0.1:
                ax.text(j, i, f"{cm[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=6,
                        color="#ffffff" if cm[i, j] > 0.5 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
