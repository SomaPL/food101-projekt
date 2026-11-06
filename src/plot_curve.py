"""Krzywa uczenia: liczba zdjęć na klasę vs accuracy na teście, osobna linia dla każdego wariantu.

Uruchomienie: python -m src.plot_curve   (czyta results/learning_curve.csv)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import CURVE_CSV, FIGURES, LEVELS
from .utils import COLORS, INK, MUTED, _style

LABELS = {
    "linear_probe": "tylko ostatnia warstwa (baseline)",
    "finetune": "transfer + fine-tuning",
    "finetune + aug": "transfer + fine-tuning + aug",
    "scratch": "od zera",
    "scratch + aug": "od zera + aug",
}


def main():
    df = pd.read_csv(CURVE_CSV)
    df["variant"] = df["strategy"] + df["aug"].map({True: " + aug", False: ""})
    # Jeśli jakiś run powtórzono, bierzemy ostatni wynik.
    df = df.drop_duplicates(subset=["variant", "n_per_class"], keep="last")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for variant in LABELS:
        part = df[df["variant"] == variant].sort_values("n_per_class")
        if part.empty:
            continue
        ax.plot(part["n_per_class"], part["test_acc"], color=COLORS[variant], lw=2, marker="o", ms=7,
                label=LABELS[variant])
    ax.axhline(0.65, color=MUTED, ls="--", lw=1.2)
    ax.text(LEVELS[0], 0.665, "próg 0,65", color=MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(LEVELS, [str(n) for n in LEVELS])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("zdjęć treningowych na klasę (skala log)")
    ax.set_ylabel("accuracy na teście")
    ax.set_title("Krzywa uczenia: Food-101, 20 klas, ResNet-18", color=INK)
    _style(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "learning_curve.png", dpi=130)
    plt.close(fig)
    print(df[["variant", "n_per_class", "test_acc", "test_macro_f1"]].sort_values(["variant", "n_per_class"])
          .to_string(index=False))


if __name__ == "__main__":
    main()
