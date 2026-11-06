"""Analiza danych: liczba zdjęć, rozkład klas, rozmiary obrazów, siatka przykładów.

Uruchomienie: python -m src.eda   (po src.split)
Wynik: wykresy results/figures/eda_*.png i podsumowanie w konsoli.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .config import CLASSES, FIGURES, ROOT, SEED
from .data import load_manifest
from .utils import COLORS, INK, _style


def main():
    df = load_manifest()
    FIGURES.mkdir(parents=True, exist_ok=True)

    print(f"Zdjęć razem: {len(df)}, klas: {df['cls'].nunique()}")
    print(df.groupby("split").size().rename("zdjęć").to_string())
    print("\nZdjęcia na klasę według źródła (oficjalny train / test Food-101):")
    print(df.groupby("source").size().div(len(CLASSES)).rename("na klasę").to_string())
    print("\nOryginalne rozmiary (px):")
    print(df[["orig_w", "orig_h"]].describe().loc[["min", "50%", "max"]].astype(int).to_string())

    # 1. Rozkład klas: zbiór jest zbalansowany (1000 zdjęć na klasę).
    counts = df.groupby("cls").size().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(counts)), counts.values, color=COLORS["linear_probe"], width=0.7)
    ax.set_xticks(range(len(counts)), counts.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("liczba zdjęć")
    ax.set_title("Liczba zdjęć na klasę (pula przed podziałem)", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_class_counts.png", dpi=130)
    plt.close(fig)

    # 2. Rozmiary oryginalnych zdjęć.
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(df["orig_w"], df["orig_h"], s=8, alpha=0.3, color=COLORS["linear_probe"], edgecolors="none")
    ax.set_xlabel("szerokość [px]")
    ax.set_ylabel("wysokość [px]")
    ax.set_title("Rozmiary oryginalnych zdjęć", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_image_sizes.png", dpi=130)
    plt.close(fig)

    # 3. Siatka: 4 losowe zdjęcia z każdej klasy (zawsze te same dzięki seedowi).
    per_class = 4
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.6, len(CLASSES) * 1.6))
    for row, cls in enumerate(CLASSES):
        sample = df[df["cls"] == cls].sample(per_class, random_state=SEED)
        for col, path in enumerate(sample["path"]):
            ax = axes[row, col]
            ax.imshow(Image.open(ROOT / path))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_class_grid.png", dpi=110)
    plt.close(fig)
    print(f"\nWykresy zapisane w {FIGURES}")


if __name__ == "__main__":
    main()
