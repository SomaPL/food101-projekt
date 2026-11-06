"""Krok 2 (własna implementacja): podział 80/10/10 i zagnieżdżone podzbiory treningowe.

Uruchomienie: python -m src.split
Wynik: results/manifest.csv - jeden wiersz na zdjęcie, z kolumnami split i train_rank.

Jak to działa:
1. Dla każdej klasy tasujemy jej 1000 zdjęć generatorem z seedem 1234.
2. Pierwsze 80% -> train (pula), kolejne 10% -> val, reszta -> test.
3. train_rank = pozycja zdjęcia w potasowanej puli treningowej.
   Podzbiór "N zdjęć na klasę" = zdjęcia z train_rank < N.
   Dzięki temu podzbiory są zagnieżdżone: 10 ⊂ 25 ⊂ 50 ⊂ 100.
"""
import numpy as np
import pandas as pd

from .config import CLASS_TO_GROUP, CLASSES, IMG_DIR, LEVELS, MANIFEST, SEED, SPLIT


def split_counts(n: int, fractions) -> tuple[int, int, int]:
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    return n_train, n_val, n - n_train - n_val


def make_manifest(meta: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for label, cls in enumerate(CLASSES):
        # sortowanie po ścieżce -> ta sama kolejność startowa na każdym komputerze
        files = meta[meta["cls"] == cls].sort_values("path").reset_index(drop=True)
        files = files.iloc[rng.permutation(len(files))].reset_index(drop=True)

        n_train, n_val, _ = split_counts(len(files), SPLIT)
        split = np.full(len(files), "test", dtype=object)
        split[:n_train] = "train"
        split[n_train:n_train + n_val] = "val"

        files["split"] = split
        files["label"] = label
        files["group"] = CLASS_TO_GROUP[cls]
        files["train_rank"] = np.where(split == "train", np.arange(len(files)), -1)
        parts.append(files)
    return pd.concat(parts, ignore_index=True)


def check(df: pd.DataFrame) -> None:
    """Testy poprawności - lepiej, żeby program sam się zatrzymał, niż żebyśmy trenowali na złym podziale."""
    assert df["path"].is_unique, "to samo zdjęcie w dwóch miejscach"
    counts = df.groupby(["cls", "split"]).size().unstack()
    print("Liczba zdjęć na klasę w każdej części:")
    print(counts.describe().loc[["min", "max"]].astype(int))
    for n in LEVELS:
        subset = df[(df["split"] == "train") & (df["train_rank"] < n)]
        per_class = subset.groupby("cls").size()
        assert (per_class == n).all(), f"poziom {n}: nierówna liczba zdjęć na klasę"
    small = set(df[(df["split"] == "train") & (df["train_rank"] < LEVELS[0])]["path"])
    big = set(df[(df["split"] == "train") & (df["train_rank"] < LEVELS[-1])]["path"])
    assert small <= big, "podzbiory nie są zagnieżdżone"
    print("OK: podział unikalny, poziomy równe na klasę, podzbiory zagnieżdżone.")


def main():
    meta = pd.read_csv(IMG_DIR / "meta.csv")
    df = make_manifest(meta)
    check(df)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    print(f"Zapisano {MANIFEST}")


if __name__ == "__main__":
    main()
