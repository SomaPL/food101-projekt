"""Krok 1: pobranie Food-101 z Hugging Face i zapis 20 wybranych klas jako pliki JPEG.

Uruchomienie: python -m src.prepare_data
Pobiera ok. 5 GB do data/hf_cache (jednorazowo). Wynik: data/food20/<klasa>/*.jpg + data/food20/meta.csv
"""
import pandas as pd
from datasets import load_dataset
from PIL import Image

from .config import CLASSES, HF_CACHE, IMG_DIR, RESIZE_SHORT

# W HF podział "validation" to oficjalny zbiór TESTOWY Food-101 (ręcznie sprawdzony),
# a "train" to oficjalny treningowy (celowo nieczyszczony, ma część błędnych etykiet).
SOURCES = {"train": "food101_train", "validation": "food101_test"}


def resize_short_side(img: Image.Image, short: int) -> Image.Image:
    """Zmniejsza obraz tak, żeby krótszy bok miał `short` px (większych nie powiększa)."""
    w, h = img.size
    scale = short / min(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    return img


def main():
    ds = load_dataset("ethz/food101", cache_dir=str(HF_CACHE))
    names = ds["train"].features["label"].names
    hf_id_to_class = {names.index(c): c for c in CLASSES}

    meta = []
    for split, source in SOURCES.items():
        # filter po samej kolumnie "label" -> nie dekoduje obrazów, więc jest szybki
        part = ds[split].filter(lambda label: label in hf_id_to_class, input_columns="label")
        counters = {c: 0 for c in CLASSES}
        for i, example in enumerate(part):
            cls = hf_id_to_class[example["label"]]
            img = example["image"].convert("RGB")
            out = IMG_DIR / cls / f"{source}_{counters[cls]:04d}.jpg"
            counters[cls] += 1
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                resize_short_side(img, RESIZE_SHORT).save(out, quality=90)
            meta.append({
                "path": out.relative_to(IMG_DIR.parent.parent).as_posix(),
                "cls": cls,
                "source": source,
                "orig_w": img.width,
                "orig_h": img.height,
            })
            if (i + 1) % 2000 == 0:
                print(f"  {split}: {i + 1}/{len(part)}")
        print(f"{split}: zapisano {len(part)} zdjęć")

    pd.DataFrame(meta).to_csv(IMG_DIR / "meta.csv", index=False)
    print(f"Gotowe: {len(meta)} zdjęć w {IMG_DIR}")


if __name__ == "__main__":
    main()
