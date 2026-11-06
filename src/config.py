"""Wspólne ustawienia projektu: ścieżki, seed, klasy, poziomy danych."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"                  # ignorowane przez git (surowe dane)
HF_CACHE = DATA_DIR / "hf_cache"          # cache Hugging Face (~5 GB, można usunąć po kroku 1)
IMG_DIR = DATA_DIR / "food20"             # nasze 20 klas jako pliki JPEG
RESULTS = ROOT / "results"                # trafia do repo: CSV i wykresy
FIGURES = RESULTS / "figures"
PREDICTIONS = RESULTS / "predictions"
MANIFEST = RESULTS / "manifest.csv"       # który plik należy do train / val / test
CURVE_CSV = RESULTS / "learning_curve.csv"

SEED = 1234
LEVELS = [10, 25, 50, 100]                # zdjęcia treningowe na klasę (punkty krzywej)
SPLIT = (0.8, 0.1, 0.1)                   # train / val / test liczone od puli 1000 zdjęć na klasę
RESIZE_SHORT = 256                        # zapisujemy zdjęcia z krótszym bokiem 256 px, trenujemy na 224 px
IMG_SIZE = 224

# Trzy grupy klas: pozwalają sprawdzić, czy transfer pomaga bardziej tam,
# gdzie ImageNet "widział" podobne obiekty.
CLASS_GROUPS = {
    "w_imagenet": ["pizza", "hamburger", "hot_dog", "ice_cream", "guacamole", "spaghetti_carbonara"],
    "podobne": ["steak", "filet_mignon", "prime_rib", "chocolate_cake", "chocolate_mousse", "tiramisu"],
    "poza_imagenet": ["sushi", "ramen", "falafel", "caesar_salad", "dumplings", "macarons",
                      "french_fries", "pad_thai"],
}
CLASSES = [c for group in CLASS_GROUPS.values() for c in group]
CLASS_TO_GROUP = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}
NUM_CLASSES = len(CLASSES)
assert NUM_CLASSES == 20, NUM_CLASSES
