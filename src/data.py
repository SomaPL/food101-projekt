"""Wczytywanie danych do PyTorcha: Dataset i transformacje (w tym augmentacja)."""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .config import IMG_SIZE, MANIFEST, RESIZE_SHORT, ROOT

# Średnia i odchylenie kolorów z ImageNet - wagi ImageNet "oczekują" tak znormalizowanych wejść.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transform(train: bool, aug: bool):
    if train and aug:
        steps = [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),  # losowy wycinek i skala
            T.RandomHorizontalFlip(),                         # odbicie lustrzane
            T.ColorJitter(0.2, 0.2, 0.2),                     # jasność, kontrast, nasycenie
        ]
    else:
        steps = [T.Resize(RESIZE_SHORT), T.CenterCrop(IMG_SIZE)]  # zawsze ten sam środek zdjęcia
    return T.Compose(steps + [T.ToTensor(), T.Normalize(MEAN, STD)])


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST)


def select(df: pd.DataFrame, split: str, n_per_class: int | None = None) -> pd.DataFrame:
    """Zwraca część podziału; dla train można ograniczyć do N pierwszych zdjęć na klasę."""
    part = df[df["split"] == split]
    if n_per_class is not None:
        part = part[part["train_rank"] < n_per_class]
    return part.reset_index(drop=True)


class FoodDataset(Dataset):
    """Zwraca parę (obraz jako tensor 3x224x224, numer klasy)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(ROOT / self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]
