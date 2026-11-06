"""Baseline: ResNet-18 z ImageNet, trenujemy TYLKO ostatnią warstwę (linear probe).

Uruchomienie: python -m src.linear_probe
Sztuczka: reszta sieci jest zamrożona, więc cechy (512 liczb na zdjęcie) liczymy raz
i zapisujemy. Potem trening samej warstwy 512 -> 20 trwa sekundy, dla wszystkich poziomów danych.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import CLASSES, CURVE_CSV, DATA_DIR, FIGURES, LEVELS, NUM_CLASSES, SEED
from .data import FoodDataset, get_transform, load_manifest, select
from .models import build_feature_extractor
from .utils import append_row, compute_metrics, get_device, plot_confusion, set_seed

FEATURES = DATA_DIR / "features_resnet18.pt"
STEPS, EVAL_EVERY, LR = 500, 10, 1e-2


@torch.no_grad()
def extract(model, df, device):
    loader = DataLoader(FoodDataset(df, get_transform(train=False, aug=False)), batch_size=64, num_workers=4)
    feats = [model(x.to(device)).cpu() for x, _ in loader]
    return torch.cat(feats), torch.tensor(df["label"].values)


def load_features(device):
    if FEATURES.exists():
        return torch.load(FEATURES)
    df = load_manifest()
    model = build_feature_extractor().to(device)
    feats = {}
    for split, n in [("train", max(LEVELS)), ("val", None), ("test", None)]:
        part = select(df, split, n)
        x, y = extract(model, part, device)
        feats[split] = {"x": x, "y": y, "rank": torch.tensor(part["train_rank"].values)}
        print(f"cechy {split}: {tuple(x.shape)}")
    torch.save(feats, FEATURES)
    return feats


def train_head(x_train, y_train, x_val, y_val):
    """Uczy samą warstwę 512 -> 20 na gotowych cechach; zwraca wersję najlepszą na walidacji."""
    head = nn.Linear(x_train.shape[1], NUM_CLASSES)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    best_acc, best_state, best_step = -1.0, None, 0
    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head(x_train), y_train)
        loss.backward()
        optimizer.step()
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                acc = (head(x_val).argmax(1) == y_val).float().mean().item()
            if acc > best_acc:
                best_acc, best_state, best_step = acc, {k: v.clone() for k, v in head.state_dict().items()}, step
    head.load_state_dict(best_state)
    return head, best_acc, best_step


def main():
    device = get_device()
    feats = load_features(device)
    train, val, test = feats["train"], feats["val"], feats["test"]
    for n in LEVELS:
        set_seed(SEED)
        mask = train["rank"] < n
        head, val_acc, best_step = train_head(train["x"][mask], train["y"][mask], val["x"], val["y"])
        with torch.no_grad():
            pred = head(test["x"]).argmax(1).numpy()
        m = compute_metrics(test["y"].numpy(), pred)
        print(f"linear_probe n={n}: val_acc={val_acc:.3f} test_acc={m['acc']:.3f} macro_f1={m['macro_f1']:.3f}")
        append_row(CURVE_CSV, {
            "run": f"linear_probe_n{n}", "strategy": "linear_probe", "aug": False, "n_per_class": n,
            "best_epoch": best_step, "val_acc": val_acc, "test_acc": m["acc"], "test_macro_f1": m["macro_f1"],
            "epochs": STEPS, "lr": LR, "train_time_s": 0, "device": str(device),
        })
        if n == max(LEVELS):  # baseline z polecenia: 100 zdjęć na klasę
            plot_confusion(test["y"].numpy(), pred, CLASSES, f"baseline (linear probe, n={n}): test acc {m['acc']:.2f}",
                           FIGURES / f"linear_probe_n{n}_confusion.png")


if __name__ == "__main__":
    main()
