"""Trening całej sieci: transfer + fine-tuning albo od zera, z augmentacją lub bez.

Przykłady:
  python -m src.train --strategy finetune --n 10 --aug
  python -m src.train --strategy scratch --n 10 --aug --eval-every 5
Wynik: wiersz w results/learning_curve.csv, wykresy w results/figures/, predykcje w results/predictions/.
"""
import argparse
import copy
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import CLASSES, CURVE_CSV, FIGURES, LEVELS, PREDICTIONS, SEED
from .data import FoodDataset, get_transform, load_manifest, select
from .models import build_resnet18
from .utils import append_row, compute_metrics, get_device, plot_confusion, plot_history, set_seed

# Od zera sieć potrzebuje większego kroku uczenia i dłuższego treningu niż przy fine-tuningu.
DEFAULTS = {
    "finetune": {"epochs": 15, "lr": 3e-4},
    "scratch": {"epochs": 40, "lr": 1e-3},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["finetune", "scratch"], required=True)
    p.add_argument("--n", type=int, choices=LEVELS, required=True, help="zdjęć treningowych na klasę")
    p.add_argument("--aug", action="store_true", help="włącz augmentację")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=1, help="co ile epok liczyć walidację")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wandb", action="store_true", help="loguj do Weights & Biases")
    args = p.parse_args()
    for key, value in DEFAULTS[args.strategy].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def make_loader(df, train, aug, args, generator=None):
    return DataLoader(
        FoodDataset(df, get_transform(train=train, aug=aug)),
        batch_size=args.batch_size,
        shuffle=train,
        generator=generator,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,  # na Windows tworzenie workerów co epokę jest wolne
    )


@torch.no_grad()  # przy ocenie nie liczymy gradientów: szybciej i mniej pamięci
def predict(model, loader, device):
    model.eval()  # BatchNorm i Dropout w trybie oceny
    y_true, y_pred, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        y_true.append(y.cpu())
        y_pred.append(logits.argmax(1).cpu())
    y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
    return y_true, y_pred, loss_sum / len(y_true)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()            # 1. wyzeruj gradienty z poprzedniego kroku
        logits = model(x)                # 2. przewidywanie
        loss = criterion(logits, y)      # 3. jak bardzo się pomylił
        loss.backward()                  # 4. gradienty: w którą stronę zmienić wagi
        optimizer.step()                 # 5. zmiana wag
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    run_name = f"{args.strategy}{'_aug' if args.aug else ''}_n{args.n}"
    print(f"{run_name} | urządzenie: {device} | epoki: {args.epochs} | lr: {args.lr}")

    df = load_manifest()
    train_df = select(df, "train", args.n)
    val_df, test_df = select(df, "val"), select(df, "test")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = make_loader(train_df, True, args.aug, args, generator)
    val_loader = make_loader(val_df, False, False, args)
    test_loader = make_loader(test_df, False, False, args)

    model = build_resnet18(pretrained=args.strategy == "finetune").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="food101-lowdata", name=run_name, config=vars(args))

    history, best = [], {"val_acc": -1.0}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            y_val, p_val, val_loss = predict(model, val_loader, device)
            m = compute_metrics(y_val, p_val)
            row.update(val_loss=val_loss, val_acc=m["acc"], val_macro_f1=m["macro_f1"])
            if m["acc"] > best["val_acc"]:  # zapamiętaj najlepszą epokę wg walidacji
                best = {"val_acc": m["acc"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        history.append(row)
        print(" | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if run:
            run.log(row)
    train_time = time.time() - start

    # Test liczymy RAZ, na modelu z najlepszej epoki według walidacji.
    model.load_state_dict(best["state"])
    y_test, p_test, _ = predict(model, test_loader, device)
    test = compute_metrics(y_test, p_test)
    print(f"TEST: acc={test['acc']:.3f} macro_f1={test['macro_f1']:.3f} "
          f"(najlepsza epoka {best['epoch']}, czas {train_time:.0f} s)")

    plot_history(history, run_name, FIGURES / f"{run_name}_history.png")
    plot_confusion(y_test, p_test, CLASSES, f"{run_name}: test acc {test['acc']:.2f}",
                   FIGURES / f"{run_name}_confusion.png")
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": test_df["path"], "true": y_test, "pred": p_test}).to_csv(
        PREDICTIONS / f"{run_name}.csv", index=False)
    append_row(CURVE_CSV, {
        "run": run_name, "strategy": args.strategy, "aug": args.aug, "n_per_class": args.n,
        "best_epoch": best["epoch"], "val_acc": best["val_acc"],
        "test_acc": test["acc"], "test_macro_f1": test["macro_f1"],
        "epochs": args.epochs, "lr": args.lr, "train_time_s": round(train_time), "device": str(device),
    })
    if run:
        run.summary.update({"test_acc": test["acc"], "test_macro_f1": test["macro_f1"], "train_time_s": train_time})
        run.finish()


if __name__ == "__main__":
    main()
