"""Image branch: ResNet-18 (ImageNet weights) fine-tuned on the Sentinel-2 tiles.

    python -m forest_fire_risk.cnn            # cache tiles, train, evaluate -> reports/cnn/

Same samples, years and metrics as the tabular models (see tabular.py), so the two branches
and the fusion model are compared on exactly the same rows:
    2017-2019  train. A random 15 % of these samples (stratified by label) is held out to
               stop training early and to pick the decision threshold.
    2020       never used to fit or tune anything. Only predicted, for the fusion model.
    2021-2022  test, scored once at the end, plain and season-matched.

Tiles are cached once as a uint8 array (data/tiles.npy, ~2.3 GB, git-ignored) so an epoch
takes seconds on a GPU instead of decoding 10k PNGs. Outputs (reports/cnn/): metrics.csv,
history.csv (loss / hold-out AUPRC per epoch) and predictions.csv (2020 + 2021-22
probabilities per sample_id, for fusion). Weights go to models/cnn/ (git-ignored).
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from forest_fire_risk.data import DATA_DIR, load_tile, tile_index
from forest_fire_risk.tabular import SEED, best_threshold, load_data, scores, season_weights

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "cnn"
MODEL_DIR = ROOT / "models" / "cnn"
CACHE = DATA_DIR.parents[1] / "tiles.npy"           # data/tiles.npy (+ tiles_ids.npy)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, PATIENCE, BATCH, LR, HOLDOUT = 30, 5, 64, 1e-4, 0.15
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)     # ImageNet statistics
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def tiles(sample_ids):
    """uint8 tensor [N, 3, 256, 256] with the tiles of `sample_ids`, in that order.
    Built from the cache; the cache (every tile with status=ok) is written on the first call."""
    ids_file = CACHE.with_name("tiles_ids.npy")
    if not CACHE.exists():
        ok = np.sort(tile_index().query("status == 'ok'").sample_id.to_numpy().astype(str))
        arr = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.uint8, shape=(len(ok), 256, 256, 3))
        for i, s in enumerate(ok):
            arr[i] = load_tile(s)
        arr.flush()
        np.save(ids_file, ok)
    arr, ids = np.load(CACHE, mmap_mode="r"), np.load(ids_file)
    pos = pd.Series(np.arange(len(ids)), index=ids)[np.asarray(sample_ids, dtype=str)].to_numpy()
    return torch.from_numpy(arr[pos]).permute(0, 3, 1, 2).contiguous()


def make_model(pretrained=True):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def prep(x, train=False):
    """uint8 [B, 3, H, W] -> normalised float on DEVICE. Training: random 90-degree turn
    (whole batch) and per-sample horizontal flip; satellite tiles have no 'up'."""
    x = x.to(DEVICE).float() / 255
    if train:
        x = torch.rot90(x, k=int(torch.randint(4, ())), dims=(2, 3))
        flip = torch.rand(len(x), device=DEVICE) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
    return (x - MEAN.to(DEVICE)) / STD.to(DEVICE)


@torch.no_grad()
def predict(model, X, batch=256):
    model.eval()
    return torch.cat([torch.sigmoid(model(prep(X[i:i + batch]))).squeeze(1).cpu()
                      for i in range(0, len(X), batch)]).numpy()


def train(X, y, X_hold, y_hold, epochs=EPOCHS):
    """Fine-tune on (X, y); keep the epoch with the best hold-out AUPRC, stop after
    PATIENCE epochs without improvement. Returns (model, history DataFrame)."""
    torch.manual_seed(SEED)
    model = make_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    yt = torch.tensor(np.array(y), dtype=torch.float32)
    best, best_state, since, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for idx in torch.randperm(len(X)).split(BATCH):
            loss = loss_fn(model(prep(X[idx], train=True)).squeeze(1), yt[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        auprc = average_precision_score(y_hold, predict(model, X_hold))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "holdout_auprc": auprc})
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  hold-out AUPRC {auprc:.4f}", flush=True)
        if auprc > best:
            best, best_state, since = auprc, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def main():
    tab = load_data()
    train_rows, val, test = (tab[tab.split == s] for s in ("train", "val", "test"))
    fit_rows, hold = train_test_split(train_rows, test_size=HOLDOUT, stratify=train_rows.label,
                                      random_state=SEED)
    print(f"{len(tab)} samples | fit {len(fit_rows)}  hold-out {len(hold)}  val {len(val)}  "
          f"test {len(test)} | device {DEVICE}")
    X = {name: tiles(rows.sample_id) for name, rows in
         [("fit", fit_rows), ("hold", hold), ("val", val), ("test", test)]}

    model, history = train(X["fit"], fit_rows.label, X["hold"], hold.label)
    p = {name: predict(model, X[name]) for name in ("hold", "val", "test")}
    threshold = best_threshold(hold.label, p["hold"])
    metrics = pd.DataFrame([{"model": "cnn", "evaluation": ev, **scores(test.label, p["test"], threshold, w)}
                            for ev, w in [("standard", None),
                                          ("season_matched", season_weights(test.label, test.month))]])
    print("\n", metrics.round(3).to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)
    history.to_csv(REPORT_DIR / "history.csv", index=False)
    preds = pd.concat([val, test])[["sample_id", "label", "time", "split"]].copy()
    preds["p_cnn"] = np.concatenate([p["val"], p["test"]])
    preds.to_csv(REPORT_DIR / "predictions.csv", index=False)
    pd.DataFrame({"sample_id": hold.sample_id, "label": hold.label, "p_cnn": p["hold"]}).to_csv(
        MODEL_DIR / "train_holdout.csv", index=False)
    torch.save(model.state_dict(), MODEL_DIR / "resnet18.pt")
    print(f"\nwrote {REPORT_DIR} and {MODEL_DIR}")


if __name__ == "__main__":
    main()
