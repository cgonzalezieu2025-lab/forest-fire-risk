"""Image branch: ResNet-18 (ImageNet weights) fine-tuned on the Sentinel-2 tiles.

    python -m forest_fire_risk.cnn            # cache tiles, train every seed, evaluate -> reports/cnn/
    python -m forest_fire_risk.cnn freeze     # side experiment: how many ResNet blocks to freeze

Same samples, years and metrics as the tabular models (see tabular.py), so the two branches
and the fusion model are compared on exactly the same rows:
    2017-2019  train, in two steps (the same idea as the tabular models' cross-validation + refit):
               1. fit on 2017-18 with 2019 held out; 2019 picks the number of epochs and the
                  decision threshold. A whole year, not a random share, because fires of one
                  outbreak are near-duplicates and would sit on both sides of a random split.
               2. refit on all of 2017-19 for that number of epochs. This model predicts.
    2020       never used to fit or tune anything. Only predicted, for the fusion model.
    2021-2022  test, scored once at the end, plain and season-matched.

Tiles are cached once as a uint8 array (data/tiles.npy, ~2.3 GB, git-ignored) so an epoch
takes seconds on a GPU instead of decoding 10k PNGs. Outputs (reports/cnn/): metrics.csv,
history.csv (loss / hold-out AUPRC per epoch) and predictions.csv (2020 + 2021-22
probabilities per sample_id, for fusion). GPU training is not reproducible and single runs
differ by a few points of AUPRC, so the reported CNN is the **average probability of the
len(SEEDS) trainings**; runs.csv and predictions_runs.csv hold every single training.
Weights go to models/cnn/ (git-ignored).
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from forest_fire_risk.data import DATA_DIR, load_tile, tile_index
from forest_fire_risk.tabular import SEED, best_threshold, load_data, scores, season_weights

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "cnn"
MODEL_DIR = ROOT / "models" / "cnn"
CACHE = DATA_DIR.parents[1] / "tiles.npy"           # data/tiles.npy (+ tiles_ids.npy)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, PATIENCE, BATCH, LR = 30, 5, 64, 1e-4
HOLDOUT_YEAR = 2019
SEEDS = [SEED + i for i in range(5)]                # the reported CNN averages these trainings
FREEZE = 0             # ResNet blocks kept at their ImageNet weights; set from `cnn freeze`
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


def frozen_parts(model, freeze):
    """The stem plus the first `freeze` residual blocks (nothing when freeze = 0)."""
    return list(model.children())[:4 + freeze] if freeze else []


def make_model(pretrained=True, freeze=FREEZE):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    for part in frozen_parts(model, freeze):
        part.requires_grad_(False)
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


def train(X, y, X_hold=None, y_hold=None, epochs=EPOCHS, seed=SEED, freeze=FREEZE):
    """Fine-tune on (X, y). With a hold-out: keep the epoch with the best hold-out AUPRC and stop
    after PATIENCE epochs without improvement. Without: run exactly `epochs` epochs.
    Returns (model, history DataFrame)."""
    torch.manual_seed(seed)
    model = make_model(freeze=freeze).to(DEVICE)
    opt = torch.optim.AdamW([w for w in model.parameters() if w.requires_grad], lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    yt = torch.tensor(np.array(y), dtype=torch.float32)
    best, best_state, since, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        for part in frozen_parts(model, freeze):
            part.eval()                             # frozen batch-norm keeps its ImageNet statistics
        losses = []
        for idx in torch.randperm(len(X)).split(BATCH):
            loss = loss_fn(model(prep(X[idx], train=True)).squeeze(1), yt[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if X_hold is None:
            print(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}", flush=True)
            continue
        auprc = average_precision_score(y_hold, predict(model, X_hold))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "holdout_auprc": auprc})
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  hold-out AUPRC {auprc:.4f}", flush=True)
        if auprc > best:
            best, best_state, since = auprc, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def load():
    """(rows, tiles) per split: fit 2017-18, hold 2019, train = both, val 2020, test 2021-22."""
    tab = load_data()
    train_rows = tab[tab.split == "train"]
    is_hold = pd.to_datetime(train_rows.time).dt.year == HOLDOUT_YEAR
    rows = {"fit": train_rows[~is_hold], "hold": train_rows[is_hold], "train": train_rows,
            "val": tab[tab.split == "val"], "test": tab[tab.split == "test"]}
    print(" ".join(f"{k} {len(v)}" for k, v in rows.items()), "| device", DEVICE)
    X = {k: tiles(rows[k].sample_id) for k in ("train", "val", "test")}
    X["fit"], X["hold"] = X["train"][torch.tensor(~is_hold.to_numpy())], X["train"][torch.tensor(is_hold.to_numpy())]
    return rows, X


def test_metrics(test, p, threshold):
    return pd.DataFrame([{"model": "cnn", "evaluation": ev, **scores(test.label, p, threshold, w)}
                         for ev, w in [("standard", None),
                                       ("season_matched", season_weights(test.label, test.month))]])


def run(rows, X, seed, freeze=FREEZE, refit=True):
    """One training (step 1, and step 2 when `refit`). Returns model, history, probabilities
    per split, test metrics and a summary row."""
    print(f"\n--- seed {seed}, {freeze} frozen blocks", flush=True)
    model, history = train(X["fit"], rows["fit"].label, X["hold"], rows["hold"].label, seed=seed, freeze=freeze)
    p = {"hold": predict(model, X["hold"])}             # honest 2019 predictions, for the threshold
    if refit:
        print("refit on 2017-19", flush=True)
        model, _ = train(X["train"], rows["train"].label, epochs=int(history.holdout_auprc.idxmax()) + 1,
                         seed=seed, freeze=freeze)
    p.update({k: predict(model, X[k]) for k in ("val", "test")})
    threshold = best_threshold(rows["hold"].label, p["hold"])
    metrics = test_metrics(rows["test"], p["test"], threshold)
    std, sm = metrics.iloc[0], metrics.iloc[1]
    summary = {"seed": seed, "freeze": freeze, "epochs": len(history), "best_epoch": int(history.holdout_auprc.idxmax()) + 1,
               "holdout_auprc": history.holdout_auprc.max(), "final_train_loss": history.train_loss.iloc[-1],
               "threshold": threshold, "auprc": std.auprc, "roc_auc": std.roc_auc, "f1": std.f1,
               "auprc_season_matched": sm.auprc}
    return {"model": model, "history": history, "p": p, "metrics": metrics, "summary": summary}


def compare_freeze(levels=(0, 1, 2, 3), seeds=SEEDS[:3]):
    """Side experiment against overfitting: freeze the stem + first k residual blocks.
    Step 1 only (fit 2017-18, no refit). Choose k by the 2019 hold-out column, never by the
    test columns."""
    rows, X = load()
    out = pd.DataFrame([run(rows, X, seed, k, refit=False)["summary"] for k in levels for seed in seeds])
    out.to_csv(REPORT_DIR / "freeze_experiment.csv", index=False)
    print("\n", out.groupby("freeze")[["holdout_auprc", "auprc", "auprc_season_matched", "final_train_loss"]]
          .agg(["mean", "std"]).round(3).to_string())


def main():
    rows, X = load()
    runs = [run(rows, X, seed) for seed in SEEDS]
    summary = pd.DataFrame([r["summary"] for r in runs])
    p = {k: np.mean([r["p"][k] for r in runs], axis=0) for k in ("hold", "val", "test")}   # the reported CNN
    metrics = test_metrics(rows["test"], p["test"], best_threshold(rows["hold"].label, p["hold"]))
    preds = pd.concat([rows["val"], rows["test"]])[["sample_id", "label", "time", "split"]].copy()
    all_preds = preds.assign(**{f"p_cnn_{r['summary']['seed']}": np.concatenate([r["p"]["val"], r["p"]["test"]])
                                for r in runs})
    print("\nsingle trainings:\n", summary.round(3).to_string(index=False))
    print("\nmean / sd over seeds:\n", summary[["holdout_auprc", "auprc", "roc_auc", "f1", "auprc_season_matched"]]
          .agg(["mean", "std", "min", "max"]).round(3).to_string())
    print(f"\naverage of the {len(runs)} trainings (reported):\n", metrics.round(3).to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)
    pd.concat([r["history"].assign(seed=r["summary"]["seed"]) for r in runs]).to_csv(REPORT_DIR / "history.csv", index=False)
    summary.to_csv(REPORT_DIR / "runs.csv", index=False)
    all_preds.to_csv(REPORT_DIR / "predictions_runs.csv", index=False)
    preds.assign(p_cnn=np.concatenate([p["val"], p["test"]])).to_csv(REPORT_DIR / "predictions.csv", index=False)
    pd.DataFrame({"sample_id": rows["hold"].sample_id, "label": rows["hold"].label, "p_cnn": p["hold"]}).to_csv(
        MODEL_DIR / "train_holdout.csv", index=False)
    for r in runs:
        torch.save(r["model"].state_dict(), MODEL_DIR / f"resnet18_{r['summary']['seed']}.pt")
    print(f"\nwrote {REPORT_DIR} and {MODEL_DIR}")


if __name__ == "__main__":
    compare_freeze() if sys.argv[1:] == ["freeze"] else main()
