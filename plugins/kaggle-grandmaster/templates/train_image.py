"""Image classification / regression template (timm + PyTorch).

Copy to src/, edit CONFIG, then:

    python src/train_image.py --smoke                     # 1 fold, 2 batches/epoch — crash test
    python src/train_image.py --name effb3_384 --backbone tf_efficientnet_b3.ns_jft_in1k --img-size 384
    # two GPUs: run disjoint folds in parallel; the run that completes the set logs the experiment
    CUDA_VISIBLE_DEVICES=0 python src/train_image.py --name cnx --folds 0 2 4 &
    CUDA_VISIBLE_DEVICES=1 python src/train_image.py --name cnx --folds 1 3

The train CSV needs an image path column (relative to image_dir) and the target column(s).
task: "multiclass" (single label), "multilabel" (several 0/1 columns), "regression"/"binary"
(single column, one output). Per-fold best checkpoints, OOF and test predictions land in
artifacts/<name>/; the assembled experiment is logged to the ledger with a submission file.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import kgkit  # noqa: F401
except ImportError:
    sys.path.insert(0, os.environ.get("KGKIT_HOME", ""))
from kgkit import metrics as M
from kgkit import state as S
from kgkit.experiment import Ledger, seed_everything

CONFIG = dict(
    name="img_baseline",
    train="data/train.csv",
    test="data/test.csv",
    sample="data/sample_submission.csv",
    folds="data/folds.csv",
    image_dir="data/train_images",
    test_image_dir="data/test_images",
    path_col="image_path",      # column with file names (relative to image_dir)
    path_suffix="",             # e.g. ".jpg" when the column holds bare ids
    id_col=None,                # default: competition.json
    targets=None,               # list of target columns; default: [competition.json target]
    metric=None,                # default: competition.json
    task="multiclass",          # multiclass | multilabel | binary | regression
    backbone="convnext_tiny.fb_in22k_ft_in1k",
    pretrained=True,
    img_size=384,
    epochs=15,
    batch_size=32,
    lr=2e-4,
    head_lr_mult=10.0,
    weight_decay=1e-2,
    warmup_frac=0.05,
    label_smoothing=0.05,
    ema_decay=0.999,
    grad_clip=1.0,
    num_workers=4,
    tta_hflip=True,
    seed=42,
    notes="",
)


# ----------------------------------------------------------------------------- data
def build_transforms(size: int, train: bool):
    from torchvision.transforms import v2 as T

    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.02)], p=0.5),
            T.RandomApply([T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1))], p=0.5),
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.25),
        ])
    return T.Compose([
        T.Resize((size, size), antialias=True),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class ImageDataset(Dataset):
    def __init__(self, paths, targets, transform):
        self.paths, self.targets, self.transform = list(paths), targets, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.transform(img)
        y = torch.zeros(1) if self.targets is None else torch.as_tensor(self.targets[i])
        return x, y


# ----------------------------------------------------------------------------- model utils
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay, self.module = decay, copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def param_groups(model, lr, head_mult, wd):
    """Backbone / head groups (head gets lr * head_mult); no weight decay on norms and biases."""
    head_ids = {id(p) for p in model.get_classifier().parameters()} if hasattr(model, "get_classifier") else set()
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = id(p) in head_ids
        decay = not (p.ndim <= 1 or n.endswith(".bias"))
        g = groups.setdefault((head, decay), {"params": [], "lr": lr * (head_mult if head else 1.0),
                                              "weight_decay": wd if decay else 0.0})
        g["params"].append(p)
    return list(groups.values())


def cosine_with_warmup(opt, total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def loss_fn(task, smoothing):
    if task == "multiclass":
        ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        return lambda out, y: ce(out, y.long().view(-1))
    if task in ("multilabel", "binary"):
        bce = nn.BCEWithLogitsLoss()
        return lambda out, y: bce(out, y.float().view_as(out))
    mse = nn.MSELoss()
    return lambda out, y: mse(out.view(-1), y.float().view(-1))


def to_pred(task, logits):
    if task == "multiclass":
        return logits.softmax(-1)
    if task in ("multilabel", "binary"):
        return logits.sigmoid()
    return logits


@torch.no_grad()
def predict(model, loader, device, task, amp, hflip):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            p = to_pred(task, model(x).float())
            if hflip:
                p = (p + to_pred(task, model(torch.flip(x, dims=[3])).float())) / 2
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def metric_input(task, metric, pred):
    if metric.kind == "label" and task == "multiclass":
        return pred.argmax(1)
    if task in ("binary", "regression") and pred.ndim == 2:
        return pred[:, 0]
    return pred


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--backbone")
    ap.add_argument("--img-size", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--folds", type=int, nargs="+", help="subset of folds to train (default: all)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--notes")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 2 batches per epoch, 1 epoch, nothing logged")
    a = ap.parse_args()
    cfg = dict(CONFIG)
    for k in ("name", "backbone", "img_size", "epochs", "batch_size", "lr", "notes"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.no_pretrained:
        cfg["pretrained"] = False

    import timm

    st = S.load()
    id_col = cfg["id_col"] or (st.id_col if st else None)
    targets = cfg["targets"] or [st.target]
    metric = M.get(cfg["metric"] or st.metric)
    task = cfg["task"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    seed_everything(cfg["seed"])

    train = pd.read_csv(cfg["train"])
    folds_df = pd.read_csv(cfg["folds"])
    if id_col and id_col in folds_df.columns:
        train = train.merge(folds_df[[id_col, "fold"]], on=id_col, how="left", validate="one_to_one")
    else:
        train["fold"] = folds_df["fold"].to_numpy()
    test = pd.read_csv(cfg["test"]) if Path(cfg["test"]).exists() else None

    classes = None
    if task == "multiclass":
        classes, y = np.unique(train[targets[0]].to_numpy(), return_inverse=True)
        n_out = len(classes)
    else:
        y = train[targets].to_numpy(dtype=np.float32)
        n_out = len(targets)
    y_metric = y[:, 0] if (task != "multiclass" and n_out == 1) else y  # shape the metric expects

    paths = [str(Path(cfg["image_dir"]) / f"{p}{cfg['path_suffix']}") for p in train[cfg["path_col"]]]
    test_paths = None
    if test is not None:
        test_paths = [str(Path(cfg["test_image_dir"]) / f"{p}{cfg['path_suffix']}") for p in test[cfg["path_col"]]]

    out_dir = Path("artifacts") / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    all_folds = sorted(train["fold"].unique().tolist())
    run_folds = a.folds if a.folds is not None else all_folds
    if a.smoke:
        run_folds = run_folds[:1]
        cfg["epochs"] = 1
    tf_train, tf_eval = build_transforms(cfg["img_size"], True), build_transforms(cfg["img_size"], False)
    loader_kw = dict(num_workers=cfg["num_workers"], pin_memory=amp, persistent_workers=cfg["num_workers"] > 0)
    crit = loss_fn(task, cfg["label_smoothing"])
    t0 = time.time()

    for fold in run_folds:
        tr = np.flatnonzero(train["fold"].to_numpy() != fold)
        va = np.flatnonzero(train["fold"].to_numpy() == fold)
        if a.smoke:
            tr, va = tr[: 2 * cfg["batch_size"]], va[: cfg["batch_size"]]
        dl_tr = DataLoader(ImageDataset([paths[i] for i in tr], y[tr], tf_train), batch_size=cfg["batch_size"],
                           shuffle=True, drop_last=len(tr) > cfg["batch_size"], **loader_kw)
        dl_va = DataLoader(ImageDataset([paths[i] for i in va], y[va], tf_eval), batch_size=cfg["batch_size"] * 2,
                           shuffle=False, **loader_kw)
        model = timm.create_model(cfg["backbone"], pretrained=cfg["pretrained"], num_classes=n_out).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(param_groups(model, cfg["lr"], cfg["head_lr_mult"], cfg["weight_decay"]))
        total = cfg["epochs"] * len(dl_tr)
        sched = cosine_with_warmup(opt, total, int(cfg["warmup_frac"] * total))
        scaler = torch.amp.GradScaler(enabled=amp and not torch.cuda.is_bf16_supported())
        ema = EMA(model, cfg["ema_decay"])
        best, best_pred = metric.worst(), None
        for epoch in range(cfg["epochs"]):
            model.train()
            for x, yb in dl_tr:
                x, yb = x.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp,
                                    dtype=torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16):
                    loss = crit(model(x).float(), yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                sched.step()
                ema.update(model)
            pred = predict(ema.module, dl_va, device, task, amp, hflip=False)
            score = metric(y_metric[va], metric_input(task, metric, pred))
            print(f"fold {fold} epoch {epoch} loss {loss.item():.4f} val {metric.name} {score:.5f}", flush=True)
            if metric.is_better(score, best) or best_pred is None:
                best, best_pred = score, pred
                torch.save(ema.module.state_dict(), out_dir / f"fold{fold}.pt")
        if cfg["tta_hflip"]:
            ema.module.load_state_dict(torch.load(out_dir / f"fold{fold}.pt", map_location=device))
            best_pred = predict(ema.module, dl_va, device, task, amp, hflip=True)
            best = metric(y_metric[va], metric_input(task, metric, best_pred))
        np.save(out_dir / f"fold{fold}_oof.npy", best_pred)
        np.save(out_dir / f"fold{fold}_idx.npy", va)
        if test_paths is not None and not a.smoke:
            dl_te = DataLoader(ImageDataset(test_paths, None, tf_eval), batch_size=cfg["batch_size"] * 2, **loader_kw)
            ema.module.load_state_dict(torch.load(out_dir / f"fold{fold}.pt", map_location=device))
            np.save(out_dir / f"fold{fold}_test.npy", predict(ema.module, dl_te, device, task, amp, cfg["tta_hflip"]))
        print(f"fold {fold} best {metric.name} {best:.5f}", flush=True)

    if a.smoke:
        print("smoke run OK (not logged)")
        return
    done = [f for f in all_folds if (out_dir / f"fold{f}_oof.npy").exists()]
    if done != all_folds:
        print(f"folds done {done}/{all_folds}; the run that completes the set will log the experiment")
        return

    # ------------------------------------------------------------ assemble + log
    oof = np.zeros((len(train), n_out), dtype=np.float32)
    fold_scores = []
    for f in all_folds:
        idx, p = np.load(out_dir / f"fold{f}_idx.npy"), np.load(out_dir / f"fold{f}_oof.npy")
        oof[idx] = p
        fold_scores.append(metric(y_metric[idx], metric_input(task, metric, p)))
    cv = metric(y_metric, metric_input(task, metric, oof))
    test_pred = None
    if test_paths is not None:
        test_pred = np.mean([np.load(out_dir / f"fold{f}_test.npy") for f in all_folds], axis=0)
    rec = Ledger().log(cfg["name"], cv, fold_scores=fold_scores, model=cfg["backbone"], notes=cfg["notes"],
                       params={k: v for k, v in cfg.items() if k not in ("notes",)}, oof=oof, test_pred=test_pred,
                       runtime_s=round(time.time() - t0, 1))
    print(f"[{cfg['name']}] {metric.name} CV {cv:.5f} +/- {np.std(fold_scores):.5f} -> {rec['id']}")
    if test_pred is not None and Path(cfg["sample"]).exists():
        sample = pd.read_csv(cfg["sample"])
        key, cols = sample.columns[0], list(sample.columns[1:])
        pred = pd.DataFrame(index=test.index)
        if task == "multiclass" and metric.kind == "label":
            pred[cols[0]] = classes[test_pred.argmax(1)]
        elif task == "multiclass" and len(cols) == n_out:
            pred[cols] = test_pred
        else:
            pred[cols] = test_pred[:, : len(cols)]
        if key in test.columns:
            pred.insert(0, key, test[key].to_numpy())
            sub = sample[[key]].merge(pred, on=key, how="left")
        else:
            sub = pd.concat([sample[[key]], pred.reset_index(drop=True)], axis=1)
        Path("subs").mkdir(exist_ok=True)
        out = Path("subs") / f"{rec['id']}.csv"
        sub.to_csv(out, index=False)
        Ledger().update(rec["id"], submission=str(out).replace("\\", "/"))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
