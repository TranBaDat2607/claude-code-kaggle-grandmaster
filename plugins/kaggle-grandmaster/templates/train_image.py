"""Image classification / regression template (timm + PyTorch).

Copy to src/, edit CONFIG, then:

    python src/train_image.py --smoke                     # 1 fold, 2 batches/epoch — crash test
    python src/train_image.py --bench                     # data loader vs GPU step vs real loop: input- or compute-bound?
    python src/train_image.py --name effb3_384 --backbone tf_efficientnet_b3.ns_jft_in1k --img-size 384
    # two GPUs: run disjoint folds in parallel; the run that completes the set logs the experiment
    CUDA_VISIBLE_DEVICES=0 python src/train_image.py --name cnx --folds 0 2 4 &
    CUDA_VISIBLE_DEVICES=1 python src/train_image.py --name cnx --folds 1 3
    # on Kaggle GPUs (quota-aware, resumable, both T4s busy): see the kaggle-gpu skill
    python -m kgkit gpu build --name cnx --hours 6 --cmd "python src/train_image.py --name cnx --folds {fold}"

Speed defaults (skill compute-efficiency): AMP, channels_last model and inputs, cudnn.benchmark, fused AdamW,
pinned memory and persistent workers. --compile adds torch.compile (Linux, sm_70+ GPUs: T4 yes, P100 no; pays on
runs over ~20 min). Images are decoded with PIL and resized every epoch: point image_dir at copies pre-resized to
img_size (or cache uint8 npy) when --bench says INPUT-BOUND.

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
import platform
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
from kgkit import bench as B
from kgkit import metrics as M
from kgkit import state as S
from kgkit.budget import TrainBudget
from kgkit.experiment import Ledger, seed_everything

CONFIG = dict(
    name="img_baseline",
    train="data/train.csv",
    test="data/test.csv",
    sample="data/sample_submission.csv",
    folds=os.environ.get("KG_FOLDS_FILE", "data/folds.csv"),  # kgkit recheck points this at a re-drawn split
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
        x = x.to(device, non_blocking=True, memory_format=memory_format(device))
        with torch.autocast(device_type=device.type, enabled=amp):
            p = to_pred(task, model(x).float())
            if hflip:
                p = (p + to_pred(task, model(torch.flip(x, dims=[3])).float())) / 2
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def autocast_dtype(device):
    """bf16 where the GPU has bf16 tensor cores (compute capability 8.0+: A100, L4, H100, RTX 30xx+), else fp16 with a
    GradScaler. Not torch.cuda.is_bf16_supported(): it also says True on a T4 or P100, which only emulate bf16, slowly."""
    return torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def memory_format(device):
    return torch.channels_last if device.type == "cuda" else torch.contiguous_format


def compile_ok(device) -> tuple:
    """(ok, reason): torch.compile needs Triton, i.e. Linux (or WSL2) and a GPU of compute capability 7.0+."""
    if device.type != "cuda":
        return False, "no CUDA device"
    if platform.system() == "Windows":
        return False, "native Windows has no official Triton (use WSL2)"
    if torch.cuda.get_device_capability() < (7, 0):
        return False, f"{torch.cuda.get_device_name()} is below compute capability 7.0 (P100: use a T4)"
    return True, ""


def metric_input(task, metric, pred):
    if metric.kind == "label" and task == "multiclass":
        return pred.argmax(1)
    if task in ("binary", "regression") and pred.ndim == 2:
        return pred[:, 0]
    return pred


# ----------------------------------------------------------------------------- bench
def bench(steps, loader, train_step, device, mf, cfg, use_compile):
    """Samples/s of the data loader alone, of train_step on one batch already on the device, and of the real loop.
    The slower of the first two limits training; kgkit.bench prints the verdict and the speed-up against the last
    bench of this run name (reports/bench.jsonl)."""
    it = B.forever(loader)
    x0, y0 = next(it)
    fixed = (x0.to(device, non_blocking=True, memory_format=mf), y0.to(device, non_blocking=True))

    def data_step():
        x, _ = next(it)
        return len(x)

    def compute_step():
        train_step(*fixed)
        return len(fixed[0])

    def full_step():
        x, yb = next(it)
        train_step(x.to(device, non_blocking=True, memory_format=mf), yb.to(device, non_blocking=True))
        return len(x)

    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    rates = B.measure({"data": data_step, "compute": compute_step, "full": full_step}, n=steps,
                      warmup=10 if use_compile else 3, sync=torch.cuda.synchronize if cuda else None)
    extra = {"device": torch.cuda.get_device_name() if cuda else "cpu (rates are for this CPU, not a GPU)",
             "cpu cores": os.cpu_count(), "workers": cfg["num_workers"]}
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        extra["peak GPU memory"] = (f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB of {total:.0f} MB "
                                    "(headroom: a larger batch, if the score holds)")
    config = {k: cfg[k] for k in ("backbone", "img_size", "batch_size", "num_workers", "image_dir")}
    config.update(device=extra["device"], compile=use_compile,
                  amp=("bf16" if autocast_dtype(device) == torch.bfloat16 else "fp16") if cuda else False)
    B.report(cfg["name"], rates, config=config, extra=extra, kind="train_image")


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
    ap.add_argument("--assemble", action="store_true", help="no training: assemble finished folds and log")
    ap.add_argument("--bench", type=int, nargs="?", const=30, metavar="STEPS",
                    help="time the data loader alone, the training step alone and the real loop on the first "
                         "fold (default 30 steps each), print the verdict and the speed-up against the last bench; "
                         "trains and logs nothing")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model for training (Linux, sm_70+)")
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input size: cuDNN picks the fastest kernels once
    use_compile = False
    if a.compile:
        use_compile, why = compile_ok(device)
        if not use_compile:
            print(f"note: --compile skipped: {why}", flush=True)
    mf = memory_format(device)

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
    if a.bench:
        run_folds = run_folds[:1]
    tf_train, tf_eval = build_transforms(cfg["img_size"], True), build_transforms(cfg["img_size"], False)
    loader_kw = dict(num_workers=cfg["num_workers"], pin_memory=amp, persistent_workers=cfg["num_workers"] > 0)
    crit = loss_fn(task, cfg["label_smoothing"])
    t0 = time.time()
    stopped = None  # "timeout" | "pruned": the session deadline or the baseline curve ended this run early

    for fold in ([] if a.assemble else run_folds):
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
        fwd = torch.compile(model) if use_compile else model  # shares the weights; EMA and checkpoints use model
        opt = torch.optim.AdamW(param_groups(model, cfg["lr"], cfg["head_lr_mult"], cfg["weight_decay"]),
                                **({"fused": True} if device.type == "cuda" else {}))
        total = cfg["epochs"] * len(dl_tr)
        sched = cosine_with_warmup(opt, total, int(cfg["warmup_frac"] * total))
        amp_dtype = autocast_dtype(device)
        scaler = torch.amp.GradScaler(enabled=amp and amp_dtype == torch.float16)
        ema = EMA(model, cfg["ema_decay"])
        best, best_pred, start_epoch = metric.worst(), None, 0
        budget = TrainBudget(out_dir, fold, cfg["epochs"], metric.greater_is_better)

        def train_step(x, yb):
            with torch.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
                loss = crit(fwd(x).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            return loss

        if a.bench:
            bench(a.bench, dl_tr, train_step, device, mf, cfg, use_compile)
            return
        resume = out_dir / f"fold{fold}_resume.pt"
        if resume.exists() and not a.smoke:  # continue a fold the previous session checkpointed
            ck = torch.load(resume, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            ema.module.load_state_dict(ck["ema"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            best, best_pred, start_epoch = ck["best"], ck["best_pred"], ck["epoch"]
            print(f"fold {fold}: resumed at epoch {start_epoch} (best {best:.5f})", flush=True)
        for epoch in range(start_epoch, cfg["epochs"]):
            budget.start_epoch()
            model.train()
            for x, yb in dl_tr:
                loss = train_step(x.to(device, non_blocking=True, memory_format=mf), yb.to(device, non_blocking=True))
            pred = predict(ema.module, dl_va, device, task, amp, hflip=False)
            score = metric(y_metric[va], metric_input(task, metric, pred))
            print(f"fold {fold} epoch {epoch} loss {loss.item():.4f} val {metric.name} {score:.5f}", flush=True)
            if metric.is_better(score, best) or best_pred is None:
                best, best_pred = score, pred
                torch.save(ema.module.state_dict(), out_dir / f"fold{fold}.pt")
            action = "continue" if a.smoke else budget.end_epoch(epoch, score)
            if action == "timeout":
                torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1,
                            "best": best, "best_pred": best_pred}, resume)
                budget.write_status("timeout", next_epoch=epoch + 1)
                print(f"fold {fold}: session deadline near - checkpointed after epoch {epoch}; resume later", flush=True)
            elif action == "prune":
                budget.write_status("pruned", epoch=epoch)
                print(f"fold {fold}: PRUNED at epoch {epoch} - best {best:.5f} trails the baseline curve", flush=True)
            if action != "continue":
                stopped = "timeout" if action == "timeout" else "pruned"
                break
        if stopped:
            break
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
        if not a.smoke:
            budget.best = best
            budget.write_status("done")
            resume.unlink(missing_ok=True)
        print(f"fold {fold} best {metric.name} {best:.5f}", flush=True)

    if a.smoke:
        print("smoke run OK (not logged)")
        return
    if stopped:
        print(f"run stopped early ({stopped}); not logged")
        return
    if os.environ.get("KG_DEFER_ASSEMBLE") and not a.assemble:
        print("fold(s) finished; assembly deferred to the remote runner")
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
