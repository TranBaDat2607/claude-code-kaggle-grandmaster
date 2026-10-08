"""Transformer fine-tuning template for text classification / regression (Hugging Face).

Copy to src/, edit CONFIG, then:

    python src/train_transformer.py --smoke
    python src/train_transformer.py --name debv3b_512 --model microsoft/deberta-v3-base --max-len 512
    CUDA_VISIBLE_DEVICES=0 python src/train_transformer.py --name x --folds 0 2 4 &   # split folds across GPUs
    CUDA_VISIBLE_DEVICES=1 python src/train_transformer.py --name x --folds 1 3

Recipe: mean/attention pooling head, layer-wise LR decay, linear warmup + cosine decay, AMP,
gradient accumulation, evaluation several times per epoch with best-checkpoint selection,
length-sorted dynamic-padding inference. task: regression | binary | multiclass | multilabel.
For offline Kaggle kernels point --model at a local folder (save_pretrained output).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import kgkit  # noqa: F401
except ImportError:
    sys.path.insert(0, os.environ.get("KGKIT_HOME", ""))
from kgkit import metrics as M
from kgkit import state as S
from kgkit.experiment import Ledger, seed_everything

CONFIG = dict(
    name="deberta_baseline",
    train="data/train.csv",
    test="data/test.csv",
    sample="data/sample_submission.csv",
    folds="data/folds.csv",
    text_cols=["text"],          # joined with the tokenizer's sep token
    id_col=None,                 # default: competition.json
    targets=None,                # default: [competition.json target]
    metric=None,                 # default: competition.json
    task="regression",           # regression | binary | multiclass | multilabel
    model="microsoft/deberta-v3-base",
    max_len=512,
    batch_size=8,
    grad_accum=2,
    epochs=3,
    lr=2e-5,
    head_lr=1e-3,
    llrd=0.9,                    # layer-wise LR decay factor (1.0 = off)
    weight_decay=0.01,
    warmup_frac=0.1,
    pooling="mean",              # mean | cls | attention
    dropout=0.0,                 # 0 is usually best for regression heads
    evals_per_epoch=4,
    grad_clip=1.0,
    num_workers=2,
    seed=42,
    notes="",
)


class TextDataset(Dataset):
    def __init__(self, enc, targets):
        self.enc, self.targets = enc, targets

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.targets is not None:
            item["labels"] = self.targets[i]
        return item


class Collate:
    def __init__(self, tokenizer, task):
        self.tok, self.task = tokenizer, task

    def __call__(self, batch):
        labels = [b.pop("labels") for b in batch] if "labels" in batch[0] else None
        out = self.tok.pad(batch, return_tensors="pt")
        if labels is not None:
            dtype = torch.long if self.task == "multiclass" else torch.float32
            out["labels"] = torch.as_tensor(np.array(labels), dtype=dtype)
        return out


class Model(nn.Module):
    def __init__(self, name, n_out, pooling, dropout):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(name)
        for k in ("hidden_dropout_prob", "attention_probs_dropout_prob"):
            if hasattr(cfg, k) and dropout == 0.0:
                setattr(cfg, k, 0.0)
        self.backbone = AutoModel.from_pretrained(name, config=cfg)
        h = cfg.hidden_size
        self.pooling = pooling
        if pooling == "attention":
            self.att = nn.Sequential(nn.Linear(h, h // 2), nn.Tanh(), nn.Linear(h // 2, 1))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, n_out)

    def forward(self, input_ids, attention_mask, **kw):
        kw = {k: v for k, v in kw.items() if k in ("token_type_ids",)}
        hs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kw).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        if self.pooling == "cls":
            x = hs[:, 0]
        elif self.pooling == "attention":
            w = self.att(hs).masked_fill(m == 0, -1e4).softmax(1)
            x = (w * hs).sum(1)
        else:
            x = (hs * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(self.drop(x))


def llrd_groups(model, lr, head_lr, decay, wd):
    """Layer-wise LR decay: top encoder layer gets lr, each layer below gets lr * decay^depth."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm", "norm.weight")
    layers = None
    for attr in ("encoder.layer", "layers", "transformer.layer", "encoder.layers"):
        obj = model.backbone
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            layers = list(obj)
            break
        except AttributeError:
            continue
    groups, assigned = [], set()

    def add(params_named, group_lr):
        dec = [p for n, p in params_named if not any(nd in n for nd in no_decay)]
        nodec = [p for n, p in params_named if any(nd in n for nd in no_decay)]
        for ps, w in ((dec, wd), (nodec, 0.0)):
            if ps:
                groups.append({"params": ps, "lr": group_lr, "weight_decay": w})
        assigned.update(id(p) for _, p in params_named)

    head = [(n, p) for n, p in model.named_parameters() if not n.startswith("backbone.")]
    add(head, head_lr)
    if layers:
        for depth, layer in enumerate(reversed(layers)):
            add(list(layer.named_parameters()), lr * decay ** depth)
        rest = [(n, p) for n, p in model.backbone.named_parameters() if id(p) not in assigned]
        add(rest, lr * decay ** len(layers))  # embeddings & anything else
    else:
        add(list(model.backbone.named_parameters()), lr)
    return groups


def loss_fn(task):
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    if task in ("binary", "multilabel"):
        bce = nn.BCEWithLogitsLoss()
        return lambda o, y: bce(o, y.view_as(o))
    mse = nn.MSELoss()
    return lambda o, y: mse(o, y.view_as(o))


def to_pred(task, logits):
    if task == "multiclass":
        return logits.softmax(-1)
    if task in ("binary", "multilabel"):
        return logits.sigmoid()
    return logits


@torch.no_grad()
def predict(model, ds, collate, device, task, amp, bs):
    model.eval()
    order = np.argsort([len(x) for x in ds.enc["input_ids"]])  # length-sorted batches = less padding
    out = np.zeros((len(ds), model.head.out_features), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s: s + bs]
        batch = collate([{k: v for k, v in ds[i].items() if k != "labels"} for i in idx])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=amp):
            out[idx] = to_pred(task, model(**batch).float()).cpu().numpy()
    return out


def metric_input(task, metric, pred):
    if metric.kind == "label" and task == "multiclass":
        return pred.argmax(1)
    if task in ("binary", "regression") and pred.ndim == 2 and pred.shape[1] == 1:
        return pred[:, 0]
    return pred


def main():
    ap = argparse.ArgumentParser()
    for k, t in (("name", str), ("model", str), ("max_len", int), ("batch_size", int), ("epochs", int),
                 ("lr", float), ("notes", str)):
        ap.add_argument("--" + k.replace("_", "-"), type=t)
    ap.add_argument("--folds", type=int, nargs="+")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = dict(CONFIG)
    for k in ("name", "model", "max_len", "batch_size", "epochs", "lr", "notes"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)

    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

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
        y_metric = y
    else:
        y = train[targets].to_numpy(dtype=np.float32)
        n_out = len(targets)
        y_metric = y[:, 0] if n_out == 1 else y

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    sep = f" {tok.sep_token} " if tok.sep_token else " \n "

    def encode(df):
        texts = df[cfg["text_cols"]].fillna("").astype(str).agg(sep.join, axis=1).tolist()
        return tok(texts, truncation=True, max_length=cfg["max_len"])

    enc = encode(train)
    enc_test = encode(test) if test is not None else None
    collate = Collate(tok, task)
    out_dir = Path("artifacts") / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    all_folds = sorted(train["fold"].unique().tolist())
    run_folds = a.folds if a.folds is not None else all_folds
    if a.smoke:
        run_folds, cfg["epochs"] = run_folds[:1], 1
    crit = loss_fn(task)
    t0 = time.time()

    def subset(e, idx):
        return {k: [v[i] for i in idx] for k, v in e.items()}

    for fold in run_folds:
        folds_arr = train["fold"].to_numpy()
        tr, va = np.flatnonzero(folds_arr != fold), np.flatnonzero(folds_arr == fold)
        if a.smoke:
            tr, va = tr[: 4 * cfg["batch_size"]], va[: 2 * cfg["batch_size"]]
        ds_tr, ds_va = TextDataset(subset(enc, tr), y[tr]), TextDataset(subset(enc, va), y[va])
        dl = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate,
                        num_workers=cfg["num_workers"], drop_last=len(tr) > cfg["batch_size"])
        model = Model(cfg["model"], n_out, cfg["pooling"], cfg["dropout"]).to(device)
        opt = torch.optim.AdamW(llrd_groups(model, cfg["lr"], cfg["head_lr"], cfg["llrd"], cfg["weight_decay"]))
        steps = math.ceil(len(dl) / cfg["grad_accum"]) * cfg["epochs"]
        sched = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_frac"] * steps), steps)
        use_bf16 = amp and torch.cuda.is_bf16_supported()
        scaler = torch.amp.GradScaler(enabled=amp and not use_bf16)
        eval_at = set(np.linspace(0, len(dl), cfg["evals_per_epoch"] + 1, dtype=int)[1:].tolist())
        best, best_pred = metric.worst(), None
        for epoch in range(cfg["epochs"]):
            model.train()
            for step, batch in enumerate(dl, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                with torch.autocast(device_type=device.type, enabled=amp,
                                    dtype=torch.bfloat16 if use_bf16 else torch.float16):
                    loss = crit(model(**batch).float(), labels) / cfg["grad_accum"]
                scaler.scale(loss).backward()
                if step % cfg["grad_accum"] == 0 or step == len(dl):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                if step in eval_at:
                    pred = predict(model, ds_va, collate, device, task, amp, cfg["batch_size"] * 4)
                    score = metric(y_metric[va], metric_input(task, metric, pred))
                    print(f"fold {fold} epoch {epoch} step {step}/{len(dl)} val {metric.name} {score:.5f}", flush=True)
                    if best_pred is None or metric.is_better(score, best):
                        best, best_pred = score, pred
                        torch.save(model.state_dict(), out_dir / f"fold{fold}.pt")
                    model.train()
        np.save(out_dir / f"fold{fold}_oof.npy", best_pred)
        np.save(out_dir / f"fold{fold}_idx.npy", va)
        if enc_test is not None and not a.smoke:
            model.load_state_dict(torch.load(out_dir / f"fold{fold}.pt", map_location=device))
            np.save(out_dir / f"fold{fold}_test.npy",
                    predict(model, TextDataset(enc_test, None), collate, device, task, amp, cfg["batch_size"] * 4))
        print(f"fold {fold} best {metric.name} {best:.5f}", flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if a.smoke:
        print("smoke run OK (not logged)")
        return
    done = [f for f in all_folds if (out_dir / f"fold{f}_oof.npy").exists()]
    if done != all_folds:
        print(f"folds done {done}/{all_folds}; the run that completes the set will log the experiment")
        return

    oof = np.zeros((len(train), n_out), dtype=np.float32)
    fold_scores = []
    for f in all_folds:
        idx, p = np.load(out_dir / f"fold{f}_idx.npy"), np.load(out_dir / f"fold{f}_oof.npy")
        oof[idx] = p
        fold_scores.append(metric(y_metric[idx], metric_input(task, metric, p)))
    cv = metric(y_metric, metric_input(task, metric, oof))
    test_pred = np.mean([np.load(out_dir / f"fold{f}_test.npy") for f in all_folds], axis=0) if test is not None else None
    rec = Ledger().log(cfg["name"], cv, fold_scores=fold_scores, model=cfg["model"], notes=cfg["notes"],
                       params={k: v for k, v in cfg.items() if k != "notes"},
                       oof=oof[:, 0] if n_out == 1 else oof,
                       test_pred=None if test_pred is None else (test_pred[:, 0] if n_out == 1 else test_pred),
                       runtime_s=round(time.time() - t0, 1))
    print(f"[{cfg['name']}] {metric.name} CV {cv:.5f} +/- {np.std(fold_scores):.5f} -> {rec['id']}")
    if test_pred is not None and Path(cfg["sample"]).exists():
        sample = pd.read_csv(cfg["sample"])
        key, cols = sample.columns[0], list(sample.columns[1:])
        pred = pd.DataFrame(index=test.index)
        if task == "multiclass" and metric.kind == "label":
            pred[cols[0]] = classes[test_pred.argmax(1)]
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
