"""Transformer fine-tuning template for text classification / regression (Hugging Face).

Copy to src/, edit CONFIG, then:

    python src/train_transformer.py --smoke
    python src/train_transformer.py --bench          # data loader vs GPU step vs real loop, padding share
    python src/train_transformer.py --name debv3b_512 --model microsoft/deberta-v3-base --max-len 512
    CUDA_VISIBLE_DEVICES=0 python src/train_transformer.py --name x --folds 0 2 4 &   # split folds across GPUs
    CUDA_VISIBLE_DEVICES=1 python src/train_transformer.py --name x --folds 1 3
    # on Kaggle GPUs (quota-aware, resumable, both T4s busy): see the kaggle-gpu skill
    python -m kgkit gpu build --name x --hours 8 --cmd "python src/train_transformer.py --name x --folds {fold}"

Recipe: mean/attention pooling head, layer-wise LR decay, linear warmup + cosine decay, AMP,
gradient accumulation, evaluation several times per epoch (progress and pruning),
length-sorted dynamic-padding inference. task: regression | binary | multiclass | multilabel.
Honest OOF (#109): each fold keeps its final weights (select="last"). Keeping the evaluation that scored best on the
fold it then predicts (--select best, the old behaviour) picks the checkpoint on those rows and makes the OOF
optimistic. Choose the number of epochs once, from the val scores of all folds of an earlier run.
Early stopping, done legally (#119): --es-holdout 0.05 holds 5% of each fold's training part out of training
(stratified, or whole groups with group_col), keeps the evaluation that scores best on it and stops after es_patience
evaluations without a gain there. The scored fold chooses nothing, so the OOF stays honest.
Pruning (kgkit.budget, `gpu build --prune-against`): the learning curve records each epoch's last evaluation. With
select="last" a run also stops when that score trails the control's and has fallen below the run's own peak for two
epochs; --select best judges the best evaluation only.
Speed defaults (skill compute-efficiency): text tokenised once, length-bucketed training batches with dynamic
padding (--no-bucket for plain shuffled batches), fused AdamW on CUDA, pinned memory and persistent workers.
Set max_len from the token-length distribution (95-99th percentile), not the model maximum.
For offline Kaggle kernels point --model at a local folder (save_pretrained output).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import kgkit  # noqa: F401  (a copy in third_party/: `python -m kgkit vendor third_party/`)
except ImportError:  # <workspace>/third_party/kgkit, else the plugin's (KGKIT_HOME, set by its session hook)
    _tp = Path(__file__).resolve().parents[1] / "third_party"
    sys.path.insert(0, str(_tp) if (_tp / "kgkit").is_dir() else os.environ.get("KGKIT_HOME", ""))
from kgkit import bench as B
from kgkit import metrics as M
from kgkit import state as S
from kgkit.budget import TrainBudget
from kgkit.experiment import Ledger, seed_everything
try:  # a kgkit vendored before #127 has no report_comparison: the run then ends without the verdict
    from kgkit.experiment import report_comparison
except ImportError:
    report_comparison = None

CONFIG = dict(
    name="deberta_baseline",
    train="data/train.csv",
    test="data/test.csv",
    sample="data/sample_submission.csv",
    folds=os.environ.get("KG_FOLDS_FILE", "data/folds.csv"),  # kgkit recheck points this at a re-drawn split
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
    select="last",               # last: final weights | best: best evaluation on the scored fold (optimistic OOF)
                                 # | holdout: best evaluation on a slice held out of the training part (--es-holdout)
    es_holdout=0.0,              # holdout: share of each fold's training part held out to choose the checkpoint (0.05)
    es_patience=2,               # holdout: stop after this many evaluations with no gain on it (0: never early)
    checkpoint_every=0,          # N > 0: a resume checkpoint every N epochs, so a killed fold resumes there
                                 # (0: only before a session deadline; #125)
    group_col=None,              # holdout: column whose groups (authors, prompts) stay whole in the held-out slice
    grad_clip=1.0,
    num_workers=2,
    length_buckets=True,         # training batches of similar length: much less padding (--no-bucket: off)
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


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batches of similar token length with a random order: shuffle, sort inside chunks of `chunk` batches, cut
    into batches, shuffle the batches. Most of the padding goes while every epoch stays random."""

    def __init__(self, lengths, batch_size, drop_last, seed, chunk=50):
        self.lengths, self.bs, self.drop_last = np.asarray(lengths), batch_size, drop_last
        self.span, self.seed, self.epoch = batch_size * chunk, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx, batches = rng.permutation(len(self.lengths)), []
        for s in range(0, len(idx), self.span):
            part = idx[s: s + self.span]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[i: i + self.bs] for i in range(0, len(part), self.bs)]
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.bs]
        rng.shuffle(batches)
        return iter([b.tolist() for b in batches])

    def __len__(self):
        n, full = len(self.lengths), (len(self.lengths) // self.span) * (self.span // self.bs)
        rest = n % self.span
        return full + (rest // self.bs if self.drop_last else -(-rest // self.bs))


class Model(nn.Module):
    def __init__(self, name, n_out, pooling, dropout):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(name)
        for k in ("hidden_dropout_prob", "attention_probs_dropout_prob"):
            if hasattr(cfg, k) and dropout == 0.0:
                setattr(cfg, k, 0.0)
        # fp32 master weights; autocast runs the math in fp16/bf16. transformers 5 loads the checkpoint's own dtype
        # (fp16 for DeBERTa-v3), which GradScaler refuses ("Attempting to unscale FP16 gradients").
        self.backbone = AutoModel.from_pretrained(name, config=cfg).float()
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


def holdout_split(tr, frac, seed, fold, strat=None, groups=None):
    """(fit rows, held-out rows) of a fold's training part ``tr`` for select="holdout": the held-out rows choose the
    checkpoint and stop training early, so the fold being scored chooses nothing (#109, #119). Whole groups when
    ``groups`` is given, else stratified by ``strat`` (class labels; None draws uniformly). The same seed and fold
    draw the same rows, so a resumed fold trains on the same split."""
    rng = np.random.default_rng([int(seed), int(fold), 119])
    hold = np.zeros(len(tr), dtype=bool)
    if groups is not None:
        g = np.unique(groups[tr])
        take = rng.permutation(g)[: max(1, int(round(frac * len(g))))]
        hold = np.isin(groups[tr], take)
    else:
        keys = strat[tr] if strat is not None else np.zeros(len(tr))
        for c in np.unique(keys):
            idx = np.flatnonzero(keys == c)
            hold[rng.permutation(idx)[: int(round(frac * len(idx)))]] = True
        if not hold.any():
            hold[rng.integers(len(tr))] = True
    if hold.all():
        raise SystemExit("--es-holdout would hold out the whole training part: lower it")
    return tr[~hold], tr[hold]


def holdout_config(cfg, select_arg=None):
    """``select`` and ``es_holdout`` made consistent: an es_holdout share switches select to holdout, and holdout
    without a share holds out 5%. Off, the holdout settings are cleared so the ledger's params don't show them."""
    cfg = dict(cfg)
    frac = float(cfg.get("es_holdout") or 0.0)
    if frac and select_arg == "best":
        raise SystemExit("--es-holdout chooses the checkpoint on the held-out slice; it does not combine with "
                         "--select best")
    if frac:
        cfg["select"] = "holdout"
    if cfg["select"] == "holdout":
        cfg["es_holdout"] = frac or 0.05
        if not 0 < cfg["es_holdout"] < 0.5:
            raise SystemExit(f"--es-holdout {cfg['es_holdout']}: give a share of the training part in (0, 0.5)")
        cfg["es_patience"] = max(0, int(cfg.get("es_patience") or 0))
    else:
        cfg["es_holdout"], cfg["es_patience"] = 0.0, None
    return cfg


def metric_input(task, metric, pred):
    if metric.kind == "label" and task == "multiclass":
        return pred.argmax(1)
    if task in ("binary", "regression") and pred.ndim == 2 and pred.shape[1] == 1:
        return pred[:, 0]
    return pred


def bench(steps, loader, train_step, device, cfg):
    """Samples/s of the data loader alone, of train_step on batches already on the device, and of the real loop, plus
    the share of padding tokens; kgkit.bench prints the verdict and the speed-up against the last bench. Batch lengths
    vary, so the step phase cycles over a pool of real batches rather than repeating one."""
    it = B.forever(loader)
    pool = [{k: v.to(device, non_blocking=True) for k, v in next(it).items()} for _ in range(max(2, min(16, steps)))]
    turn = {"i": 0}
    tokens = {"pad": 0, "all": 0}

    def data_step():
        b = next(it)
        m = b["attention_mask"]
        tokens["pad"] += int((m == 0).sum())
        tokens["all"] += m.numel()
        return len(m)

    def compute_step():
        b = pool[turn["i"] % len(pool)]
        turn["i"] += 1
        train_step(dict(b))
        return len(b["input_ids"])

    def full_step():
        b = {k: v.to(device, non_blocking=True) for k, v in next(it).items()}
        train_step(b)
        return len(b["input_ids"])

    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    rates = B.measure({"data": data_step, "compute": compute_step, "full": full_step}, n=steps,
                      sync=torch.cuda.synchronize if cuda else None)
    extra = {"device": torch.cuda.get_device_name() if cuda else "cpu (rates are for this CPU, not a GPU)",
             "padding": f"{100 * tokens['pad'] / max(1, tokens['all']):.0f}% of the tokens in training batches",
             "step phase": f"cycles over {len(pool)} batches already on the device (their lengths vary)"}
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        extra["peak GPU memory"] = f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB of {total:.0f} MB"
    config = {k: cfg[k] for k in ("model", "max_len", "batch_size", "num_workers", "length_buckets")}
    config.update(device=extra["device"],
                  amp=("bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16") if cuda else False)
    B.report(cfg["name"], rates, config=config, extra=extra, kind="train_transformer")


def main():
    ap = argparse.ArgumentParser()
    for k, t in (("name", str), ("model", str), ("max_len", int), ("batch_size", int), ("epochs", int),
                 ("lr", float), ("notes", str)):
        ap.add_argument("--" + k.replace("_", "-"), type=t)
    ap.add_argument("--folds", type=int, nargs="+")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--assemble", action="store_true", help="no training: assemble finished folds and log")
    ap.add_argument("--track", help="model line this run belongs to (gbdt, cnn, vit-l...): compared with that line's accepted baseline, not the main one; with exp finish ... baseline it starts the line")
    ap.add_argument("--parent", help="ledger id this experiment changes (the accepted baseline, usually); the "
                    "run ends with the paired comparison against it")
    ap.add_argument("--bench", type=int, nargs="?", const=30, metavar="STEPS",
                    help="time the data loader alone, the training step alone and the real loop on the first "
                         "fold (default 30 steps each); trains and logs nothing")
    ap.add_argument("--no-bucket", action="store_true", help="plain shuffled batches instead of length buckets")
    ap.add_argument("--select", choices=["last", "best", "holdout"],
                    help="last (default): each fold's final weights; best: the evaluation that scores best on the fold "
                         "it predicts (optimistic OOF, the old behaviour); holdout: the evaluation that scores best on "
                         "a slice held out of the training part (--es-holdout)")
    ap.add_argument("--es-holdout", type=float, metavar="FRAC",
                    help="hold FRAC (e.g. 0.05) of each fold's training part out of training to choose the checkpoint "
                         "and stop early: select=holdout, an honest OOF")
    ap.add_argument("--es-patience", type=int, metavar="N",
                    help="select=holdout: stop after N evaluations without a gain on the held-out slice (0: never)")
    ap.add_argument("--checkpoint-every", type=int, metavar="N",
                    help="save a resume checkpoint every N epochs (default 0: off), so a fold killed with its "
                         "session resumes mid-schedule when the same command runs again")
    a = ap.parse_args()
    cfg = dict(CONFIG)
    for k in ("name", "model", "max_len", "batch_size", "epochs", "lr", "notes", "select", "es_holdout",
              "es_patience", "checkpoint_every"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.no_bucket:
        cfg["length_buckets"] = False
    cfg = holdout_config(cfg, a.select)

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
    if a.bench:
        run_folds = run_folds[:1]
    crit = loss_fn(task)
    t0 = time.time()
    stopped = None  # "timeout" | "pruned": the session deadline or the baseline curve ended this run early
    # who runs this, for a later session (#125, `kgkit grind status`): is the process alive, which folds are left
    run_file = None if (a.smoke or a.bench or a.assemble) else out_dir / f"run_{os.getpid()}.json"
    run_info = {"pid": os.getpid(), "host": socket.gethostname(), "cmd": " ".join(["python", *sys.argv]),
                "started": time.time(), "heartbeat": time.time(), "folds": [int(f) for f in run_folds],
                "all_folds": [int(f) for f in all_folds], "finished": False}

    def beat(**info):
        if run_file is not None:
            run_info.update(heartbeat=time.time(), **info)
            try:
                run_file.write_text(json.dumps(run_info), encoding="utf-8")
            except OSError:
                pass

    beat()
    if cfg["select"] == "best":
        print("note: --select best keeps the evaluation that scores best on the fold it predicts: the OOF is "
              "optimistic", flush=True)
    holdout = cfg["select"] == "holdout"
    if holdout:
        print(f"select holdout: {cfg['es_holdout']:.0%} of each fold's training part is held out of training to choose "
              f"the checkpoint (early stop after {cfg['es_patience'] or 'no'} evaluations without a gain there)",
              flush=True)
    strat = y if task == "multiclass" else (y[:, 0] if task == "binary" else None)
    groups = train[cfg["group_col"]].to_numpy() if cfg["group_col"] else None

    def subset(e, idx):
        return {k: [v[i] for i in idx] for k, v in e.items()}

    for fold in ([] if a.assemble else run_folds):
        folds_arr = train["fold"].to_numpy()
        tr, va = np.flatnonzero(folds_arr != fold), np.flatnonzero(folds_arr == fold)
        if a.smoke:
            tr, va = tr[: 4 * cfg["batch_size"]], va[: 2 * cfg["batch_size"]]
        es, ds_es = None, None
        if holdout:  # the rows that choose the checkpoint come from the training part, never from the scored fold
            tr, es = holdout_split(tr, cfg["es_holdout"], cfg["seed"], fold, strat, groups)
            ds_es = TextDataset(subset(enc, es), y[es])
        ds_tr, ds_va = TextDataset(subset(enc, tr), y[tr]), TextDataset(subset(enc, va), y[va])
        loader_kw = dict(collate_fn=collate, num_workers=cfg["num_workers"], pin_memory=amp,
                         persistent_workers=cfg["num_workers"] > 0)
        drop = len(tr) > cfg["batch_size"]
        if cfg["length_buckets"]:
            lengths = [len(x) for x in ds_tr.enc["input_ids"]]
            dl = DataLoader(ds_tr, batch_sampler=LengthBucketSampler(lengths, cfg["batch_size"], drop,
                                                                     cfg["seed"] + fold), **loader_kw)
        else:
            dl = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, drop_last=drop, **loader_kw)
        model = Model(cfg["model"], n_out, cfg["pooling"], cfg["dropout"]).to(device)
        opt = torch.optim.AdamW(llrd_groups(model, cfg["lr"], cfg["head_lr"], cfg["llrd"], cfg["weight_decay"]),
                                **({"fused": True} if device.type == "cuda" else {}))
        steps = math.ceil(len(dl) / cfg["grad_accum"]) * cfg["epochs"]
        sched = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_frac"] * steps), steps)
        # bf16 only with bf16 tensor cores (compute capability 8.0+). torch.cuda.is_bf16_supported() also says True on a
        # T4 or P100, which emulate bf16 slowly: fp16 + GradScaler is their fast path
        use_bf16 = amp and torch.cuda.get_device_capability(device)[0] >= 8
        scaler = torch.amp.GradScaler(enabled=amp and not use_bf16)
        eval_at = set(np.linspace(0, len(dl), cfg["evals_per_epoch"] + 1, dtype=int)[1:].tolist())
        # best: best val score so far (progress, pruning); kept/kept_pred: the evaluation that becomes the OOF
        best, kept, kept_pred, start_epoch = metric.worst(), metric.worst(), None, 0
        es_best, es_bad, early = None, 0, None  # holdout: best score on the held-out slice, evals since, early stop
        # pruning judges the score this run keeps: its last evaluation, or with --select best its best one (#119)
        budget = TrainBudget(out_dir, fold, cfg["epochs"], metric.greater_is_better,
                             mode="best" if cfg["select"] == "best" else "last")
        if a.bench:
            def bench_step(batch):  # one optimiser step per batch, the cost of a training step
                labels = batch.pop("labels")
                with torch.autocast(device_type=device.type, enabled=amp,
                                    dtype=torch.bfloat16 if use_bf16 else torch.float16):
                    loss = crit(model(**batch).float(), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            model.train()
            bench(a.bench, dl, bench_step, device, cfg)
            return
        resume = out_dir / f"fold{fold}_resume.pt"
        if resume.exists() and not a.smoke:  # continue a fold the previous session checkpointed
            ck = torch.load(resume, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            best, start_epoch = ck["best"], ck["epoch"]
            kept, kept_pred = ck.get("kept", ck["best"]), ck.get("kept_pred", ck["best_pred"])
            es_best, es_bad = ck.get("es_best"), ck.get("es_bad", 0)
            print(f"fold {fold}: resumed at epoch {start_epoch} (best {best:.5f})", flush=True)
        def save_resume(epoch):  # what a resumed fold needs: the session deadline and --checkpoint-every
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best, "best_pred": kept_pred,
                    "kept": kept, "kept_pred": kept_pred, "es_best": es_best, "es_bad": es_bad}, resume)

        for epoch in range(start_epoch, cfg["epochs"]):
            budget.start_epoch()
            model.train()
            for step, batch in enumerate(dl, 1):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
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
                    if metric.is_better(score, best):
                        best = score
                    if holdout:  # the held-out slice of the training part chooses the checkpoint
                        es_pred = predict(model, ds_es, collate, device, task, amp, cfg["batch_size"] * 4)
                        es_score = metric(y_metric[es], metric_input(task, metric, es_pred))
                        print(f"fold {fold} epoch {epoch} step {step}/{len(dl)} holdout {metric.name} "
                              f"{es_score:.5f}", flush=True)
                        keep = es_best is None or metric.is_better(es_score, es_best)
                        es_best, es_bad = (es_score, 0) if keep else (es_best, es_bad + 1)
                    else:
                        keep = cfg["select"] == "last" or kept_pred is None or metric.is_better(score, kept)
                    if keep:
                        kept, kept_pred = score, pred
                        torch.save(model.state_dict(), out_dir / f"fold{fold}.pt")
                    model.train()
                    if (holdout and cfg["es_patience"] and es_bad >= cfg["es_patience"]
                            and not (epoch + 1 == cfg["epochs"] and step == len(dl))):
                        early = f"epoch {epoch} step {step}"
                        break
            # the curve gets this epoch's last evaluation, the score a run that keeps its final weights would keep:
            # a run that peaks and then declines shows it (#119); TrainBudget tracks the best one itself
            action = "continue" if a.smoke else budget.end_epoch(epoch, score)
            if action == "timeout":
                save_resume(epoch)
                budget.write_status("timeout", next_epoch=epoch + 1)
                print(f"fold {fold}: session deadline near - checkpointed after epoch {epoch}; resume later", flush=True)
            elif action == "prune":
                budget.write_status("pruned", epoch=epoch)
                print(f"fold {fold}: PRUNED at epoch {epoch} - {budget.prune_reason}", flush=True)
            if action != "continue":
                stopped = "timeout" if action == "timeout" else "pruned"
                break
            beat(fold=int(fold), epoch=int(epoch))
            if (cfg["checkpoint_every"] and not a.smoke and (epoch + 1) % cfg["checkpoint_every"] == 0
                    and epoch + 1 < cfg["epochs"]):
                save_resume(epoch)  # killed after this, the same command resumes here
            if early is not None:
                print(f"fold {fold}: early stop at {early} - no gain on the held-out slice for {es_bad} evaluations; "
                      "keeps the best of them", flush=True)
                break
        if stopped:
            break
        np.save(out_dir / f"fold{fold}_oof.npy", kept_pred)
        np.save(out_dir / f"fold{fold}_idx.npy", va)
        if enc_test is not None and not a.smoke:
            model.load_state_dict(torch.load(out_dir / f"fold{fold}.pt", map_location=device))
            np.save(out_dir / f"fold{fold}_test.npy",
                    predict(model, TextDataset(enc_test, None), collate, device, task, amp, cfg["batch_size"] * 4))
        if not a.smoke:
            budget.best = kept
            budget.write_status("done", **({"early_stop": early} if early is not None else {}))
            resume.unlink(missing_ok=True)
        print(f"fold {fold} {cfg['select']} {metric.name} {kept:.5f} (best evaluation {best:.5f})", flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    beat(finished=True, outcome=stopped or "done")
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

    oof = np.zeros((len(train), n_out), dtype=np.float32)
    fold_scores = []
    for f in all_folds:
        idx, p = np.load(out_dir / f"fold{f}_idx.npy"), np.load(out_dir / f"fold{f}_oof.npy")
        oof[idx] = p
        fold_scores.append(metric(y_metric[idx], metric_input(task, metric, p)))
    cv = metric(y_metric, metric_input(task, metric, oof))
    test_pred = np.mean([np.load(out_dir / f"fold{f}_test.npy") for f in all_folds], axis=0) if test is not None else None
    rec = Ledger().log(cfg["name"], cv, fold_scores=fold_scores, model=cfg["model"], notes=cfg["notes"],
                       parent=a.parent, track=a.track,
                       params={k: v for k, v in cfg.items() if k != "notes"},
                       oof=oof[:, 0] if n_out == 1 else oof,
                       test_pred=None if test_pred is None else (test_pred[:, 0] if n_out == 1 else test_pred),
                       runtime_s=round(time.time() - t0, 1),
                       oof_selection="scored-fold" if cfg["select"] == "best" else "independent")
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
    if report_comparison is not None:  # #127: the verdict against --parent (else the accepted baseline)
        report_comparison(rec, parent=a.parent, truth=f"{cfg['train']}:{','.join(targets)}")


if __name__ == "__main__":
    main()
