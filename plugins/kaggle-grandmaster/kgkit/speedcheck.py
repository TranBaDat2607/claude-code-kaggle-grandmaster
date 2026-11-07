"""Static checks for slow patterns in training and inference scripts, and the compute defaults per domain.

Standard library only: the PostToolUse hook imports it when Claude writes model code in a workspace, and
``python -m kgkit speedcheck <file>`` runs the same checks on demand.

High precision over recall: each check fires only on a clear pattern that the compute-efficiency skill says to
change. ``warn`` findings cost real time on Kaggle hardware; ``info`` findings are options worth knowing.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# domain -> (reference file of the compute-efficiency skill, the defaults to apply in one line)
DOMAINS = {
    "vision": ("vision.md",
               "AMP (fp16 + GradScaler on T4/P100, bf16 on Ampere+), channels_last model and inputs, "
               "torch.backends.cudnn.benchmark = True for fixed input sizes, inputs cached once as uint8 at the "
               "training size (never decode 2-4K images or DICOM every epoch), num_workers = CPU cores / processes "
               "with persistent_workers, augmentation on the GPU when the loader is the bottleneck, one fold per GPU"),
    "text": ("text-and-llm.md",
             "AMP, tokenise once, length-bucketed batches with dynamic padding, max_len from the token-length "
             "distribution, fused AdamW, SDPA where the model supports it, one fold per GPU"),
    "llm": ("inference.md",
            "training: QLoRA/LoRA on all linear layers, fp16 on T4 (bf16 on Ampere+), gradient checkpointing; "
            "inference: one forward pass reading answer-token logits instead of generate() for classification, "
            "inputs sorted by length, vLLM (dtype='half' on T4) or LMDeploy, 8-bit GPTQ/W8A8 or AWQ measured on a "
            "validation set, prefix caching, both T4s"),
    "tabular": ("tabular.md",
                "float32/categorical dtypes, vectorised feature code (no row loops), XGBoost tree_method='hist', "
                "LightGBM num_threads = physical cores; on a GPU: XGBoost device='cuda', CatBoost task_type='GPU', "
                "cudf.pandas for groupby/merge-heavy features on millions of rows, cuML for kNN/SVR/RF"),
    "audio": ("audio-and-signal.md",
              "waveforms decoded once at the model's sample rate, spectrograms computed on the GPU inside the model, "
              "augmentation on the GPU batch; CPU-only inference: ONNX then OpenVINO fp16 with a fixed input shape, "
              "one spectrogram per window shared by all models, all CPU cores"),
}

_DOMAIN_RX = [
    ("audio", re.compile(r"^\s*(?:import|from)\s+(?:torchaudio|librosa|soundfile|nnAudio|audiomentations)\b", re.M)),
    ("llm", re.compile(r"^\s*(?:import|from)\s+(?:vllm|peft|bitsandbytes|unsloth|lmdeploy|trl)\b|"
                       r"AutoModelForCausalLM", re.M)),
    ("text", re.compile(r"^\s*(?:import|from)\s+(?:transformers|tokenizers|sentence_transformers)\b", re.M)),
    ("vision", re.compile(r"^\s*(?:import|from)\s+(?:timm|torchvision|albumentations|cv2|PIL|monai|"
                          r"segmentation_models_pytorch|ultralytics|pydicom|dicomsdl|kornia)\b", re.M)),
    ("tabular", re.compile(r"^\s*(?:import|from)\s+(?:lightgbm|xgboost|catboost|sklearn|cudf|cuml|polars)\b", re.M)),
]
_TORCH = re.compile(r"^\s*(?:import|from)\s+torch\b", re.M)
_TRAIN = re.compile(r"\.backward\(|\.fit\(|\b(?:lgb|lightgbm|xgb|xgboost|catboost)\.train\(|\b(?:SFT)?Trainer\(")
_INFER = re.compile(r"\.predict(?:_proba)?\(|inference_mode|\.generate\(|\bLLM\(|from_pretrained\(|torch\.load\(")
_AMP = re.compile(r"autocast|GradScaler|\bamp\b|bf16|fp16|bfloat16|float16|\.half\(|mixed_precision|precision\s*=")
_CONV = re.compile(r"\btimm\b|torchvision\.models|Conv[123]d|segmentation_models_pytorch|\bmonai\b|ultralytics")
_TREE = re.compile(r"""tree_method['"]?\s*[:=]\s*['"](exact|approx|gpu_hist)['"]""")
_SKLEARN_GPU = re.compile(r"\b(SVC|SVR|KNeighbors(?:Classifier|Regressor)|RandomForest(?:Classifier|Regressor)|"
                          r"NearestNeighbors)\(")
# Hugging Face model classes (not tokenizers, configs or processors), and LoRA/QLoRA setups whose frozen base is
# meant to stay in fp16 or 4-bit
_HF_MODEL = re.compile(r"Model|For[A-Z]")
_PEFT = re.compile(r"^\s*(?:import|from)\s+(?:peft|unsloth|bitsandbytes)\b|get_peft_model|LoraConfig|"
                   r"BitsAndBytesConfig|load_in_[48]bit|prepare_model_for_kbit_training|FastLanguageModel", re.M)


@dataclass
class Finding:
    code: str
    line: int
    level: str  # "warn" | "info"
    message: str


def domain(text: str) -> str | None:
    """vision / text / llm / tabular / audio from the imports, the first that matches in that order of priority."""
    for name, rx in _DOMAIN_RX:
        if rx.search(text):
            return name
    return None


def kind(text: str) -> str | None:
    """'train' when the code fits a model, 'infer' when it only runs one, else None."""
    if _TRAIN.search(text):
        return "train"
    if _INFER.search(text):
        return "infer"
    return None


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _first_line(text: str, rx: re.Pattern) -> int:
    m = rx.search(text)
    return _line_of(text, m.start()) if m else 1


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _receiver(node: ast.Call) -> str:
    """'AutoModel' for AutoModel.from_pretrained(...) or transformers.AutoModel.from_pretrained(...)."""
    v = node.func.value if isinstance(node.func, ast.Attribute) else None
    return v.id if isinstance(v, ast.Name) else v.attr if isinstance(v, ast.Attribute) else ""


def _kw(node: ast.Call, name: str):
    for k in node.keywords:
        if k.arg == name:
            return k.value
    return None


def _is_const(node, *values) -> bool:
    return isinstance(node, ast.Constant) and node.value in values


def check(text: str) -> list:
    """Findings for slow patterns in a Python training or inference script, ordered by line."""
    out: list = []
    torch = bool(_TORCH.search(text))
    k = kind(text)
    dom = domain(text)
    backward = re.search(r"\.backward\(", text)

    if torch and backward and not _AMP.search(text):
        out.append(Finding("NO_AMP", _line_of(text, backward.start()), "warn",
                           "the training loop runs in fp32. fp16 autocast + GradScaler on T4/P100 (bf16 on Ampere+) "
                           "is usually about 2x faster and halves activation memory"))
    if torch and backward and dom == "vision" and _CONV.search(text) and "cudnn.benchmark" not in text \
            and not re.search(r"cudnn\.deterministic\s*=\s*True", text):
        out.append(Finding("NO_CUDNN_BENCHMARK", _line_of(text, backward.start()), "warn",
                           "convolutions without torch.backends.cudnn.benchmark = True. With fixed input sizes it "
                           "picks the fastest kernels once (often +10-20 %)"))
    if re.search(r"cudnn\.deterministic\s*=\s*True", text):
        out.append(Finding("CUDNN_DETERMINISTIC", _first_line(text, re.compile(r"cudnn\.deterministic")), "info",
                           "deterministic cuDNN kernels are slower: keep them for debugging, not for every run"))
    for m in _TREE.finditer(text):
        method = m.group(1)
        msg = ("gpu_hist is deprecated since XGBoost 2.0: use tree_method='hist' with device='cuda'"
               if method == "gpu_hist" else
               f"tree_method='{method}' is much slower than 'hist' (add device='cuda' on a GPU)")
        out.append(Finding("XGB_TREE_METHOD", _line_of(text, m.start()), "warn", msg))
    if k == "train" and dom == "tabular":
        m = _SKLEARN_GPU.search(text)
        if m:
            out.append(Finding("SKLEARN_ON_GPU", _line_of(text, m.start()), "info",
                               f"{m.group(1)} is slow on 100k+ rows on the CPU. cuML runs it on a GPU unchanged "
                               "(`python -m cuml.accel script.py`; Linux/WSL2, T4 yes, P100 no)"))

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        tree = None
    if tree is not None:
        row_loops = []
        # a model trained in the checkpoint's dtype: training code, not a LoRA/QLoRA setup (frozen fp16/4-bit base)
        hf_train = bool(backward) and not _PEFT.search(text)
        upcast = {id(n.func.value) for n in ast.walk(tree)
                  if hf_train and isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and isinstance(n.func.value, ast.Call)
                  and (n.func.attr == "float" or (n.func.attr == "to" and "float32" in ast.unparse(n)))}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name == "DataLoader" and not any(kw.arg is None for kw in node.keywords):
                nw = _kw(node, "num_workers")
                if nw is None or _is_const(nw, 0):
                    out.append(Finding("LOADER_IN_MAIN_PROCESS", node.lineno, "warn",
                                       "DataLoader without workers: batches are built in the training process while "
                                       "the GPU waits. Set num_workers to CPU cores / processes, with "
                                       "persistent_workers=True and pin_memory=True"))
                elif _kw(node, "persistent_workers") is None:
                    out.append(Finding("WORKERS_RESTART", node.lineno, "info",
                                       "persistent_workers=True keeps the loader workers between epochs (their "
                                       "start-up is slow, especially with spawn on Windows)"))
            elif name == "iterrows" or (name == "apply" and _is_const(_kw(node, "axis"), 1, "columns")):
                row_loops.append(node.lineno)
            elif name == "is_bf16_supported" and not node.args and not node.keywords:
                out.append(Finding("BF16_EMULATED", node.lineno, "warn",
                                   "torch.cuda.is_bf16_supported() is True on a T4 or P100 too, where bf16 is emulated "
                                   "and slow. Choose bf16 with torch.cuda.get_device_capability()[0] >= 8 (or "
                                   "is_bf16_supported(including_emulation=False)), else fp16 + GradScaler"))
            elif name == "from_pretrained" and hf_train and _HF_MODEL.search(_receiver(node)) \
                    and id(node) not in upcast and not any(k.arg in (None, "dtype", "torch_dtype")
                                                           for k in node.keywords):
                out.append(Finding("HF_CHECKPOINT_DTYPE", node.lineno, "warn",
                                   "transformers 5 loads a model in its checkpoint's dtype (fp16 for DeBERTa-v3, "
                                   "bf16 for most LLMs). Under GradScaler fp16 weights fail ('Attempting to unscale "
                                   "FP16 gradients'), and bf16 master weights drop small updates. For full "
                                   "fine-tuning load fp32: from_pretrained(...).float(), or dtype=torch.float32; "
                                   "autocast still runs the math in fp16/bf16"))
        if row_loops:
            more = f" (and {len(row_loops) - 1} more)" if len(row_loops) > 1 else ""
            out.append(Finding("ROW_LOOP", min(row_loops), "warn",
                               f"row-wise Python loop over a DataFrame{more}: vectorise it (column arithmetic, "
                               "np.where, groupby().transform, merge). Row loops are slow on any hardware"))
    if torch and k == "infer" and not re.search(r"inference_mode|no_grad|set_grad_enabled\(\s*False", text) \
            and re.search(r"\.eval\(\)|torch\.load\(|from_pretrained\(", text) \
            and not re.search(r"\bvllm\b|\bLLM\(|pipeline\(", text):
        out.append(Finding("NO_INFERENCE_MODE", _first_line(text, re.compile(r"\.eval\(\)|torch\.load\(|from_pretrained\(")),
                           "warn", "prediction without torch.inference_mode() (or no_grad): autograd keeps "
                                   "activations, costing memory and time"))
    return sorted(out, key=lambda f: (f.line, f.code))


def format_findings(path: str, findings: list, levels=("warn", "info")) -> list:
    return [f"{path}:{f.line}: [{f.level}] {f.code}: {f.message}" for f in findings if f.level in levels]


def main_speedcheck(a) -> None:
    """``kgkit speedcheck FILE...``: slow patterns and the domain's compute defaults (advisory; exits 0)."""
    from pathlib import Path

    from . import cli

    for name in a.files:
        p = Path(name)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"{name}: cannot read ({e})")
            continue
        dom, k = domain(text), kind(text)
        found = check(text)
        what = f"{k or 'no model'} code" + (f", {dom}" if dom else "")
        print(f"{name}: {what}")
        for line in format_findings(name, found):
            print(f"  {line}")
        if not found:
            print("  no slow patterns found")
        if dom and k:
            ref, defaults = DOMAINS[dom]
            print(f"  defaults for {dom}: {defaults}")
            skill = cli._plugin_file("skills/compute-efficiency/references/" + ref)
            print(f"  reference: {skill.as_posix() if skill else 'skill compute-efficiency, references/' + ref}")
