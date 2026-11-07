# Text and LLMs: training speed

Contents: [Encoder fine-tuning](#encoder-fine-tuning) · [LLM fine-tuning](#llm-fine-tuning) ·
[Long documents](#long-documents) · [Inference](#inference)

## Encoder fine-tuning

DeBERTa-v3, RoBERTa, ModernBERT and similar models.

- Load the backbone with `.float()`, as the template does. transformers 5 keeps the checkpoint's dtype, and the
  DeBERTa-v3 checkpoints are fp16, which GradScaler refuses to unscale (training-loop.md, "fp32 master weights").
- Tokenise once, before the fold loop (the template does this). Cache the token ids on disk (npy, or an arrow
  dataset) when tokenising takes minutes.
- **Length-bucketed batches with dynamic padding.** Sort the training indices by length inside chunks of about 50
  batches, cut each chunk into batches, and shuffle the batches. On skewed length distributions this removes most of
  the padding, which can be half of the compute with plain shuffling. `train_transformer.py` does this by default
  (`length_buckets` in CONFIG; `--no-bucket` compares the two on one fold). Measured in a Kaggle T4 session
  (November 2026), DeBERTa-v3-xsmall on the AES 2.0 essays at `max_len` 1024: buckets cut padding from 40 % to 3 %
  of the tokens and trained 1.7x faster.
- `max_len` from the distribution: the 95-99th percentile of token lengths, not the model maximum. Attention cost
  grows with the square of the length.
- AMP: fp16 + GradScaler on T4, bf16 on Ampere+.
- SDPA: Hugging Face uses it by default for the architectures that support it (BERT, RoBERTa, ModernBERT and most
  decoder LLMs). DeBERTa-v2/v3 has its own disentangled attention and no SDPA path, so its speed comes from
  shorter sequences and bucketing.
- Fused AdamW on CUDA.
- For exploration screens, freezing the embeddings (or the lower layers) saves time and memory. Unfreeze for the
  finals, and compare on the same fold.
- Evaluate a few times per epoch on a length-sorted validation set (the template does this).
- Two T4s: one fold per GPU.

## LLM fine-tuning

7B-14B models on 2xT4, or larger off Kaggle.

- QLoRA: a 4-bit NF4 base (bitsandbytes), LoRA on all linear layers, `bnb_4bit_compute_dtype=torch.float16` on T4,
  gradient checkpointing, a paged 8-bit AdamW.
- Unsloth cuts memory and time for the models it supports. Jigsaw 2025 (1st) switched to it to fine-tune 14B models
  in 16 GB. ARC Prize 2025 (1st) ran per-puzzle LoRA fine-tuning with Unsloth and FlashAttention-2 (bf16, on Ampere+
  GPUs; on T4 use fp16).
- Classification with an LLM: a sequence-classification head, or the logits of the label tokens from one forward
  pass. Both avoid generation.
- Pack short examples into fixed-length sequences (TRL `packing=True`). AIMO Progress Prize 1 (1st) packed into 2048
  tokens with gradient checkpointing and DeepSpeed ZeRO-3, off Kaggle.
- On 2xT4, DDP (one 4-bit copy per GPU) is faster than `device_map="auto"`, which splits the layers so that one GPU
  waits while the other works. Split the layers only when the model does not fit one GPU.
- Train off Kaggle (rented A100/H100) when the rules allow: bf16, FlashAttention-2 and more memory make one such GPU
  hour worth many T4 hours. Ship the adapters, or merged and quantised weights, as a dataset.
- Distil a large teacher's soft labels into the model that fits the inference budget.

## Long documents

- Chunk with a stride and pool the chunk outputs (max or attention pooling), running the chunks as one batch.
- Head + tail truncation is a strong cheap baseline. Check it against chunking on one fold.

## Inference

See [inference.md](inference.md#llm-inference): one forward pass instead of `generate()` for classification, sorting
by length, quantisation measured against a validation set, vLLM or LMDeploy settings for T4, prefix caching, and
averaging fold adapters into one model (LMSYS 1st).
