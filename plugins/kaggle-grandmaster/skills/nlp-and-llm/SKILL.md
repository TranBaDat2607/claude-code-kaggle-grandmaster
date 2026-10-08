---
name: nlp-and-llm
description: Winning playbook for Kaggle NLP and LLM competitions — transformer fine-tuning (DeBERTa-v3 and friends), LLM fine-tuning with LoRA/QLoRA, LLM-as-solver (math/reasoning/agents), retrieval & RAG, synthetic data generation, distillation, efficient inference with vLLM under Kaggle time limits, and text-specific validation. Use for any text, chat, code, or LLM-related competition.
---

# NLP & LLM Playbook

## 1. Pick the paradigm

| Problem | Strong default |
|---|---|
| Text classification/regression, ≤ ~1k tokens, 1k–100k samples | DeBERTa-v3 (base/large) fine-tune, 5 folds |
| Token classification (NER, PII, span extraction) | DeBERTa-v3 token classification, long context via striding; threshold tuning |
| Long documents | Longformer/BigBird, or chunk + aggregate; or decoder LLMs with long context |
| Preference / pairwise judging, nuanced semantics | Fine-tune a 7–14B decoder LLM (Gemma/Qwen/Llama/Mistral family) with LoRA as a classifier/regressor head |
| Reasoning / math / code generation | Strongest open reasoning LLM that fits the hardware; prompt engineering, self-consistency (majority voting), tool use (Python execution), test-time compute scaling, fine-tuning on curated solutions (SFT, then RL such as GRPO when feasible) |
| Retrieval / matching | bi-encoder (e5/bge/gte) recall → cross-encoder or LLM re-rank |
| Generation evaluated by metric (e.g. ROUGE, similarity) | fine-tuned seq2seq/decoder + decoding search tuned to the metric |

Always check the rules about which pretrained models/APIs are allowed and model licences.

## 2. Encoder fine-tuning recipe (DeBERTa-v3)

- max_len from token-length percentiles (cover 95–99%); dynamic padding, sort-by-length batching.
- lr 1e-5–3e-5 (large: 1e-5), layer-wise LR decay 0.8–0.95, warmup 5–10%, cosine/linear decay,
  2–5 epochs, batch 8–32 (grad accumulation), weight decay 0.01, AMP, gradient clipping 1.0.
- Pooling: mean pooling or attention pooling often > CLS; multi-sample dropout head;
  re-initialise the last 1–2 layers on small data.
- Evaluate several times per epoch (best checkpoints often mid-epoch); AWP / FGM adversarial
  training and EMA for robustness; freeze embeddings for small data.
- Seeds vary a lot on small data — average 2–3 seeds per fold for final models.
- Template: `templates/train_transformer.py`.

## 3. LLM fine-tuning (7B–70B) on Kaggle-class hardware

- LoRA/QLoRA (PEFT) with r 16–64, alpha = 2r, dropout 0.05, target all linear projections;
  lr 1e-4–2e-4, 1–2 epochs, cosine, bf16 (on GPUs that support it) — fp16 on T4.
- Classification with an LLM: `AutoModelForSequenceClassification` with a score head, or
  next-token prediction of label tokens and read logits. Put the most informative text where
  it won't be truncated; truncate long fields individually.
- Train off-Kaggle (cloud A100/H100) when allowed, then upload LoRA adapters / merged weights
  as a Kaggle dataset; inference in-kernel with 4/8-bit quantisation (bitsandbytes, AWQ, GPTQ)
  on 2×T4 or with vLLM.
- Knowledge distillation: large teacher (e.g. 70B) soft labels → smaller student that fits the
  inference budget; typically beats training the student on hard labels.
- Synthetic data: generate extra labelled examples with a strong LLM (if rules allow); validate
  only on real data; deduplicate against test-like distributions to avoid optimistic CV.

## 4. LLM-as-solver competitions (math, reasoning, ARC-style, agents)

- Build a local evaluation set that mirrors the hidden test difficulty; never tune on
  a handful of public examples only.
- Inference engine: vLLM (tensor parallel across both GPUs), prefix caching, batched generation,
  max token budgets per problem; track wall-clock per item vs total time limit.
- Techniques: chain-of-thought / reasoning models, program-aided solving (generate Python,
  execute in a sandbox, feed back errors), self-consistency voting over N samples, answer
  normalisation and validation (e.g. integer answers mod constraints), early stopping when votes
  converge, difficulty-adaptive sample counts.
- Fine-tune on high-quality solution traces relevant to the domain; RL with verifiable rewards
  (GRPO-style) when compute allows.
- Time management code is part of the solution: hard per-question timeouts, graceful degradation.

## 5. Validation for text

- Group by author/prompt/essay set/topic/source when the test contains unseen ones.
- Check duplication & near-duplication (MinHash) between train and test, and within train.
- Tokenizer differences change truncation — re-check coverage when switching models.
- For generated/judge-style tasks, the metric may be noisy; use more folds and seeds.

## 6. Classic features still matter

For GBDT stackers on top of transformer OOFs: lengths, readability, counts of errors/typos,
TF-IDF + SVD, embedding similarities, LLM perplexity features. Blending transformer OOFs with a
GBDT over such features is a common winning pattern (e.g. essay scoring).

## 7. Inference efficiency (code competitions)

- Sort by length and batch dynamically; FP16; `torch.inference_mode()`; max_len per model.
- Use both T4 GPUs (data parallel by splitting the test set across two processes).
- Quantise LLMs (AWQ/GPTQ 4-bit) and use vLLM; cache tokenised inputs; avoid recomputation
  across folds by sharing tokenisation.
- Measure the whole pipeline on a hidden-test-sized dummy input before the deadline.
