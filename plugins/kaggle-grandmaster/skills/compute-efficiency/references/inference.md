# Inference under a time limit

Contents: [Budget and measurement](#budget-and-measurement) · [GPU inference](#gpu-inference) ·
[LLM inference](#llm-inference) · [CPU-only inference](#cpu-only-inference) ·
[Order of work in a kernel](#order-of-work-in-a-kernel)

## Budget and measurement

- Budget = runtime limit x 0.8. Measure on the competition's hardware with a hidden-test-sized input (copies of the
  public test, or the training set without labels).
- Time each stage: loading and preprocessing, each model, TTA, post-processing. Print the elapsed time in the kernel.
  On 4 CPU cores, preprocessing is often the slowest stage.
- Model loading, compiling and engine building are fixed costs. Extrapolate from a 5-10 % slice only after them.
- Locally, `python -m kgkit bench --run "python kernels/infer/inference_kernel.py"` times the code path and reports
  GPU utilisation. The real number comes from a Kaggle run.
- `inference_kernel.py` runs members in priority order, skips optional ones when the budget runs short, and always
  writes a valid submission.

## GPU inference

Images, 3D volumes and encoders.

- `torch.inference_mode()`, fp16 (`model.half()` or autocast), batches as large as memory allows, `channels_last`,
  and `cudnn.benchmark` for fixed shapes.
- Both T4s: one process per GPU over half of the test set, or one model per GPU. CZII CryoET (1st) and Image Matching
  2024 and 2025 (1st) split the work across the two T4s.
- Text: sort by length to minimise padding.
- Share preprocessing: decode, resize or tokenise once and feed every model from the same tensors. BirdCLEF+ 2025
  (2nd) computed the spectrograms once for all models.
- Fewer forward passes: LMSYS (1st) averaged the LoRA adapters and heads of its five fold models into one model.
  Weight averaging only works for models fine-tuned from the same checkpoint, so check the OOF score first.
- TTA only where the OOF shows a gain. Run flips as one batch with the originals.
- Exports, in order of effort:
  1. `torch.compile`: about 2x in SenNet (1st), which made 8x TTA at high resolution fit the limit.
  2. ONNX Runtime with the CUDA provider: about the speed of `torch.jit` in CZII CryoET (1st).
  3. TensorRT, through ONNX Runtime's `TensorrtExecutionProvider` (`trt_fp16_enable`, an engine cache) or
     `torch_tensorrt`. CZII CryoET (1st) got about 2x over `torch.jit`, with all models in one ONNX graph. An
     engine is tied to the GPU type and the TensorRT version: build it once in a Kaggle notebook on a T4, save it as
     a dataset, and load it in the submission (which saved them about 10 minutes per submission). BYU 2025 (3rd)
     also ran TensorRT engines.
- Check parity: the OOF score of the exported model against PyTorch on one fold, because fp16 exports can shift scores.

## LLM inference

- **vLLM on 2xT4:** `LLM(model, dtype="half", tensor_parallel_size=2, gpu_memory_utilization=0.90,
  max_model_len=<what you need>, enable_prefix_caching=True)`. For a model that fits one T4, two independent
  processes (one per GPU, each on half of the data) avoid the tensor-parallel traffic and usually give more
  throughput.
- **Quantisation to fit 9-14B models on T4s:**
  - GPTQ 8-bit: LMSYS (1st, Gemma-2 9B on 2xT4) and AIMO Progress Prize 1 (1st).
  - W8A8 SmoothQuant INT8 in LMDeploy on T4: MAP 2025 (1st).
  - AWQ int4: Eedi (1st) and Make Data Count (1st, a 32B model on vLLM). It is the fastest and smallest option, but
    AIMO Progress Prize 2 (1st) found int4 lost accuracy where int8 and fp8 did not. Measure on a validation set.
  - Calibrate on task data (Eedi 1st, AIMO Progress Prize 1 1st). FP8 needs Ada or Hopper GPUs, not T4.
- **Classification and multiple choice:** one forward pass that reads the logits of the answer tokens (or a
  classification head) instead of `generate()`, with the test sorted by length. Jigsaw 2025 (1st) did both.
- **Generation:** batch the requests, cap the tokens per problem, cache shared prompt prefixes (AIMO Progress Prize 3,
  1st) and use an 8-bit KV cache to fit more parallel samples (AIMO Progress Prize 2, 2nd: LMDeploy
  `quant_policy: 8`). Stop early when answers agree: AIMO Progress Prize 2 (1st) cancelled the remaining generations
  once 4 of the first 5 agreed. Speculative decoding with a draft head is another option (AIMO Progress Prize 2 1st,
  with TensorRT-LLM).
- **Engines:** vLLM is the default. LMDeploy's TurboMind engine gave AIMO Progress Prize 2 (2nd) more throughput and a
  faster start than vLLM. CTranslate2 int8 suits seq2seq models: Deep Past (1st) ran ByT5-XL on two T4s with it.
- **Start-up:** loading 10-30 GB of weights counts against the limit. Read the weight files once while other setup
  runs, so they are in the page cache when the engine loads them (AIMO Progress Prize 3, 1st).
- **Bigger is not always better:** in WSDM Cup (1st), a 72B AWQ model got through only 5 % of the samples with TTA
  in the time limit.

## CPU-only inference

- Kaggle CPU kernels have 4 cores. Use all of them: one process per core over chunks of the data
  (`ProcessPoolExecutor` or joblib) with `torch.set_num_threads(1)` in each, or one process with 4 intra-op threads.
  Measure both.
- OpenVINO on x86 CPUs: convert ONNX with `ovc model.onnx --compress_to_fp16` (BirdCLEF+ 2025, 2nd) and compile it with
  a fixed input shape (BirdCLEF 2024, 1st). INT8 post-training quantisation (NNCF) is faster again, but check the score.
- ONNX Runtime on the CPU: graph optimisations on (`ORT_ENABLE_ALL`), `intra_op_num_threads` set, and dynamic int8
  quantisation for transformers (`onnxruntime.quantization.quantize_dynamic`).
- Distil the ensemble into a smaller student when the budget is the binding constraint.
- Compute shared work (spectrograms, tokens, features) once for all models.

## Order of work in a kernel

1. Load the test data and preprocess it once for all models, on the CPU while the first model loads.
2. Run the models in priority order, each on its own GPU when there are two. Write intermediate predictions to
   `/kaggle/working`.
3. Track the elapsed time, skip optional members when the remaining budget is short, and always write a valid
   submission (the fallback is the best single model).
