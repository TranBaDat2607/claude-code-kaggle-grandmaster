# Evidence: what winning solutions did for speed

Each row is a technique a top team used, from their write-up or code. Write-ups were read through the
first-place-solution index at github.com/juliusuhlmann/kaggle-first-place-solutions. Code was read in each team's
public repository. The place is the team's final private-leaderboard rank.

Contents: [Images and 3D](#images-and-3d) · [Text and LLMs](#text-and-llms) · [Tabular](#tabular) ·
[Audio](#audio) · [Simulation and other](#simulation-and-other)

## Images and 3D

| Competition | Place | Technique | Source |
|---|---|---|---|
| CZII CryoET Object Identification (2025) | 1st | All models in one ONNX graph, run through ONNX Runtime's TensorRT provider: about 2x over `torch.jit`. The engine was built on a Kaggle T4 and loaded from a dataset (about 10 min saved per submission). Both T4s in parallel | write-up; github.com/BloodAxe/Kaggle-2024-CryoET (WRITEUP.md) |
| SenNet + HOA (2024) | 1st | `torch.compile` gave about 2x and made 8x flip/transpose TTA at high resolution fit the limit | write-up |
| BYU Flagellar Motors (2025) | 1st | MixUp applied to the batch inside the model (on the GPU); heavy 3D augmentation | github.com/brendanartley/BYU-competition |
| BYU Flagellar Motors (2025) | 3rd | TensorRT engines for inference, fused optimisers, gradient checkpointing | github.com/dangnh0611/kaggle_byu |
| RSNA Abdominal Trauma (2023) | 1st | DICOM decoded with `dicomsdl` and cached as npy before training, preprocessing in parallel processes | github.com/Nischaydnk/RSNA-2023-1st-place-solution |
| RSNA Intracranial Aneurysm (2025) | 7th | nnU-Net with `torch.compile` and `cudnn.benchmark` on by default | github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution |
| UBC-OCEAN (2024) | 1st | At most 200 tissue patches per slide; two slides at a time with Ray processes on a P100 | write-up |
| Image Matching Challenge (2024) | 1st | Keypoints and descriptors cached per image, mixed precision, both T4s running SfM in parallel | write-up |
| Image Matching Challenge (2025) | 1st | Compiled CUDA kernels, attention swapped for `scaled_dot_product_attention`, AMP fixed, scene subsets split across the two T4s | write-up |
| ISIC 2024 | 2nd | `torch.compile` of the model and its EMA copy | github.com/uchiyama33/isic-2024-2nd-place |
| Drawing with LLMs (2025) | 1st | Tiny autoencoder VAE, DPM-Solver with 12 steps, DeepCache, the work split across two T4s | write-up |

## Text and LLMs

| Competition | Place | Technique | Source |
|---|---|---|---|
| LMSYS Chatbot Arena (2024) | 1st | Gemma-2 9B with LoRA merged, quantised to 8-bit GPTQ to run on 2xT4 in the limit; the fold models' LoRA adapters averaged into one model; TTA preferred over a longer context | write-up |
| Jigsaw Agile Community Rules (2025) | 1st | Test sorted by length, one forward pass reading the last token's logits instead of `generate()`; Unsloth to fine-tune 14B models in 16 GB | write-up |
| MAP Charting Student Math Misunderstandings (2025) | 1st | SmoothQuant W8A8 INT8 kernels in LMDeploy on T4 GPUs | write-up |
| Eedi Mining Misconceptions (2024) | 1st | AutoAWQ quantisation calibrated on task data | write-up |
| AI Mathematical Olympiad, Progress Prize 1 (2024) | 1st | 8-bit GPTQ calibrated on training data to fit 2xT4; vLLM; full fine-tune off Kaggle with packing, gradient checkpointing and DeepSpeed ZeRO-3 | write-up; github.com/project-numina/aimo-progress-prize |
| AI Mathematical Olympiad, Progress Prize 2 (2025) | 1st | TensorRT-LLM with FP8 weights and a ReDrafter speculative-decoding head; 12 asynchronous generations per problem, cancelled when 4 of the first 5 agreed; int4 was faster but lost accuracy where int8 and fp8 did not | write-up |
| AI Mathematical Olympiad, Progress Prize 2 (2025) | 2nd | LMDeploy TurboMind (more throughput and a faster start than vLLM), AWQ, 8-bit KV cache (`quant_policy: 8`), prefix caching | github.com/imagination-research/aimo2 |
| AI Mathematical Olympiad, Progress Prize 3 (2026) | 1st | Prefix caching across parallel samples, an 8-bit KV cache, weight files read into the page cache before the engine starts | write-up |
| Make Data Count (2025) | 1st | Qwen2.5-Coder-32B-AWQ zero-shot on vLLM, shown only a short context snippet | write-up |
| WSDM Cup Multilingual Chatbot Arena (2025) | 1st | A 72B AWQ second pass on vLLM covered only 5 % of samples with TTA: dropped | write-up |
| Deep Past, Akkadian to English (2026) | 1st | ByT5-XL compiled to CTranslate2 int8, data-parallel on two T4s; test sorted by length; MBR decoding on all 4 CPU cores | write-up |
| ARC Prize (2025) | 1st | Per-puzzle LoRA test-time fine-tuning with Unsloth and FlashAttention-2 | write-up |
| LMSYS Chatbot Arena (2024) | 3rd | 4-bit bitsandbytes QLoRA, `torch.compile` | github.com/rbiswasfc/lmsys-arena |

## Tabular

| Competition | Place | Technique | Source |
|---|---|---|---|
| Regression with an Insurance Dataset (Playground, 2024) | 1st | About 145,000 random column combinations evaluated as groupby features with cuDF-pandas | write-up |
| Predict Podcast Listening Time (Playground, 2025) | 1st | cuML Lasso and SVR on 6,000 features; GBDT on their residuals; TabPFN on 10,000-row subsamples | write-up |
| Predict Calorie Expenditure (Playground, 2025) | 1st | Hundreds of GBDT, NN and cuML models fed into GPU hill climbing | write-up |
| Predicting Optimal Fertilizers (Playground, 2025) | 1st | A cuML linear model's logits passed to XGBoost as `base_margin` | write-up |
| Predict Customer Churn (Playground, 2026) | 1st | cuML random forests for diversity; a cuML kNN graph for a GraphSAGE member | write-up |
| Binary Prediction of Poisonous Mushrooms (Playground, 2024) | 1st | LightGBM `num_threads` set to the number of CPU cores | write-up |
| Optiver Trading at the Close (2024) | 1st | One h5 file per day loaded into a preallocated float32 array; top 300 features kept to fit memory and allow more online-training rounds | write-up |
| USPTO Explainable AI (2024) | 1st | Set intersections of sorted id arrays with `cupy.intersect1d` on the GPU; only test patents kept in memory | write-up |
| Ariel Data Challenge (2025) | 1st | Signal processing in cupy | github.com/jcottaar/ariel2 |

## Audio

| Competition | Place | Technique | Source |
|---|---|---|---|
| BirdCLEF 2024 | 1st | OpenVINO with a fixed input shape on the CPU; mel spectrograms computed in parallel with joblib and cached in RAM | write-up |
| BirdCLEF+ 2025 | 2nd | ONNX ensemble converted to OpenVINO with `--compress_to_fp16`, no quantisation; spectrograms computed once for all models; multiprocess loading | write-up; github.com/VSydorskyy/BirdCLEF_2025_2nd_place |
| BirdCLEF 2023 | 1st | The ensemble exported to ONNX for CPU inference | github.com/VSydorskyy/BirdCLEF_2023_1st_place |
| HMS Harmful Brain Activity (2024) | 3rd | `torchaudio` MelSpectrogram as the model's first layer, on the GPU | github.com/darraghdog/kaggle-hms-3rd-place-solution |

## Simulation and other

| Competition | Place | Technique | Source |
|---|---|---|---|
| Lux AI Season 3 (2025) | 1st | GPUs split between training and experience generation, bf16, `torch.compile` (about 1.5x), pinned zero-copy transfers, separate processes | write-up |
| Yale/UNC-CH Geophysical Waveform Inversion (2025) | 1st | `torch.compile` on the forward simulator: from about 10 to over 5,000 images per minute per GPU | write-up |
| Stanford RNA 3D Folding Part 2 (2026) | 1st | Models routed by runtime budget: expensive models on short sequences only | write-up |
| Orbit Wars (2026) | 1st | int8 and 4-bit NormalFloat codebook quantisation to fit the file-size limit; a fallback to a small model when time runs low | write-up |
