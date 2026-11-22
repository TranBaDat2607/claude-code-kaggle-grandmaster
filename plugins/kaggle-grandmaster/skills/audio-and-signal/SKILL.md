---
name: audio-and-signal
description: "Playbook for Kaggle audio and biosignal competitions (bioacoustics, speech, EEG/ECG, seismic). Use when the inputs are sound or sensor waveforms."
---

# Audio & Signal Playbook

## 1. Representations

- **Mel spectrograms** (log-mel, 128 mels, hop chosen so the event of interest spans enough
  frames) fed to ImageNet-pretrained CNNs (EfficientNet, NFNet, ConvNeXt, eca_nfnet) — the
  default winner for bioacoustics. Normalise per-sample or per-dataset; try PCEN.
- **Raw waveform** 1D CNNs / wav2vec2 / BEATs / audio transformers (AST, PaSST) for diversity.
- Compute: decode audio once at the model's sample rate (npy cache) and compute the spectrogram on the GPU inside
  the model (torchaudio, as HMS 3rd did). See `compute-efficiency` (`references/audio-and-signal.md`).
- **Biosignals (EEG/ECG)**: spectrograms per channel/bipolar montage stacked as image tiles,
  or 1D CNN/GRU/transformer on raw signals with band-pass filtering; also domain features
  (band powers, HRV) for GBDT stackers.

## 2. Labels & training

- Weak/clip-level labels with short events: random crops of 5–10s, attention/SED heads
  (frame-wise predictions + attention pooling), BCE (multilabel) rather than softmax.
- Secondary labels (other species present) → soft targets (e.g. 0.3–0.5).
- Label noise: knowledge distillation from an ensemble, pseudo-label unlabelled soundscapes,
  loss truncation.
- Class imbalance: sample weighting by class frequency^-0.5, upsampling rare classes, focal BCE.
- Augmentations: mixup (very effective for audio, also mixing with background noise/
  no-call clips), time/frequency masking (SpecAugment), gain, pitch/time shift, random
  filtering, adding external noise recordings.

## 3. Validation

- Group by recording/site/subject/session; check whether the test is from different
  recording conditions (soundscapes vs focal recordings) — build validation from
  test-like data (e.g. labelled soundscapes) when possible.
- Metric often macro-averaged (padded cmAP, macro AUC): rare classes dominate — monitor
  per-class scores.

## 4. Inference under CPU-only limits

Bioacoustics comps often require CPU inference on hours of audio: export to ONNX, then OpenVINO with fp16 weights
and a fixed input shape (BirdCLEF 2024 1st, BirdCLEF+ 2025 2nd), one spectrogram per window shared by every model,
smaller backbones, 32 kHz → lower sample rate if acceptable, all 4 cores
(multiprocessing); post-process with temporal smoothing across neighbouring windows and
file-level max/mean pooling boosts (e.g. add a fraction of clip-level max to each window).
Measure throughput on the exact hardware early.

## 5. Common winning tricks

- Pretrain on previous years' competition data / xeno-canto (if allowed) then fine-tune.
- Ensembles of different backbones, input lengths, mel settings.
- Distil a big ensemble into a small fast student for the CPU budget.
- Temporal post-processing: smoothing predictions over adjacent windows; thresholding absent
  classes by recording-level evidence.

## 6. Cheap screens (sweeps)
Many values of one lever are one sweep (`/kg-grind --sweep`): screened cheap against a control at the same fidelity, the best confirmed by a full run judged as the best of all of them.

Screen on whole recordings, subjects or sessions (`--group`), as the folds keep them; per-clip scores averaged per
recording log a unit table (`mean`, grouped by recording). Cheap settings that usually keep the ranking: shorter
crops, fewer mel bins or a lower sample rate, a third of the epochs. Not a screen question: crop length and
resolution themselves, subject effects seen on a few subjects.
