---
name: audio-and-signal
description: Playbook for Kaggle audio and biosignal competitions — bioacoustics (BirdCLEF-style), speech, EEG/ECG/EMG, seismic and other 1D signals — covering spectrogram vs raw-waveform models, augmentation, weak labels, soundscape inference under CPU limits, and subject-grouped validation. Use for any audio or sensor-waveform data.
---

# Audio & Signal Playbook

## 1. Representations

- **Mel spectrograms** (log-mel, 128 mels, hop chosen so the event of interest spans enough
  frames) fed to ImageNet-pretrained CNNs (EfficientNet, NFNet, ConvNeXt, eca_nfnet) — the
  default winner for bioacoustics. Normalise per-sample or per-dataset; try PCEN.
- **Raw waveform** 1D CNNs / wav2vec2 / BEATs / audio transformers (AST, PaSST) for diversity.
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

Bioacoustics comps often require CPU inference on hours of audio: use ONNX/OpenVINO, smaller
backbones, 32 kHz → lower sample rate if acceptable, precompute mel on the fly in batches,
multiprocessing; post-process with temporal smoothing across neighbouring windows and
file-level max/mean pooling boosts (e.g. add a fraction of clip-level max to each window).
Measure throughput on the exact hardware early.

## 5. Common winning tricks

- Pretrain on previous years' competition data / xeno-canto (if allowed) then fine-tune.
- Ensembles of different backbones, input lengths, mel settings.
- Distil a big ensemble into a small fast student for the CPU budget.
- Temporal post-processing: smoothing predictions over adjacent windows; thresholding absent
  classes by recording-level evidence.
