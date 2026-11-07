# Audio and signals

## Training

- Decode once: resample to the model's sample rate (32 kHz is common for bird audio) and cache the waveforms as npy
  (float16 halves the size). Decoding compressed audio every epoch is the usual bottleneck.
- Compute spectrograms on the GPU: put `torchaudio.transforms.MelSpectrogram` and `AmplitudeToDB` inside the model,
  so the CPU only slices waveforms. HMS (3rd) built the mel spectrogram as the model's first layer. nnAudio computes
  CQT on the GPU.
- Augment on the GPU batch: mixing clips, gain, time shift, and SpecAugment (time and frequency masking) on the
  spectrogram.
- Take random crops of long recordings each epoch from memmapped arrays.
- EEG, ECG and other sensors: filter, re-reference and build montages once, and store float32 npy. Compute expensive
  features such as wavelet transforms offline.

## CPU-only inference

Bird-audio competitions often allow only CPU inference over hours of audio.

- Export the ensemble to ONNX (BirdCLEF 2023, 1st), convert it to OpenVINO with fp16 weights
  (`ovc model.onnx --compress_to_fp16`, BirdCLEF+ 2025, 2nd) and compile it with a fixed input shape
  (BirdCLEF 2024, 1st).
- Compute each window's spectrogram once and feed it to every model (BirdCLEF+ 2025, 2nd). BirdCLEF 2024 (1st)
  computed spectrograms in parallel with joblib and kept them in RAM.
- Batch windows across files, and use all 4 cores (one process per shard of files).
- Use smaller backbones, or a student distilled from the ensemble. Try a lower sample rate or a shorter FFT only
  when the score holds.
- Measure the throughput early, in a Kaggle CPU session with a test-sized input. The hardware behind the hidden
  rerun is the one that counts.

More on exports and multiprocessing: [inference.md](inference.md#cpu-only-inference).
