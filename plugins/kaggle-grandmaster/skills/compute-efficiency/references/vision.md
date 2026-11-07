# Images, 3D volumes and video

Contents: [Decode once, cache at training size](#decode-once-cache-at-training-size) ·
[Augmentation on the GPU](#augmentation-on-the-gpu) · [2D models](#2d-models) · [3D volumes](#3d-volumes) ·
[Whole-slide images and video](#whole-slide-images-and-video) · [Inference](#inference)

## Decode once, cache at training size

Decoding is the most common bottleneck in image competitions. A Kaggle GPU session has 4 CPU cores, and decoding a
2-4K PNG or a DICOM takes tens of milliseconds, so two T4s wait on the CPU.

- Convert the images once to the training resolution (or 1.25x of it, to keep crop augmentation) and store uint8:
  - npy (one file per image, or one memmap per split): fastest to read, largest on disk.
  - PNG: lossless and smaller, but decoding is slower.
  - JPEG at quality 90-95: smallest and fast to decode; check that the score holds.
- A UAT project measured a decoded cache about 25x faster to read than the original files. In RSNA 2023 (1st),
  the winners converted the DICOM series with `dicomsdl` to npy before training.
- Measured in a Kaggle T4 x2 session (November 2026): resnet34 at 384 px in fp16 trained 200 images/s on one T4,
  but four loader workers decoding 1536 px JPEGs fed only 44/s (`--bench`: INPUT-BOUND). With the JPEGs resized
  once to 384 px, the loader fed 140/s and the real loop ran 2.6x faster. `torch.compile` then gained nothing,
  because the run was still input-bound.
- DICOM: `dicomsdl` or `pydicom` with `pylibjpeg`/`gdcm` for compressed transfer syntaxes. Apply windowing and
  rescaling once, during the conversion.
- Convert in parallel: `concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())` over files. Keep the
  code under `if __name__ == "__main__":` so it also works on Windows.
- Publish the cache as a Kaggle dataset (or keep it as a kernel output) and attach it to remote runs
  (`kgkit gpu build --dataset ...`), so sessions do not repeat the conversion.
- When JPEG decoding is still the bottleneck: decode on the GPU with `torchvision.io.decode_jpeg(data,
  device="cuda")` (nvJPEG, works on lists of images) or NVIDIA DALI (Linux only).
- `train_image.py` decodes with PIL and resizes every epoch. Point `image_dir` at a pre-resized copy, or replace
  `ImageDataset.__getitem__` with an npy read.

## Augmentation on the GPU

- Keep the CPU workers to reading and a random crop. Do flips, rot90, colour jitter, MixUp and CutMix on the batch,
  on the GPU (torch ops, `torchvision.transforms.v2` on tensors, or kornia).
- Normalise on the GPU: send uint8 tensors, then `x.float().div_(255).sub_(mean).div_(std)`.
- BYU 2025 (1st) applied MixUp to the batch inside the model's forward pass.

## 2D models

- Defaults: AMP (fp16 + GradScaler on T4), `channels_last` model and inputs, `cudnn.benchmark = True` (fixed sizes),
  fused AdamW, EMA, a batch size that is a multiple of 8.
- `torch.compile` for runs longer than about 20 min, on Linux with T4 or newer (`train_image.py --compile`).
- Progressive resizing, then full resolution for the last part of the schedule.
- Screen backbones and resolutions at low resolution on one fold. Throughput (images/s on a T4) belongs in the
  backbone choice next to the score: a backbone 2x slower must earn its place.

## 3D volumes

- Resample once to the target spacing and store npy (or zarr with blosc compression for large volumes). Read random
  patches from memmaps (`np.load(path, mmap_mode="r")`) instead of loading whole volumes.
- `channels_last_3d` with AMP. 3D convolutions are memory-bound, so a patch size and batch that fit (with
  gradient checkpointing when needed) beat a larger patch that does not.
- 2.5D (adjacent slices stacked as channels into a 2D backbone) is often a cheaper alternative to full 3D.
- nnU-Net (RSNA Intracranial Aneurysm 2025, 7th) runs `torch.compile` and `cudnn.benchmark` by default on CUDA.
- Sliding-window inference: the overlap sets the cost (0.5 overlap is about 4x the windows of 0 overlap in 2D, 8x in
  3D). Use the smallest overlap that holds the OOF score.

## Whole-slide images and video

- Pathology slides: tile once at the target magnification, drop background tiles, and cap the tiles per slide. UBC-OCEAN
  (1st) sampled at most 200 tissue patches per slide and processed two slides at a time with Ray processes.
- Video: decode once at a reduced frame rate to npy or JPEG frames. For large sets, decode on the GPU (DALI, Linux).

## Inference

The full checklist is in [inference.md](inference.md). For images:

- Preprocess once and feed every ensemble member from the same tensors.
- Batch TTA: concatenate the flipped batch with the original instead of running two passes.
- Exports that won: TensorRT through ONNX Runtime's `TensorrtExecutionProvider` gave about 2x over `torch.jit` in CZII
  CryoET (1st), with the engine built once on a Kaggle T4 and loaded from a dataset. `torch.compile` gave about 2x in
  SenNet (1st), which made 8x TTA at high resolution fit the time limit. BYU 2025 (3rd) ran TensorRT engines.
- Both T4s: one process per GPU, each on half of the test set (CryoET 1st, Image Matching 2024 and 2025 1st).
