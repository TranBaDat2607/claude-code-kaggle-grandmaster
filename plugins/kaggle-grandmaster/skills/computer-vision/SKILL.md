---
name: computer-vision
description: Winning playbook for Kaggle computer vision competitions — image classification, object detection, segmentation, medical imaging (CT/MRI/X-ray/pathology), video, satellite — covering backbone choice (timm), resolution, augmentations, losses, training recipes, TTA, pseudo-labelling, multi-stage pipelines and inference under time limits. Use for any image/video data.
---

# Computer Vision Playbook

## 1. Baseline recipe (works for most classification tasks)

- Backbone from `timm`: start with a fast, strong ImageNet-pretrained model
  (EfficientNet-B0/B3, ConvNeXt-Tiny/Small, EfficientNetV2-S, NFNet-F0, or `tf_efficientnet`
  variants); ViT/EVA/DINOv2/SigLIP backbones for semantic tasks with enough data.
- Input: native-ish resolution of the objects that matter (resolution is often the #1
  lever — try 384→512→768+ once the pipeline is solid).
- AdamW, lr 1e-4–3e-4 (backbone) with head ×10, cosine schedule with warmup (1 epoch),
  weight decay 1e-2, mixed precision (bf16/fp16), batch as large as fits, EMA of weights.
- Augmentations (albumentations): flips, shift-scale-rotate, random resized crop,
  brightness/contrast, hue/saturation, CoarseDropout; MixUp/CutMix for classification.
  Choose augmentations that preserve the label (no vertical flips for text, careful with
  colour in medical/satellite data).
- 5 folds (grouped by patient/source!), 10–30 epochs, save best-by-metric per fold, OOF preds.
- Template: `templates/train_image.py`.

## 2. Levers in rough order of typical impact

1. **Correct grouping in CV** (patients, studies, scenes, near-duplicates via image hashing).
2. **Data understanding**: look at hundreds of images per class, at the errors, at label noise.
3. **Resolution & cropping** (ROI crops via a first-stage detector/segmenter are huge in
   medical/satellite comps).
4. **Backbone family & size** (scale after the pipeline is right; try 2–4 families for diversity).
5. **Loss**: BCE/CE with label smoothing; focal loss for imbalance; ArcFace/sub-center ArcFace
   for retrieval/identification (whale/landmark style); Lovasz/Dice+BCE for segmentation.
6. **Augmentation strength & schedule** (reduce augmentation in the last epochs).
7. **External / extra data and pretraining on in-domain data** (if rules allow).
8. **Pseudo-labelling** on test / unlabelled data (soft labels, high-confidence, several rounds).
9. **TTA** (flips, multi-scale) and **ensembles** of different backbones/resolutions.
10. **Post-processing** tuned on OOF (thresholds, morphological ops, small-object removal, WBF).

## 3. Task-specific notes

**Detection**: YOLO (Ultralytics v8/v11 — check licence vs competition rules), RT-DETR,
EfficientDet, Faster/Cascade R-CNN (mmdetection/detectron2). Ensemble with Weighted Boxes
Fusion (WBF), tune conf/IoU thresholds on OOF to the metric (mAP@[.5:.95], F2...).

**Segmentation**: `segmentation_models_pytorch` U-Net / U-Net++ / FPN / DeepLabV3+ with strong
encoders; SegFormer/Mask2Former for semantic tasks; 2.5D (stack neighbouring slices as channels)
for volumes, or 3D U-Nets; tile large images with overlap and blend; tune binarisation threshold
and minimum component size on OOF.

**Medical 3D (CT/MRI)**: DICOM handling (pydicom, correct windowing / rescale slope-intercept,
pixel spacing), series ordering by ImagePositionPatient, 2.5D CNN + sequence model (GRU/
transformer over slice embeddings) is a strong pattern; multi-stage: localise → crop → classify.
Patient-level aggregation of slice predictions.

**Pathology / huge images**: tile at the right magnification, filter background tiles, multiple
instance learning (attention MIL) over tile embeddings from a foundation model.

**Video**: sample frames, per-frame backbone + temporal model (1D CNN/transformer), or video
models (X3D, VideoMAE); event detection → tune tolerance windows to the metric.

**Retrieval / identification**: metric learning (ArcFace) + kNN on embeddings, query expansion,
re-ranking, threshold for "new individual".

**Satellite / remote sensing**: multispectral channels (adapt first conv), large tiles,
geography-grouped CV, temporal stacks.

## 4. Training hygiene

- Overfit one batch first; verify augmentations visually; check label ↔ image alignment.
- Log train/val loss and metric per epoch; watch for metric peaking before loss.
- Use `torch.compile`, channels_last, AMP, `persistent_workers`, DALI/decoding on GPU for speed;
  pre-resize images to disk once.
- Gradient accumulation for large effective batch; gradient checkpointing for big backbones.
- Keep a fixed validation transform; same pre-processing in inference kernels.

## 5. Inference under limits (code competitions)

Budget = runtime_limit × safety 0.8. Measure per-image latency of each model on the target
hardware (Kaggle T4×2 / P100 / L4). Trade-offs: fewer folds of a bigger model vs more folds of
smaller ones; TTA only where OOF shows gain; FP16; TensorRT/ONNX when beneficial; run models in
parallel on both GPUs. See `code-competitions`.
