---
description: "Image training on Kaggle 2xT4 at 35% GPU utilisation, decoding 2500px PNGs on the CPU: input-bound fixes first, measured"
tags: [portable, capability, compute]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

For a Kaggle competition I train an EfficientNet-B3 at 512px on 40,000 chest X-ray PNGs (about 2500x2500 pixels
each). One epoch takes 25 minutes on the 2x T4 machine. nvidia-smi shows about 35% GPU utilisation, and only one of
the two GPUs is busy. I use albumentations on the CPU with num_workers=8. Should I turn on torch.compile or move to a
bigger GPU? How do I make training faster without losing score?
