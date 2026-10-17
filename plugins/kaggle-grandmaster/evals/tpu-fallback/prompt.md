---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

Kaggle knee-MRI competition, code competition, deadline in 6 days. `kaggle quota` shows GPU 27.3h of 30h
used and TPU 0h of 20h used; both reset in 17 hours. I need to finish training my ConvNeXt-small model: 8
more epochs for 2 seeds, which takes about 10 GPU-hours on 2x T4 (one seed per T4, ~77 min per epoch). The
code is PyTorch with fp16 autocast + GradScaler and resumes from a CUDA checkpoint. Should I wait for the
GPU reset or use the TPU, and how do I do it without wasting a run?
