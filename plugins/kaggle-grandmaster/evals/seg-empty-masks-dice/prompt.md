---
description: "Per-image Dice with 60% empty masks: compute the metric like the host, gate empty predictions"
tags: [portable, regression, vision]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

Kaggle segmentation competition. The metric is the mean Dice coefficient computed per image; an image whose
ground-truth mask is empty scores 1 if you predict an empty mask and 0 otherwise. About 60% of test images have
no mask. My U-Net with a 0.5 threshold gets validation Dice 0.61, which I compute over all pixels of the
validation set at once. How do I improve my score?
