---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

I'm in an image classification competition (knee X-rays, 2,000 DICOM studies) and I train on Kaggle
notebooks because I have no local GPU. `kaggle quota` says I have 11 GPU hours left this week and the
quota refreshes in about 30 hours; the competition ends in 16 days. One 5-fold ConvNeXt-Tiny run at
512px takes ~9 hours on a single GPU and my notebook decodes the DICOMs on the fly. I have six ideas
I want to try (different losses, resolutions, a second backbone, more augmentation). How should I
spend my GPU time to get the best final leaderboard?
