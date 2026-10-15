---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

My inference notebook runs fine on Kaggle, but submitting it to the code competition fails. I ran
`kaggle competitions submit rsna-knee -k me/knee-infer -v 4 -f submission.csv -m "0012_effnet | CV 0.71"` and all
I get is:

```
400 Client Error: Bad Request for url: https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission
```

The notebook uses my fold weights from a dataset I uploaded. What is wrong and how do I find out?
