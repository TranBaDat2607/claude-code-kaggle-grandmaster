# Making a notebook and weights public (`/kg-showcase` step 9)

Audit before anything goes public on Kaggle, and show the user the findings:
- **Inputs** of the notebook (`kernel-metadata.json`). A private dataset or model breaks the public notebook for
  readers: publish it first or drop it. Other users' inputs: credit them.
- **Saved outputs** in the `.ipynb`: API keys, tokens, `kaggle.json` paths, absolute paths with a username. Also
  printed competition data beyond what the competition itself shows on Kaggle.
- **A weights dataset** (`kaggle datasets files <owner>/<slug>`, or the local upload folder): weights and the code
  to load them, nothing else. No competition data: no CSV, parquet, npy or image files, no cached or pre-decoded
  inputs, no OOF or label tables keyed by the competition's IDs. No credentials, no logs.
- **Checkpoints**: load each with `torch.load(path, map_location="cpu", weights_only=True)`. One that only loads with
  `weights_only=False` pickles objects. Such a file runs code on load for every reader, and can carry label
  tables, optimizer state or a config with local paths. Re-save it as a plain `state_dict` or as safetensors.
- `dataset-metadata.json`: a description that says what the files are and how they were trained, and a license.

Then, only on the user's go-ahead:
- An existing notebook or dataset is made public from its page on the Kaggle website: no new version, no re-run.
  For a notebook that already ran, prefer this. `kaggle kernels push` with `"is_private": false` creates and runs
  a new version, which spends quota and can fail.
- A new weights dataset: `kaggle datasets create -p <dir>` (private by default). The user checks it on the
  website, then makes it public there.
