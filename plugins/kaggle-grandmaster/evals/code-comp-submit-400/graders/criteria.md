---
type: llm
weight: 1
---

The response explains that the CLI hides the reason and that it is in the HTTP response body, and shows how to read
it (for example calling the Kaggle Python API `competition_submit_code` inside try/except HTTPError and printing
`e.response.text`). It names a dataset attached to the notebook that is not versioned (FAILED_PRECONDITION
"non-versioned datasets") as a likely cause and gives the fix: attach a versioned dataset or pin a dataset version,
push a new kernel version, submit that version. Other preconditions (rules not accepted, internet enabled, output
file name) are a plus. Fail if it only blames the submission file format or the rules without mentioning the
response body or dataset versioning.
