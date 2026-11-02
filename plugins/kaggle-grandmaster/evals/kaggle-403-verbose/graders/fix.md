---
type: llm
---

PASS when the response names the likely cause and fix: the user has not joined the competition and accepted its
rules on the Kaggle website (required before downloading), which only the user can do in the browser. Checking
the competition slug, or reading the error's response body (for example through the Kaggle Python API inside a
try/except that prints `e.response.text`), is a plus.

FAIL if it never mentions joining / accepting the competition rules.
