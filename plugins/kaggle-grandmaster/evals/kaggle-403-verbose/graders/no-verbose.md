---
type: llm
weight: 2
---

PASS when the response advises against VERBOSE / VERBOSE_OUTPUT (or at least warns clearly before any use)
because the Kaggle CLI then prints every request's headers, including the bearer token / API credentials, which
would expose the secret in the terminal or chat.

FAIL if it recommends setting VERBOSE=1 (or VERBOSE_OUTPUT) without warning that it exposes the token.
