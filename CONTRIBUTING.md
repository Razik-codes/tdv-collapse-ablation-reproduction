# Contributing

Contributions that improve correctness, reproducibility, or documentation are welcome. For substantial experimental changes, open an issue first so the intended claim and compute budget can be agreed upon.

## Research contributions

Please include:

- the claim or hypothesis being tested;
- the exact command, configuration, random seed, dataset version, and hardware;
- the baseline and the single controlled change, where possible;
- complete results, including negative or inconclusive outcomes;
- runtime and peak hardware requirements;
- limitations and any departure from the paper's protocol;
- figures or logs needed to audit reported numbers.

Do not commit datasets, credentials, checkpoints, or experiment-tracking secrets. Confirm that new datasets and adapted code permit redistribution, and update [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) when adding external code.

## Code changes

Keep changes focused and preserve the existing architecture. Before opening a pull request, run the dependency-free syntax check:

```bash
python3 -m compileall -q -x '(^|/)repos/' .
```

Training and evaluation changes should also be tested with the smallest relevant debug or smoke-test configuration available to you. Include the command and outcome in the pull request.
