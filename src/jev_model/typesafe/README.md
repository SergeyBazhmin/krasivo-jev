# typesafe

TypeSafe's hosted `jev` model (`typesafe/jev-1.13` by default), called through its `system_one` API with
[`typesafe-sdk`](https://docs.typesafe.ai/sdk/python/). It is an external reference for the trained models on the same
test rows (`jev_model.data.partition`). Nothing is trained and nothing runs locally.

## How it works

Each sample is one request with one question, named `answer`:

- `state` is the sample's state, or its question when the state is empty (the API needs something to read).
- A sample of `type` `noul` goes as a `noul` question (`instructions` = the question). The `yes` option gets
  the returned `noul` probability, `no` gets the rest. `predict` has no `type`, so it treats options that are
  exactly `yes`/`no` as `noul`, the same rule the datasets use; so does `eval` on datasets built before `type`.
- A `choice` sample goes as a `choice` question. The criteria keys are the option texts. If texts repeat or are
  empty, the keys are option numbers (`"1"`, `"2"`, ...) and the texts become their descriptions. The returned
  `probabilities` map back to the options and are renormalised.

Unlike `zero_shot`, the answer is a distribution, so `nll` and `ece` are real. Probabilities are floored at 1e-9
so a confident miss has a finite NLL.

The SDK retries timeouts, 429 and 5xx with backoff (`retries`). A sample the API rejects (400/422, for example
one that is too long) is logged and counted as uniform. `answered` is the share that was not rejected.

## Metrics

`jev_model.metrics.scores` over the log probabilities: `accuracy`, macro `precision`/`recall`/`f1`, `nll`, `ece`,
plus `answered`. Every answer is saved to `RUN/api/<partition>-<dataset>.jsonl` with its probabilities and the label.

## Usage

```bash
echo TYPESAFE_API_KEY=... >> .env                                   # git-ignored, loaded on start, never stored in the run
uv run jev-model train typesafe -s max_samples=500 --run-dir runs/typesafe/jev-1.13   # records the config; caps rows (and the bill)
uv run jev-model eval runs/typesafe/jev-1.13
uv run jev-model train typesafe -s model=~typesafe/jev-latest --run-dir runs/typesafe/latest
uv run jev-model compare runs/typesafe/jev-1.13 runs/pointer/Y
```

## Config

| key           | default                  | meaning |
|---------------|--------------------------|---------|
| `datasets`    | all built                | datasets to score |
| `base_url`    | `https://routerai.ru/api`| API root |
| `model`       | `typesafe/jev-1.13`      | model name or alias (`~typesafe/jev-latest`) |
| `max_samples` | 0 (all)                  | rows per dataset, picked by a content hash so runs score the same ones |
| `concurrency` | 16                       | requests in flight |
| `retries`     | 5                        | retries per request after the first attempt |
| `timeout`     | 60                       | seconds per HTTP operation |
