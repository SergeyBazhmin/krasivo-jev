# zero_shot

A pretrained chat (instruct) LM used as it is, with no training. An OpenAI-compatible server (a local vLLM) generates an answer,
and the option number it replies with is the prediction. It is a baseline for the trained models on the same test rows
(`jev_model.data.partition`). It scores only `datasets`: benchmarks are not added unless named. Nothing runs on the local GPU; only the tokenizer is loaded, to measure prompts.

## How it works

The options are numbered, and the model is asked to reply with the number of one. The request goes to
`chat/completions`, and the server's chat template wraps it, so the served model must have one:

```
system:  Choose one option. Reply with its number only.
user:    {state}

         {question}

         1. first option
         2. second option
```

`thinking` is passed to the template as `chat_template_kwargs.enable_thinking`.

Decoding is greedy (`temperature=0`), with up to `max_new_tokens` new tokens, or `think_tokens` when thinking.
Thinking is cut off at `</think>`. If the server runs a reasoning parser, the content already comes without it.
A thought that runs out of tokens leaves no answer. The answer is read as the number at its start
(`2`, `(2)`, `2. text`), or after "answer" anywhere (`The answer is 2`). An answer without a number in range counts as wrong.

A message over `max_length` tokens, measured with `tokenizer` (by default the served model's name on the Hub),
loses the middle of its state (`jev_model/prompt.py:cut_middle`). Keep `max_length` under the server's
`max_model_len`. A sample whose question and options do not fit on their own is skipped.

## Metrics

`accuracy` against the argmax of the label.
`parsed` is the share of answers with a number in range. Generation gives no probabilities, so `nll` and `ece` are
NaN. Every answer is saved to `RUN/generate/<partition>-<dataset>.jsonl` in the format `pointer` and `typesafe`
use: `options`, `probs` (one-hot on the option the answer names, `null` when it names none) and `label`, plus the raw
`answer`. `jev-model thresholds` reads them; with one-hot probabilities every cut-off gives the same precision and
recall, those of the generated answers, and unparsed answers are skipped.

## Usage

```bash
vllm serve Qwen/Qwen3.5-2B --max-model-len 8192 --port 8080              # in another shell
uv run jev-model train zero_shot --run-dir runs/zero_shot/chat           # records the config, trains nothing
uv run jev-model eval runs/zero_shot/chat
uv run jev-model thresholds runs/zero_shot/chat                          # precision/recall of the noul datasets
uv run jev-model train zero_shot -s url=http://host:8001/v1 --run-dir runs/zero_shot/other   # another server
```

## Config

| key              | default                    | meaning |
|------------------|----------------------------|---------|
| `datasets`       | all built                  | datasets to score; benchmarks only when named (or when empty) |
| `url`            | `http://localhost:8080/v1` | OpenAI-compatible server |
| `model`          | first served               | served model name |
| `tokenizer`      | `model`                    | Hub tokenizer that measures prompts |
| `max_length`     | 4096                       | prompt tokens |
| `max_samples`    | 0 (all)                    | rows per dataset, picked by a content hash so runs score the same ones |
| `concurrency`    | 16                         | requests in flight |
| `retries`        | 5                          | attempts per request, with exponential backoff |
| `thinking`       | `false`                    | let a reasoning model think first |
| `max_new_tokens` | 16                         | answer budget without thinking |
| `think_tokens`   | 2048                       | answer budget with thinking |
