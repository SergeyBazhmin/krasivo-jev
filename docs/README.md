# Docs

How to use, serve and train the `pointer` model: a Qwen3.5 decoder with a LoRA and a pointer head that answers a
multiple-choice question about a given text in one forward pass, with calibrated probabilities.

| page | read it when you want to |
|------|--------------------------|
| [Usage](usage.md) | ask a trained model questions from Python, the CLI or the browser |
| [Serving with vLLM](vllm.md) | put a trained model behind an HTTP endpoint |
| [Finetuning](finetuning.md) | train a model on your own data, evaluate it and publish it |
| [Adding a dataset](adding-a-dataset.md) | write a converter that brings a new source into the training mix |
| [How the model works](model.md) | understand what the model can and cannot do, and why its inputs look the way they do |

The full dataset reference (every converter, how splits are made) is in [`src/jev_datasets/README.md`](../src/jev_datasets/README.md)
and `AGENTS.md`. The reference for every model and file is [`src/jev_model/README.md`](../src/jev_model/README.md) and
[`src/jev_model/pointer/README.md`](../src/jev_model/pointer/README.md).

## Which way to run a model

| | `JevHolder` (Python) | `jev-model predict` / `ui` | vLLM |
|---|---|---|---|
| loads | a run, a stage folder or a Hub repo | a run | an `export` folder |
| weights | base + LoRA, unmerged | base + LoRA, unmerged | merged, bf16 |
| batching | the questions of one call | one question | continuous, across requests |
| best for | notebooks, scripts, offline batch jobs | quick checks by hand | production traffic |

All three lay out the prompt with the same `Encoder` and read answers with the same `jev_model.system_one`, so a
question gets the same answer from each, up to numeric noise.

## Setup

```bash
uv sync --extra model                   # training, evaluation and Python inference
uv sync --extra model --extra ui        # + the Streamlit page
```

A CUDA GPU is expected. Inference runs on CPU too, but slowly. The base model (`unsloth/Qwen3.5-2B-Base` by default)
is downloaded from the Hugging Face Hub on first use. Install `flash-linear-attention` (in the `model` extra) on a
GPU machine; without its kernels the linear-attention layers run more than ten times slower.
