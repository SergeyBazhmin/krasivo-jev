# Serving with vLLM

A trained run is merged into one folder and served by vLLM 0.17.1 with two plugins from this repo
(`src/jev_vllm/`). The endpoint takes the same body as TypeSafe's `system_one` API and answers in its shape.

```
runs/pointer/X ──jev-model export──▶ exports/X ──serve_pointer.sh──▶ vLLM :8080 /pooling
 (base name + LoRA + head)            (merged bf16 weights,             (Qwen3_5PointerModel +
                                        config, tokenizer)                jev_pointer IO processor)
```

## 1. Export

```bash
uv run jev-model export runs/pointer/X --out exports/X [--stage ce] [--device cpu]
```

`export` loads the stage, merges the LoRA into the base weights and writes:

| file | what it is |
|------|------------|
| `model.safetensors` | the text model (bf16) as `model.*` and the pointer head (float32) as `pooler.head.*` |
| `config.json` | the base model's config plus `architectures: ["Qwen3_5PointerModel"]`, `io_processor_plugin: "jev_pointer"` and the `pointer_*` settings: head size, `max_length`, the `</opt>` token id, the fitted temperature |
| `tokenizer*`, `vocab.json`, `merges.txt`, ... | the base model's tokenizer files, copied unchanged (vLLM still runs transformers 4) |

The folder is self-contained: the server does not need the run, the Hub or this repo's training code beyond the
plugins. Export on any machine with enough memory for the 2B model; `--device cpu` works.

### From a Hub model

`jev-model export` takes a run directory. A model pushed to the Hub is a single stage folder, so download it and
call the exporter directly:

```bash
uv run python -c '
from pathlib import Path
from huggingface_hub import snapshot_download
from jev_model.pointer.export import export
export(Path(snapshot_download("user/jev-pointer")), Path("exports/jev-pointer"), "cpu")'
```

Add `revision=` to `snapshot_download` to pin a version. For a private repo, log in or set `HF_TOKEN` first.

## 2. Serve

### With Docker (recommended)

```bash
bash src/scripts/serve_pointer.sh exports/X                       # extra arguments go to vLLM
bash src/scripts/serve_pointer.sh exports/X --max-num-seqs 512
```

The script builds `docker/vllm.Dockerfile` (the `vllm/vllm-openai:v0.17.1` image plus this package installed with
`--no-deps`, so the image keeps its own torch and transformers), mounts the export read-only at `/model`, and starts
the container detached on port 8080 with `--runner pooling --served-model-name pointer`. Follow it with
`docker logs -f <container>`; it is ready when `/health` returns 200.

### Without Docker

Install vLLM 0.17.1 and this package into one environment, then:

```bash
pip install vllm==0.17.1
pip install --no-deps .                            # registers the jev_vllm entry points
pip install loguru python-dotenv datasets accelerate
vllm serve exports/X --runner pooling --served-model-name pointer --port 8080
```

Install with `--no-deps`: the package's own pins (transformers 5, a newer torch) would break vLLM. The plugins are
found through the `vllm.general_plugins` and `vllm.io_processor_plugins` entry points in `pyproject.toml`; check
them with

```bash
python -c "from importlib.metadata import entry_points as e; print(e(group='vllm.io_processor_plugins').names)"
```

## 3. Call it

`POST /pooling` with the model name and a `data` object holding `state` and named questions:

```bash
curl localhost:8080/pooling -H 'Content-Type: application/json' -d '{
  "model": "pointer",
  "data": {
    "state": "I was charged twice. Please help.",
    "questions": {
      "billing": {"type": "noul", "instructions": "Is this about billing?"},
      "tone": {"type": "choice", "instructions": "What is the tone?",
               "criteria": {"angry": null, "calm": "polite and patient"}}
    }
  }
}'
```

```json
{"data": {"answers": {
  "billing": {"type": "noul", "noul": 0.98},
  "tone": {"type": "choice", "choice": "calm", "confidence": 0.7, "probabilities": {"angry": 0.3, "calm": 0.7}}
}}, ...}
```

| field | type | meaning |
|-------|------|---------|
| `state` | string | the text the questions are about; may be `""` |
| `questions` | object, at least one entry | question name -> question |
| `questions.*.type` | `"noul"` or `"choice"` | yes/no, or one label out of several |
| `questions.*.instructions` | string | the question itself |
| `questions.*.criteria` | object, `choice` only, at least 2 entries | label -> description, or `null` |

Unknown fields are rejected (the schema forbids extras), as is a `choice` with fewer than two criteria. The answers
mirror [`JevHolder.system_one`](usage.md): `noul` is P(yes); a `choice` gives the top label, its probability and
every label's.

From Python:

```python
import httpx

body = {"model": "pointer", "data": {"state": state, "questions": questions}}
answers = httpx.post("http://localhost:8080/pooling", json=body, timeout=60).json()["data"]["answers"]
```

## How a request runs

1. The IO processor (`jev_vllm/io_processor.py`) validates `data` and builds each question's options with
   `jev_model.system_one.options`, exactly as `JevHolder` does.
2. Each question becomes one prompt, laid out by the training `Encoder`:
   `<state> state <q> question <opt> o1 </opt> ... <decide>`. A state over `pointer_max_length` tokens loses its
   middle; a question whose instructions and options alone do not fit fails the request.
3. The model (`jev_vllm/model.py`) runs Qwen3.5's text stack. Its pooler takes the hidden state at each `</opt>`
   and at `<decide>` (the last token), scores them with the pointer head and divides by the fitted temperature.
4. The IO processor applies a softmax and returns the answers by name.

There is no generation, so a request costs one prefill per question and nothing else. vLLM batches prompts across
requests, which is where it beats `JevHolder` on throughput.

## Notes and limits

- **Prefix caching is off.** Qwen3.5 is a hybrid model (most layers are linear attention); the model is marked
  `hybrid` so a cached prefix cannot leave holes in the hidden states the pooler reads. Several questions about one
  state therefore each pay for the state.
- **Only Qwen3.5 backbones** are supported by `Qwen3_5PointerModel`. A run trained on another base model exports,
  but vLLM will not load it.
- **Numbers match the torch path up to bf16 noise.** The merged weights are bf16 and the head runs in float32, as in
  training. Spot-check a new export against `JevHolder` on a few rows before switching traffic.
- **One model per server.** Serve several exports on different ports or GPUs (`CUDA_VISIBLE_DEVICES`, `-p`).
- **Memory.** `serve_pointer.sh` passes `--gpu-memory-utilization 0.95`; lower it when the GPU is shared.

## Troubleshooting

| symptom | likely cause |
|---------|--------------|
| vLLM says `Qwen3_5PointerModel` is not a supported architecture | the `jev_vllm` plugins are not installed in vLLM's environment |
| vLLM cannot find the `jev_pointer` IO processor | same; check the entry points as above |
| `question 'x' and its options alone exceed N tokens` | shorten the instructions or the criteria descriptions, or re-export a run trained with a larger `max_length` |
| `huggingface-hub` / `transformers` import errors at start | the package was installed with its dependencies; reinstall it with `--no-deps` |
| answers differ from `JevHolder` by more than ~0.01 | the export comes from another stage than the one you compare with (`--stage`) |
