# vLLM with the `jev_vllm` plugins, for serving folders written by `jev-model export` (`src/scripts/serve_pointer.sh`).
# The version matches the one `jev_vllm` is written against; vLLM still runs transformers 4.
FROM vllm/vllm-openai:v0.17.1

WORKDIR /opt/krasivo-jev
COPY pyproject.toml README.md ./
COPY src ./src

# --no-deps keeps the image's own torch and transformers; the rest is what importing `jev_model` pulls in
RUN pip install --no-cache-dir --no-deps . \
    && pip install --no-cache-dir loguru python-dotenv datasets accelerate \
    && python3 -c "from importlib.metadata import entry_points as e; \
assert 'jev_pointer' in e(group='vllm.io_processor_plugins').names"

WORKDIR /
