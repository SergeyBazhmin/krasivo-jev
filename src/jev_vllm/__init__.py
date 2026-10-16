"""vLLM plugins that serve an exported `pointer` run (`jev-model export`). Installed next to vLLM, they are found
through the entry points in `pyproject.toml`; nothing here is imported by `jev_datasets` or `jev_model`."""

ARCHITECTURE = "Qwen3_5PointerModel"


def register_model():
    """`vllm.general_plugins`: runs in every vLLM process, so it only registers the class by name."""
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, "jev_vllm.model:Qwen3_5PointerModel")


def io_processor() -> str:
    """`vllm.io_processor_plugins`: the class that turns `/pooling` requests into prompts."""
    return "jev_vllm.io_processor.PointerIOProcessor"
