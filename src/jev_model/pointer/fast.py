"""Unsloth, when there is a GPU. It patches transformers and peft as it is imported, so every pointer entry point
imports this module before anything that loads them. It cannot be imported without a GPU, so CPU runs (smoke tests)
keep plain transformers + peft."""

import torch

ENABLED = torch.cuda.is_available()
if ENABLED:
    import unsloth  # noqa: F401
