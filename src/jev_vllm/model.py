from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE
from vllm.model_executor.models.interfaces_base import VllmModelForPooling, attn_type
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from jev_model.pointer.model import PointerHead


class PointerPooler(Pooler):
    """Logits over a prompt's options, divided by the fitted temperature, as `PointerModel` computes them: the hidden
    state at each `</opt>` against the one at `<decide>`, the last token. `AllPool` gathers a prompt's hidden
    states, across chunks when its prefill is split."""

    def __init__(self, head: PointerHead, close_id: int):
        super().__init__()
        self.pool = AllPool()
        self.head = head
        self.close_id = close_id

    def get_supported_tasks(self) -> set[PoolingTask]:
        # `plugin` comes through the IO processor; `token_classify` takes ids from `Encoder.assemble`
        return {"plugin", "token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata) -> list[torch.Tensor | None]:
        outputs = []
        for hidden, ids in zip(self.pool(hidden_states, pooling_metadata), pooling_metadata.get_prompt_token_ids()):
            if hidden is None:
                # a prefill chunk that is not the last one
                outputs.append(None)
                continue
            closes = (ids.to(hidden.device) == self.close_id).nonzero().squeeze(-1)
            hidden = hidden.float()
            logits = self.head.key(hidden[closes]) @ self.head.query(hidden[-1]) * self.head.scale
            outputs.append(logits / self.head.temperature)
        return outputs


@attn_type("hybrid")
class Qwen3_5PointerModel(nn.Module, HasInnerState, IsHybrid, SupportsMRoPE, VllmModelForPooling):
    """Qwen3.5's text model with the pointer head as its pooler, for folders written by `jev-model export`. vLLM
    registers no text-only Qwen3.5 class, so the hybrid-cache and M-RoPE hooks of the multimodal one are reused.
    Being `hybrid` keeps prefix caching off: a cached prefix would leave holes in the hidden states."""

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.model = Qwen3_5Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # float32 as in training, where the head runs outside autocast
        head = PointerHead(vllm_config.model_config.hf_text_config.hidden_size, config.pointer_head_dim).float()
        head.temperature.fill_(config.pointer_temperature)
        self.pooler = PointerPooler(head, config.pointer_close_id)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self, skip_prefixes=["mtp.", "lm_head."]).load_weights(weights)

    def get_mrope_input_positions(self, input_tokens: list[int], mm_features: list) -> tuple[torch.Tensor, int]:
        # text only: the three M-RoPE axes share one position, which is plain RoPE
        return torch.arange(len(input_tokens)).unsqueeze(0).expand(3, -1).clone(), 0

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
