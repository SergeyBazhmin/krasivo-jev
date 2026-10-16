from collections.abc import Sequence

from pydantic import BaseModel
from vllm.config import VllmConfig
from vllm.inputs import TokensPrompt
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.renderers import BaseRenderer

from jev_model.pointer.encode import Encoder


class Question(BaseModel):
    state: str = ""
    question: str
    options: list[str]


class Answer(BaseModel):
    """Calibrated logits and probabilities, in the order the options were sent."""

    logits: list[float]
    probs: list[float]


class PointerIOProcessor(IOProcessor[list[Question], list[Answer]]):
    """`/pooling` with `{"data": {"state", "question", "options"}}` (or a list of them) answers each question.
    Prompts are laid out by the same `Encoder` as in training, so a long state loses its middle and `<|..|>` in the
    text cannot forge a delimiter."""

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__(vllm_config, renderer)
        config = vllm_config.model_config.hf_config
        self.encoder = Encoder(renderer.get_tokenizer(), config.pointer_max_length)

    def parse_data(self, data: object) -> list[Question]:
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise TypeError("data must be a question or a list of questions")
        return [Question.model_validate(item) for item in data]

    def pre_process(self, prompt: list[Question], request_id: str | None = None, **kwargs) -> Sequence[TokensPrompt]:
        encoder = self.encoder
        prompts = []
        for question, tokens in zip(prompt, encoder.tokenize([(q.state, q.question, q.options) for q in prompt])):
            if len(question.options) < 2:
                raise ValueError(f"need at least 2 options, got {len(question.options)}")
            if not encoder.fits(tokens):
                raise ValueError(f"question and options alone exceed {encoder.max_length} tokens")
            ids, _ = encoder.assemble(tokens, list(range(len(question.options))))
            prompts.append(TokensPrompt(prompt_token_ids=ids))
        return prompts

    def post_process(
        self, model_output: Sequence[PoolingRequestOutput], request_id: str | None = None, **kwargs
    ) -> list[Answer]:
        answers = []
        for output in model_output:
            logits = output.outputs.data.float()
            answers.append(Answer(logits=logits.tolist(), probs=logits.softmax(dim=-1).tolist()))
        return answers
