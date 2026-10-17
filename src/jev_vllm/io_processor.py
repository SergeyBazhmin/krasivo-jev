from collections.abc import Sequence
from dataclasses import asdict
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from vllm.config import VllmConfig
from vllm.inputs import TokensPrompt
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.renderers import BaseRenderer

from jev_model import system_one
from jev_model.pointer.encode import Encoder


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: str


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: str
    # label -> when it applies; None leaves the label to speak for itself
    criteria: dict[str, str | None]


class SystemOneRequest(BaseModel):
    """TypeSafe's `system_one` body without `model`, which vLLM's `/pooling` takes beside `data`."""

    model_config = ConfigDict(extra="forbid")

    state: str
    questions: dict[str, Annotated[NoulQuestion | ChoiceQuestion, Field(discriminator="type")]] = Field(min_length=1)

    def asked(self) -> dict[str, system_one.Noul | system_one.Choice]:
        return {
            name: system_one.Noul(q.instructions)
            if isinstance(q, NoulQuestion)
            else system_one.Choice(q.instructions, q.criteria)
            for name, q in self.questions.items()
        }


class PointerIOProcessor(IOProcessor[SystemOneRequest, dict]):
    """`/pooling` with `{"data": {"state": ..., "questions": {name: noul | choice}}}`, as TypeSafe's `system_one`,
    answers `{"answers": {name: {"type": "noul", "noul": p} | {"type": "choice", "choice", "confidence",
    "probabilities"}}}`. Options are built and answers read by `jev_model.system_one`, so a question gets the same
    answer as from `JevHolder`. Prompts are laid out by the same `Encoder` as in training, so a long state loses its
    middle and `<|..|>` in the text cannot forge a delimiter."""

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__(vllm_config, renderer)
        config = vllm_config.model_config.hf_config
        self.encoder = Encoder(renderer.get_tokenizer(), config.pointer_max_length)
        # request id -> its questions by name; `post_process` only gets the outputs, one per question in order
        self.pending: dict[str | None, dict[str, system_one.Noul | system_one.Choice]] = {}

    def parse_data(self, data: object) -> SystemOneRequest:
        return SystemOneRequest.model_validate(data)

    def pre_process(self, prompt: SystemOneRequest, request_id: str | None = None, **kwargs) -> Sequence[TokensPrompt]:
        encoder = self.encoder
        questions = prompt.asked()
        batch = [(prompt.state, q.instructions, system_one.options(q)) for q in questions.values()]
        prompts = []
        for name, (_, _, options), tokens in zip(questions, batch, encoder.tokenize(batch)):
            if not encoder.fits(tokens):
                raise ValueError(f"question {name!r} and its options alone exceed {encoder.max_length} tokens")
            ids, _ = encoder.assemble(tokens, list(range(len(options))))
            prompts.append(TokensPrompt(prompt_token_ids=ids))
        self.pending[request_id] = questions
        return prompts

    def post_process(
        self, model_output: Sequence[PoolingRequestOutput], request_id: str | None = None, **kwargs
    ) -> dict:
        questions = self.pending.pop(request_id)
        answers = {}
        for (name, question), output in zip(questions.items(), model_output):
            probs = output.outputs.data.float().softmax(dim=-1).tolist()
            answers[name] = {"type": "noul" if isinstance(question, system_one.Noul) else "choice"}
            answers[name].update(asdict(system_one.answer(question, probs)))
        return {"answers": answers}
