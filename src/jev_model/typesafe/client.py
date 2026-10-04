"""Option probabilities from the TypeSafe `system_one` API (`typesafe-sdk`): one request per sample, one question
per request."""

import asyncio
from collections.abc import Callable

from loguru import logger
from tqdm import tqdm
from typesafe_sdk import (
    Answer,
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
    TypeSafeBadRequestError,
    TypeSafeUnprocessableEntityError,
)

from jev_model.typesafe.config import TypesafeConfig

# the name the single question goes under; the answer comes back keyed by it
NAME = "answer"


def labels(options: list[str]) -> list[str]:
    """`choice` criteria keys, one per option. Keys must be distinct, so options with repeated or empty texts are
    keyed by their number instead, and the text goes in as the description."""
    unique = len(set(options)) == len(options) and all(text.strip() for text in options)
    return options if unique else [str(i) for i in range(1, len(options) + 1)]


def kind(options: list[str]) -> str:
    """The `type` a dataset stamps on its rows, for options that come without one (`predict`)."""
    return "noul" if sorted(text.strip().lower() for text in options) == ["no", "yes"] else "choice"


def ask(question: str, options: list[str], type: str) -> Noul | Choice:
    if type == "noul":
        return Noul(instructions=question)
    keys = labels(options)
    criteria = {key: None if key == text else text for key, text in zip(keys, options)}
    return Choice(instructions=question, criteria=criteria)


def probabilities(answer: Answer, options: list[str]) -> list[float]:
    """The answer as a distribution over `options`, in their order."""
    if isinstance(answer, NoulAnswer):
        return [answer.noul if text.strip().lower() == "yes" else 1 - answer.noul for text in options]
    if not isinstance(answer, ChoiceAnswer):
        raise TypeError(f"expected a noul or choice answer, got {type(answer).__name__}")
    probs = [answer.probabilities.get(key, 0.0) for key in labels(options)]
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else [1 / len(options)] * len(options)


class Client:
    """Sends (state, question, options, type) to the API; `type` is the sample's `noul` or `choice`. A sample the
    API rejects (400/422, e.g. too long) gets None; other errors are retried by the SDK and raised once retries
    run out."""

    def __init__(self, config: TypesafeConfig):
        self.config = config
        logger.info(f"answering with {config.model} at {config.base_url}")

    def predict(
        self,
        questions: list[tuple[str, str, list[str], str]],
        on_answer: Callable[[int, list[float] | None], None] | None = None,
    ) -> list[list[float] | None]:
        """Probabilities over the options of each (state, question, options, type), in the given order.
        `on_answer(i, probs)` is called as each one arrives."""
        return asyncio.run(self.predict_all(questions, on_answer))

    async def predict_all(
        self,
        questions: list[tuple[str, str, list[str], str]],
        on_answer: Callable[[int, list[float] | None], None] | None,
    ) -> list[list[float] | None]:
        config = self.config
        semaphore = asyncio.Semaphore(config.concurrency)
        # the default policy gives up after 30 s in total, too soon for a long eval under rate limits
        retry = RetryPolicy(max_retries=config.retries, backoff_max=30.0, timeout=None)
        async with AsyncTypeSafeClient(
            base_url=config.base_url, model=config.model, timeout=config.timeout, retry=retry
        ) as client:

            async def one(i: int, state: str, question: str, options: list[str], type: str) -> list[float] | None:
                async with semaphore:
                    try:
                        # a sample with no state is all question; the API needs a state to read
                        response = await client.system_one(
                            state=state or question, questions={NAME: ask(question, options, type)}
                        )
                        probs = probabilities(response.answers[NAME], options)
                    except (TypeSafeBadRequestError, TypeSafeUnprocessableEntityError) as error:
                        tqdm.write(f"sample {i} rejected: {error}")
                        probs = None
                if on_answer:
                    on_answer(i, probs)
                return probs

            return list(await asyncio.gather(*(one(i, *q) for i, q in enumerate(questions))))
