"""Answers from an OpenAI-compatible server, such as a local vLLM, as `jev_datasets/translate.py` does."""

import asyncio
from collections.abc import Callable

import httpx
from loguru import logger
from tqdm import tqdm

from jev_model.zero_shot.config import ZeroShotConfig
from jev_model.zero_shot.prompt import SYSTEM, Prompter

THINK_END = "</think>"


def served_model(url: str) -> str:
    """The first model the server lists."""
    response = httpx.get(f"{url.rstrip('/')}/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


class Client:
    """Sends user messages to `url`/chat/completions with greedy decoding; the server applies the chat template.
    Any thinking is cut off, and a thought that ran out of tokens leaves no answer."""

    def __init__(self, config: ZeroShotConfig, model: str):
        self.config = config
        self.url = config.url.rstrip("/")
        self.model = model

    def answer(self, messages: list[str], on_answer: Callable[[int, str], None] | None = None) -> list[str]:
        """The answers in the order of `messages`. `on_answer(i, answer)` is called as each one arrives."""
        return asyncio.run(self.answer_all(messages, on_answer))

    async def answer_all(self, messages: list[str], on_answer: Callable[[int, str], None] | None) -> list[str]:
        semaphore = asyncio.Semaphore(self.config.concurrency)
        limits = httpx.Limits(max_connections=self.config.concurrency)
        async with httpx.AsyncClient(timeout=httpx.Timeout(600), limits=limits) as client:

            async def one(i: int, message: str) -> str:
                async with semaphore:
                    answer = await self.request(client, message)
                if on_answer:
                    on_answer(i, answer)
                return answer

            return list(await asyncio.gather(*(one(i, message) for i, message in enumerate(messages))))

    async def request(self, client: httpx.AsyncClient, message: str) -> str:
        config = self.config
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}],
            "temperature": 0.0,
            "max_tokens": config.think_tokens if config.thinking else config.max_new_tokens,
            # vLLM passes this to the chat template; Qwen3-style models then answer without thinking
            "chat_template_kwargs": {"enable_thinking": config.thinking},
        }
        for attempt in range(config.retries):
            try:
                response = await client.post(f"{self.url}/chat/completions", json=body)
                response.raise_for_status()
                choice = response.json()["choices"][0]
                break
            except httpx.HTTPError as error:
                if attempt == config.retries - 1:
                    raise
                tqdm.write(f"request failed ({error!r}), retrying")
                await asyncio.sleep(2**attempt)
        text = choice["message"]["content"] or ""
        if THINK_END in text:
            return text.rsplit(THINK_END, 1)[1].strip()
        # out of tokens mid-thought: what is there is reasoning, not an answer
        if config.thinking and choice.get("finish_reason") == "length":
            return ""
        return text.strip()


def connect(config: ZeroShotConfig) -> tuple[Client, Prompter]:
    """The client for the served model, and the prompter that measures prompts with its tokenizer."""
    from transformers import AutoTokenizer

    model = config.model or served_model(config.url)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer or model)
    logger.info(f"answering with {model} at {config.url}")
    return Client(config, model), Prompter(tokenizer, config.max_length)
