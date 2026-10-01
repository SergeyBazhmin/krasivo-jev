"""Translate built datasets to Russian through an OpenAI-compatible server, such as a local vLLM."""

import asyncio
import json
import re
from pathlib import Path

import httpx
from datasets import DatasetDict, load_from_disk
from loguru import logger
from tqdm import tqdm

SYSTEM = """You translate text from English into Russian.
Reply with the translation only: no notes, no quotes, no explanations.
Keep the meaning, line breaks, lists, numbering and punctuation.
Leave code, URLs, e-mail addresses, file paths, variable names and other identifiers as they are.
If the text is already Russian or has nothing to translate (a number, a symbol, code), repeat it unchanged."""

# reasoning models may think aloud before the answer even with thinking turned off
THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def texts(data: DatasetDict) -> set[str]:
    """Every distinct state, question and option text. Questions come from a few paraphrases and option texts
    repeat across rows, so translating each distinct string once is far cheaper than translating rows."""
    found = set()
    for split in data.values():
        found.update(split["state"])
        found.update(split["question"])
        for options in split["options"]:
            found.update(option["text"] for option in options)
    found.discard("")
    return found


class Translator:
    """Translates strings through `url`/chat/completions and keeps every translation in a JSONL cache, so an
    interrupted run resumes where it stopped and texts shared by several datasets are translated once."""

    def __init__(self, url: str, model: str | None, cache_path: Path, concurrency: int = 64, retries: int = 5):
        self.url = url.rstrip("/")
        self.model = model or self.served_model()
        self.cache_path = cache_path
        self.concurrency = concurrency
        self.retries = retries
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            with cache_path.open() as f:
                for line in f:
                    entry = json.loads(line)
                    self.cache[entry["source"]] = entry["target"]
        logger.info(f"translating with {self.model} at {self.url}, {len(self.cache)} cached")

    def served_model(self) -> str:
        """The first model the server lists."""
        response = httpx.get(f"{self.url}/models", timeout=30)
        response.raise_for_status()
        return response.json()["data"][0]["id"]

    def translate(self, sources: set[str]) -> dict[str, str]:
        """{source: translation} for `sources`, translating the ones that are not cached yet."""
        missing = sorted(sources - self.cache.keys())
        if missing:
            asyncio.run(self.translate_all(missing))
        return {source: self.cache[source] for source in sources}

    async def translate_all(self, sources: list[str]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency)
        async with httpx.AsyncClient(timeout=httpx.Timeout(600), limits=limits) as client:

            async def one(source: str) -> tuple[str, str]:
                async with semaphore:
                    return source, await self.request(client, source)

            with self.cache_path.open("a") as f, tqdm(total=len(sources), desc="translate") as progress:
                for task in asyncio.as_completed([one(source) for source in sources]):
                    source, target = await task
                    self.cache[source] = target
                    f.write(json.dumps({"source": source, "target": target}, ensure_ascii=False) + "\n")
                    f.flush()
                    progress.update()

    async def request(self, client: httpx.AsyncClient, source: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": source}],
            "temperature": 0.0,
            # vLLM passes this to the chat template; Qwen3-style models then answer without thinking
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(self.retries):
            try:
                response = await client.post(f"{self.url}/chat/completions", json=body)
                response.raise_for_status()
                choice = response.json()["choices"][0]
                break
            except httpx.HTTPError as error:
                if attempt == self.retries - 1:
                    raise
                logger.warning(f"request failed ({error!r}), retrying")
                await asyncio.sleep(2**attempt)
        target = THINK.sub("", choice["message"]["content"] or "").strip()
        # a cut-off or empty translation would silently corrupt the sample; the source is the lesser evil
        if choice.get("finish_reason") == "length" or not target:
            logger.warning(f"no full translation, keeping the source: {source[:80]!r}")
            return source
        return target


def translate_dataset(data: DatasetDict, translator: Translator) -> DatasetDict:
    """`data` with its state, question and option texts in Russian; ids, labels and types are kept."""
    table = translator.translate(texts(data))
    table[""] = ""

    def convert(sample: dict) -> dict:
        return {
            "state": table[sample["state"]],
            "question": table[sample["question"]],
            "options": [{"id": o["id"], "text": table[o["text"]]} for o in sample["options"]],
        }

    return data.map(convert, features=next(iter(data.values())).features)


def translate_saved(name: str, input_dir: Path, output_dir: Path, translator: Translator) -> None:
    data = load_from_disk(input_dir / name)
    translated = translate_dataset(data, translator)
    translated.save_to_disk(output_dir / name)
    logger.info(f"[{name}] saved {dict(translated.num_rows)} to {output_dir / name}")
