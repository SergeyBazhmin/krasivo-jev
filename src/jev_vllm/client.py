"""A client for an exported `pointer` run served by vLLM (`uv run serve`), shaped like `typesafe-sdk`'s: the same
`system_one(state, questions)`, the same question types in, and the SDK's `SystemOneResponse` and `TypeSafe*Error`s
out. Code written against `TypeSafeClient` switches to the local model by swapping the client.

    from typesafe_sdk import Choice, Noul
    from jev_vllm.client import JevClient

    with JevClient() as client:  # http://localhost:8080, model "pointer"
        response = client.system_one(
            state="I was charged twice. Please help.",
            questions={
                "billing": Noul(instructions="Is this about billing?"),
                "tone": Choice(instructions="What is the tone?", criteria={"angry": None, "calm": "polite"}),
            },
        )
    response.nouls["billing"].noul, response.choices["tone"].choice

Imports neither vLLM nor torch, so it runs anywhere `typesafe-sdk` and `httpx` are installed."""

import dataclasses
import json
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import BaseModel
from typesafe_sdk import (
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeInternalServerError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

DEFAULT_BASE_URL = "http://localhost:8080"
# `--served-model-name` in `jev_vllm.serve`
DEFAULT_MODEL = "pointer"
POOLING_PATH = "/pooling"

# vLLM answers a request the plugin rejects (bad question, too long) with 400, an unknown model with 404
STATUS_ERRORS: dict[int, type[TypeSafeAPIError]] = {
    400: TypeSafeBadRequestError,
    401: TypeSafeAuthenticationError,
    403: TypeSafePermissionDeniedError,
    404: TypeSafeNotFoundError,
    422: TypeSafeUnprocessableEntityError,
    429: TypeSafeRateLimitError,
}


def text(value: Any) -> str:
    """The plugin reads plain text; TypeSafe also takes JSON objects and arrays, which go in as compact JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def wire_question(name: str, question: Any) -> dict[str, Any]:
    """A `typesafe_sdk` `Noul`/`Choice`, its dict form, or `jev_model.system_one`'s `Noul`/`Choice`, as the plugin's
    `{"type", "instructions"[, "criteria"]}`. Fails here on what the plugin can't answer, before any request."""
    if isinstance(question, BaseModel):
        raw = question.model_dump()
    elif dataclasses.is_dataclass(question) and not isinstance(question, type):
        raw = dataclasses.asdict(question)
        raw["type"] = "choice" if "criteria" in raw else "noul"
    elif isinstance(question, Mapping):
        raw = dict(question)
    else:
        raise TypeError(f"question {name!r}: expected Noul, Choice or a question dict, got {type(question).__name__}")
    kind = raw.get("type")
    wire: dict[str, Any] = {"type": kind, "instructions": text(raw.get("instructions") or "")}
    if kind == "noul":
        if raw.get("criteria"):
            raise ValueError(f"question {name!r}: the pointer model takes no noul criteria")
    elif kind == "choice":
        criteria = raw.get("criteria") or {}
        if len(criteria) < 2:
            raise ValueError(f"question {name!r}: a choice needs at least 2 criteria, got {len(criteria)}")
        wire["criteria"] = {str(key): None if value is None else text(value) for key, value in criteria.items()}
    else:
        raise ValueError(f"question {name!r}: the pointer model answers noul and choice questions, not {kind!r}")
    return wire


def body(model: str, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
    if not questions:
        raise ValueError("questions must not be empty")
    data = {"state": text(state), "questions": {name: wire_question(name, q) for name, q in questions.items()}}
    return {"model": model, "data": data}


def parse(response: httpx.Response, model: str) -> SystemOneResponse:
    """vLLM wraps the plugin's `{"answers": ...}` in `{"request_id", "created_at", "data"}`; errors come as
    `{"error": {"message", "type", "code"}}`."""
    try:
        decoded = response.json()
    except ValueError:
        decoded = response.text or None
    if response.is_error:
        error = STATUS_ERRORS.get(response.status_code)
        if error is None:
            error = TypeSafeInternalServerError if response.status_code >= 500 else TypeSafeAPIError
        raise error(response.status_code, decoded, response.headers, endpoint=POOLING_PATH)
    answers = decoded.get("data", {}).get("answers") if isinstance(decoded, dict) else None
    if not isinstance(answers, dict):
        raise TypeSafeAPIResponseValidationError(
            response.status_code, decoded, response.headers, "data.answers", endpoint=POOLING_PATH
        )
    # the server reports no token counts, so `usage` stays empty
    return SystemOneResponse.model_validate({"model": model, "usage": {}, "answers": answers})


class JevClient:
    """`TypeSafeClient.system_one` against a vLLM server running the `jev_vllm` plugins. All questions of a call go
    to the server in one request and are answered in one batch."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float | httpx.Timeout | None = 60.0,
        headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self._http_client = http_client or httpx.Client(base_url=base_url, timeout=timeout, headers=headers)

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> SystemOneResponse:
        """Answers to named `noul` and `choice` questions about `state` (text, or JSON sent as compact JSON)."""
        model = model or self.model
        try:
            response = self._http_client.post(
                POOLING_PATH, json=body(model, state, questions), timeout=timeout or self.timeout
            )
        except httpx.TimeoutException as error:
            raise TypeSafeAPITimeoutError(timeout or self.timeout) from error
        except httpx.TransportError as error:
            raise TypeSafeAPIConnectionError(str(error)) from error
        return parse(response, model)

    def close(self) -> None:
        self._http_client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None):
        self.close()


class AsyncJevClient:
    """`AsyncTypeSafeClient.system_one` against a vLLM server running the `jev_vllm` plugins. vLLM batches
    concurrent requests itself, so `asyncio.gather` over many calls keeps the GPU busy."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float | httpx.Timeout | None = 60.0,
        headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self._http_client = http_client or httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=headers)

    async def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> SystemOneResponse:
        """Answers to named `noul` and `choice` questions about `state` (text, or JSON sent as compact JSON)."""
        model = model or self.model
        try:
            response = await self._http_client.post(
                POOLING_PATH, json=body(model, state, questions), timeout=timeout or self.timeout
            )
        except httpx.TimeoutException as error:
            raise TypeSafeAPITimeoutError(timeout or self.timeout) from error
        except httpx.TransportError as error:
            raise TypeSafeAPIConnectionError(str(error)) from error
        return parse(response, model)

    async def close(self) -> None:
        await self._http_client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ):
        await self.close()
