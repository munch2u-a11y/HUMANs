from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Mapping, Protocol, Sequence, runtime_checkable


ChatMessage = Mapping[str, str]


class ModelUnavailableError(RuntimeError):
    pass


@runtime_checkable
class ChatModel(Protocol):
    def generate(self, messages: Sequence[ChatMessage]) -> str: ...


class OllamaChatModel:
    """Small standard-library adapter for Ollama's local chat endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 180.0,
        temperature: float = 0.7,
        context_tokens: int = 8192,
        num_gpu: int | None = None,
        maximum_tokens: int = 512,
        think: bool | None = False,
    ):
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("Ollama model name cannot be empty")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.temperature = max(0.0, float(temperature))
        self.context_tokens = max(512, int(context_tokens))
        self.num_gpu = None if num_gpu is None else max(0, int(num_gpu))
        self.maximum_tokens = max(1, int(maximum_tokens))
        self.think = None if think is None else bool(think)

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        options: dict[str, float | int] = {
            "temperature": self.temperature,
            "num_ctx": self.context_tokens,
            "num_predict": self.maximum_tokens,
        }
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        request_payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "options": options,
        }
        if self.think is not None:
            request_payload["think"] = self.think
        payload = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise ModelUnavailableError(
                f"Ollama chat failed for {self.model!r} at {self.base_url}: {error}"
            ) from error
        content = str(result.get("message", {}).get("content", "")).strip()
        if not content:
            detail = result.get("error") or "the model returned no ordinary response content"
            raise ModelUnavailableError(str(detail))
        return content
