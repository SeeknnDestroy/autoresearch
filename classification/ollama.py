"""Minimal Ollama HTTP client for local evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


class OllamaError(RuntimeError):
    """Raised when the local Ollama API is unavailable or returns bad data."""


@dataclass(frozen=True)
class OllamaGeneration:
    raw_output: str
    latency_ms: float
    model: str


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 120,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=self.timeout_seconds, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise OllamaError(f"Failed to reach Ollama at {url}: {exc}") from exc

    def list_models(self) -> list[str]:
        response = self._request("GET", "/api/tags")
        payload = response.json()
        models = payload.get("models", [])
        return [str(item.get("name")) for item in models if item.get("name")]

    def ensure_model_available(self, model: str) -> None:
        available_models = self.list_models()
        if model not in available_models:
            raise OllamaError(
                f"Model {model!r} is not available in Ollama. "
                f"Install it first, for example with `ollama pull {model}`."
            )

    def generate_json(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        options: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> OllamaGeneration:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        }
        if schema is not None:
            payload["format"] = schema
        response = self._request("POST", "/api/generate", json=payload)
        data = response.json()
        raw_output = data.get("response")
        if not isinstance(raw_output, str):
            raise OllamaError(f"Ollama response did not include a string `response`: {data!r}")
        duration_ns = data.get("total_duration")
        if isinstance(duration_ns, (int, float)):
            latency_ms = float(duration_ns) / 1_000_000.0
        else:
            latency_ms = (time.perf_counter() - started) * 1000.0
        return OllamaGeneration(raw_output=raw_output, latency_ms=latency_ms, model=model)
