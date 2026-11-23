"""Thin Ollama HTTP client (/api/tags, /api/version, streaming /api/chat).

It knows nothing about commands or security: it only talks to the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import requests

from .config import Config


class OllamaError(RuntimeError):
    """Any problem talking to Ollama."""


class OllamaConnectionError(OllamaError):
    """Ollama is not reachable."""


class OllamaModelNotFound(OllamaError):
    """The configured model is not installed."""


@dataclass
class OllamaStatus:
    reachable: bool
    model_available: bool = False
    models: list = field(default_factory=list)
    version: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.reachable and self.model_available


def model_matches(wanted: str, installed: str) -> bool:
    if wanted == installed:
        return True
    return ":" not in wanted and installed == wanted + ":latest"


def _describe(exc: Exception) -> str:
    if isinstance(exc, requests.ConnectionError):
        return "connection refused or host unreachable"
    if isinstance(exc, requests.Timeout):
        return "timed out"
    return str(exc)


class OllamaClient:
    def __init__(self, config: Config, session: Optional[requests.Session] = None):
        self.config = config
        self.session = session or requests.Session()
        self.base_url = config.ollama_host.rstrip("/")

    def _url(self, path: str) -> str:
        return self.base_url + path

    # ------------------------------------------------------------------ status
    def status(self) -> OllamaStatus:
        try:
            resp = self.session.get(self._url("/api/tags"), timeout=(self.config.connect_timeout, 10))
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
        except requests.RequestException as exc:
            return OllamaStatus(False, error=_describe(exc))
        except (ValueError, AttributeError):
            return OllamaStatus(False, error="unexpected response (is this really an Ollama server?)")

        version = None
        try:
            vresp = self.session.get(self._url("/api/version"), timeout=(self.config.connect_timeout, 5))
            version = vresp.json().get("version")
        except (requests.RequestException, ValueError, AttributeError):
            pass

        available = any(model_matches(self.config.model, m) for m in models)
        return OllamaStatus(True, available, models, version)

    # -------------------------------------------------------------------- chat
    def chat(
        self,
        messages: Sequence[dict],
        *,
        json_format: Any = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Stream a chat completion; return the full text.

        json_format: None, "json", or a JSON-schema dict (Ollama structured outputs).
        on_token: called with every streamed chunk (used by the UI for progress).
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "stream": True,
            "keep_alive": "10m",
            "options": {"temperature": self.config.temperature, "num_ctx": self.config.num_ctx},
        }
        if json_format is not None:
            payload["format"] = json_format

        parts: list[str] = []
        try:
            with self.session.post(
                self._url("/api/chat"),
                json=payload,
                stream=True,
                timeout=(self.config.connect_timeout, self.config.read_timeout),
            ) as resp:
                if resp.status_code == 404:
                    raise OllamaModelNotFound(
                        f"model {self.config.model!r} not found. Install it with: ollama pull {self.config.model}"
                    )
                if resp.status_code >= 400:
                    raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {self._error_text(resp)}")
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except ValueError:
                        raise OllamaError("received malformed data from Ollama") from None
                    if data.get("error"):
                        raise OllamaError(f"Ollama error: {data['error']}")
                    chunk = (data.get("message") or {}).get("content", "")
                    if chunk:
                        parts.append(chunk)
                        if on_token:
                            on_token(chunk)
                    if data.get("done"):
                        break
        except requests.ConnectionError:
            raise OllamaConnectionError(
                f"cannot reach Ollama at {self.base_url}. Is it running? (try: sudo systemctl start ollama, or: ollama serve)"
            ) from None
        except requests.Timeout:
            raise OllamaError(
                f"Ollama stopped responding for {self.config.read_timeout:.0f}s (model still loading? raise read_timeout)"
            ) from None
        except requests.RequestException as exc:
            raise OllamaError(f"request to Ollama failed: {exc}") from None
        return "".join(parts)

    @staticmethod
    def _error_text(resp: requests.Response) -> str:
        try:
            return str(resp.json().get("error") or resp.text[:200])
        except ValueError:
            return resp.text[:200]
