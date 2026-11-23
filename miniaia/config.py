"""Configuration.

Precedence (lowest -> highest):
    built-in defaults < OLLAMA_HOST < config.toml < MINIAIA_* env vars < CLI flags
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None  # type: ignore[assignment]


class ConfigError(ValueError):
    """Raised when the configuration is invalid."""


def normalize_host(host: str) -> str:
    """'127.0.0.1:11434' -> 'http://127.0.0.1:11434' (Ollama's own OLLAMA_HOST style)."""
    host = host.strip().rstrip("/")
    if "://" not in host:
        host = "http://" + host
    return host


@dataclass(frozen=True)
class Config:
    # --- Ollama connection -------------------------------------------------
    ollama_host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b"
    connect_timeout: float = 5.0     # seconds to establish the connection
    read_timeout: float = 120.0      # max silence between streamed tokens
    temperature: float = 0.2
    num_ctx: int = 4096              # context window requested from Ollama
    # --- Command execution -------------------------------------------------
    command_timeout: float = 30.0    # hard limit per command
    max_output_chars: int = 6000     # output kept per command
    interactive_sudo: bool = True    # let sudo ask for the password on the real terminal
    # --- Agent -------------------------------------------------------------
    max_agent_steps: int = 8
    memory_char_budget: int = 12000  # how much history is sent to the model
    # --- Files -------------------------------------------------------------
    log_dir: str = ""                # default: ~/.local/state/miniaia

    @property
    def state_dir(self) -> Path:
        if self.log_dir:
            return Path(self.log_dir).expanduser()
        base = os.environ.get("XDG_STATE_HOME")
        root = Path(base) if base else Path.home() / ".local" / "state"
        return root / "miniaia"

    @property
    def log_path(self) -> Path:
        return self.state_dir / "commands.log"

    @property
    def history_path(self) -> Path:
        return self.state_dir / "input_history"

    def validate(self) -> "Config":
        parsed = urlparse(self.ollama_host)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ConfigError(f"ollama_host must look like http://host:port (got {self.ollama_host!r})")
        if not self.model.strip():
            raise ConfigError("model must not be empty")
        for name in ("connect_timeout", "read_timeout", "command_timeout"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be > 0")
        if not 0 <= self.temperature <= 2:
            raise ConfigError("temperature must be between 0 and 2")
        if self.num_ctx < 512:
            raise ConfigError("num_ctx must be >= 512")
        if self.max_output_chars < 100:
            raise ConfigError("max_output_chars must be >= 100")
        if not 1 <= self.max_agent_steps <= 50:
            raise ConfigError("max_agent_steps must be between 1 and 50")
        if self.memory_char_budget < 1000:
            raise ConfigError("memory_char_budget must be >= 1000")
        return self


_DEFAULTS = Config()
_FIELD_NAMES = tuple(f.name for f in fields(Config))
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "miniaia" / "config.toml"


def _coerce(name: str, raw: Any) -> Any:
    default = getattr(_DEFAULTS, name)
    try:
        if isinstance(default, bool):
            if isinstance(raw, bool):
                return raw
            text = str(raw).strip().lower()
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise ValueError
        if isinstance(default, int):
            if isinstance(raw, bool):
                raise ValueError
            return int(raw)
        if isinstance(default, float):
            if isinstance(raw, bool):
                raise ValueError
            return float(raw)
        return str(raw)
    except (TypeError, ValueError):
        raise ConfigError(f"invalid value for {name!r}: {raw!r}") from None


def load_config(
    config_path: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Config:
    env = os.environ if env is None else env
    values: dict[str, Any] = {}

    if env.get("OLLAMA_HOST"):
        values["ollama_host"] = env["OLLAMA_HOST"]

    explicit = config_path or env.get("MINIAIA_CONFIG")
    path = Path(explicit).expanduser() if explicit else default_config_path()
    if path.is_file():
        if tomllib is None:
            raise ConfigError("Python >= 3.11 is required to read config.toml")
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read {path}: {exc}") from None
        unknown = sorted(set(data) - set(_FIELD_NAMES))
        if unknown:
            raise ConfigError(f"unknown key(s) in {path}: {', '.join(unknown)}")
        values.update(data)
    elif explicit:
        raise ConfigError(f"config file not found: {path}")

    for name in _FIELD_NAMES:
        key = "MINIAIA_" + name.upper()
        if key in env:
            values[name] = env[key]

    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})

    coerced = {name: _coerce(name, raw) for name, raw in values.items()}
    if "ollama_host" in coerced:
        coerced["ollama_host"] = normalize_host(coerced["ollama_host"])
    return Config(**coerced).validate()
