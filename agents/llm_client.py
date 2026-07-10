"""LLM client — single entry point for Ollama, OpenAI, Gemini, and Groq.

All providers are accessed via the openai Python package with a base_url swap.
API keys are read from environment variables and never logged or stored to disk.

Profiles: ``call(..., profile="report")`` merges the optional ``llm_report:`` settings
block over the main ``llm:`` block (absent block → main block used entirely), and lets
that profile use a DIFFERENT API key (``api_key_env`` in the block, else
``REPORT_<PROVIDER>_API_KEY``, else the provider's standard env var). This is what lets
the Stage 8 insight report run on its own provider/model/key without touching the
pipeline agents.
"""
import json
import os
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "settings.yaml"

_PROVIDER_DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
}


class LLMError(Exception):
    pass


def _load_llm_config(profile: str = "default") -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.get("llm", {}) or {}
    if profile == "report":
        override = cfg.get("llm_report") or {}
        if override:
            return {**base, **override, "_profile": "report"}
    return {**base, "_profile": "default"}


def describe(profile: str = "default") -> str:
    """'provider/model' string for UI attribution — never touches keys."""
    cfg = _load_llm_config(profile)
    return f"{cfg.get('provider', '?')}/{cfg.get('model', '?')}"


def call(messages: list[dict], *, require_json: bool = True,
         profile: str = "default") -> dict:
    """Send messages to the configured LLM and return the parsed response.

    Raises LLMError on any failure — callers must catch it and fall back.
    API key is read from the environment at call time; never logged.
    """
    cfg = _load_llm_config(profile)
    provider = cfg.get("provider", "ollama").lower()
    model = cfg.get("model", "llama3.1:8b")
    timeout = float(cfg.get("timeout_seconds", 60))
    max_retries = int(cfg.get("max_retries", 2))

    if provider not in _PROVIDER_DEFAULTS:
        raise LLMError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {list(_PROVIDER_DEFAULTS)}. Check config/settings.yaml."
        )

    defaults = _PROVIDER_DEFAULTS[provider]

    if provider == "ollama":
        api_key = "ollama"  # Ollama's OpenAI-compat endpoint accepts any non-empty string
    else:
        # First non-empty wins: explicit name in the settings block → the report
        # profile's dedicated key → the provider's standard key. Values never logged.
        candidates = []
        if cfg.get("api_key_env"):
            candidates.append(str(cfg["api_key_env"]))
        if cfg.get("_profile") == "report":
            candidates.append(f"REPORT_{provider.upper()}_API_KEY")
        candidates.append(defaults["api_key_env"])
        api_key = next((os.environ[k] for k in candidates if os.environ.get(k)), "")
        if not api_key:
            raise LLMError(
                f"API key not found for provider '{provider}'. "
                f"Set one of {candidates} in your .env file."
            )

    # base_url in settings only overrides for ollama / custom endpoints; ignored for cloud providers
    base_url = (cfg.get("base_url") if provider == "ollama" else None) or defaults["base_url"]

    # Gemini's OpenAI-compat endpoint takes the key as a normal Bearer header like the
    # others. Do NOT append it as a ?key= URL parameter — the OpenAI client joins the
    # request path AFTER the query string (".../openai/?key=X/chat/completions"), which
    # 404s on every call.
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )

    kwargs: dict = {
        "model": model,
        "messages": messages,
        # Configurable in settings.yaml — low default keeps recipes stable run-to-run
        # (provider default of 1.0 produced wildly inconsistent strategies).
        "temperature": float(cfg.get("temperature", 0.2)),
    }
    if require_json:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        raise LLMError(f"LLM request failed ({provider}/{model}): {e}") from e

    content = response.choices[0].message.content or ""
    if require_json:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError(
                f"LLM returned invalid JSON: {e}\nFirst 500 chars: {content[:500]}"
            ) from e

    return {"text": content}
