"""LLM client — single entry point for Ollama, OpenAI, and Gemini.

All three providers are accessed via the openai Python package with a base_url swap.
API keys are read from environment variables and never logged or stored to disk.
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


def _load_llm_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("llm", {})


def call(messages: list[dict], *, require_json: bool = True) -> dict:
    """Send messages to the configured LLM and return the parsed response.

    Raises LLMError on any failure — callers must catch it and fall back.
    API key is read from the environment at call time; never logged.
    """
    cfg = _load_llm_config()
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
    # base_url in settings only overrides for ollama / custom endpoints; ignored for cloud providers
    base_url = (cfg.get("base_url") if provider == "ollama" else None) or defaults["base_url"]

    if provider == "ollama":
        api_key = "ollama"  # Ollama's OpenAI-compat endpoint accepts any non-empty string
    else:
        key_env = defaults["api_key_env"]
        api_key = os.environ.get(key_env, "")
        if not api_key:
            raise LLMError(
                f"API key not found for provider '{provider}'. "
                f"Set '{key_env}' in your .env file."
            )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )

    kwargs: dict = {"model": model, "messages": messages}
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
