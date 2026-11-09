"""Pluggable deterministic and hosted/local language-model adapters."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


class LLM(Protocol):
    def generate(
        self,
        system: str,
        user: Any,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult: ...


class MockLLM:
    """Offline deterministic generator used for tests and reproducible smoke runs."""

    def __init__(self, hallucinate: bool = False):
        self.hallucinate = hallucinate

    def generate(
        self,
        system: str,
        user: Any,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        started = time.perf_counter()
        payload = json.loads(user) if isinstance(user, str) else user
        if json_schema is not None:
            text = json.dumps({"claims": []}, sort_keys=True)
        else:
            facts = payload.get("facts", [])
            destination = payload.get("destination") or "your destination"
            nationality = payload.get("nationality") or "the stated"
            sentences = []
            for fact in facts:
                slot, value, citation = fact["slot"], fact["value"], fact["evidence_id"]
                if slot == "max_stay_days":
                    nationality_label = {"India": "Indian", "United Kingdom": "British", "United States": "American"}.get(nationality, nationality)
                    sentences.append(
                        f"For {nationality_label} passport holders travelling to {destination}, "
                        f"the listed maximum stay is {value} days [{citation}]."
                    )
                elif slot == "visa_type":
                    labels = {
                        "evisa": "an eVisa",
                        "visa_free": "visa-free entry",
                        "visa_on_arrival": "a visa on arrival",
                        "embassy_visa": "an embassy-issued visa",
                    }
                    sentences.append(f"The source describes {labels.get(str(value), str(value))} [{citation}].")
                elif slot == "vaccine_required":
                    phrase = "is required" if value else "is not generally required"
                    sentences.append(f"Vaccination proof {phrase} [{citation}].")
                elif slot == "item_allowed":
                    source_text = next(
                        (item.get("text", "").lower() for item in payload.get("evidence", []) if item.get("evidence_id") == citation),
                        "",
                    )
                    if value == "restricted" and any(term in source_text for term in ("medicine", "medication", "drug")):
                        sentences.append(
                            f"Check whether approval is needed for this specific medicine before bringing it into {destination} [{citation}]."
                        )
                    else:
                        sentences.append(f"The item is listed as {value} [{citation}].")
                elif slot == "declaration_required":
                    phrase = "must be declared" if value else "does not need to be declared"
                    sentences.append(f"The item {phrase} [{citation}].")
            text = " ".join(sentences) or "I couldn't find a supported answer in the saved evidence."
            if self.hallucinate:
                text += " Visa on arrival is available at all airports [E99]."
        return LLMResult(text=text, latency_ms=(time.perf_counter() - started) * 1000)


class OllamaLLM:
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434", timeout_s: float = 60.0):
        self.model, self.base_url, self.timeout_s = model, base_url.rstrip("/"), timeout_s

    def generate(self, system: str, user: Any, *, temperature: float = 0.0, max_tokens: int = 512, json_schema: dict[str, Any] | None = None) -> LLMResult:
        body = {
            "model": self.model,
            "stream": False,
            "format": json_schema,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 42},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user if isinstance(user, str) else json.dumps(user)},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    result = json.loads(response.read().decode("utf-8"))
                return LLMResult(
                    text=result.get("message", {}).get("content", ""),
                    tokens_in=int(result.get("prompt_eval_count", 0)),
                    tokens_out=int(result.get("eval_count", 0)),
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))
        raise RuntimeError("Ollama request failed")


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install the openai package to use this backend") from error
        self.client, self.model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), model

    def generate(self, system: str, user: Any, *, temperature: float = 0.0, max_tokens: int = 512, json_schema: dict[str, Any] | None = None) -> LLMResult:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user if isinstance(user, str) else json.dumps(user)}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = response.usage
        return LLMResult(response.choices[0].message.content or "", usage.prompt_tokens, usage.completion_tokens, (time.perf_counter() - started) * 1000)


class AnthropicLLM:
    def __init__(self, model: str = "claude-3-5-haiku-latest"):
        try:
            from anthropic import Anthropic
        except ImportError as error:
            raise RuntimeError("Install the anthropic package to use this backend") from error
        self.client, self.model = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")), model

    def generate(self, system: str, user: Any, *, temperature: float = 0.0, max_tokens: int = 512, json_schema: dict[str, Any] | None = None) -> LLMResult:
        started = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user if isinstance(user, str) else json.dumps(user)}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return LLMResult(text, response.usage.input_tokens, response.usage.output_tokens, (time.perf_counter() - started) * 1000)


def build_llm(name: str, config: dict[str, Any]) -> LLM:
    section = config.get("llm", {})
    backend = section.get(name, {})
    if name == "mock":
        return MockLLM()
    if name == "ollama":
        return OllamaLLM(str(backend.get("model", "llama3.1:8b")), str(backend.get("base_url", "http://localhost:11434")))
    if name == "openai":
        return OpenAILLM(str(backend.get("model", "gpt-4o-mini")))
    if name == "anthropic":
        return AnthropicLLM(str(backend.get("model", "claude-3-5-haiku-latest")))
    raise ValueError(f"Unsupported LLM backend: {name}")