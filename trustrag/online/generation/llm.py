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
            nationality_label = {"India": "Indian", "United Kingdom": "British", "United States": "American"}.get(nationality, nationality)
            sentences = []
            for fact in facts:
                slot, value, citation = fact["slot"], fact["value"], fact["evidence_id"]
                if slot == "visa_required":
                    phrase = "is required" if value else "is not required"
                    sentences.append(
                        f"A visa {phrase} for {nationality_label} passport holders travelling to {destination} [{citation}]."
                    )
                elif slot == "passport_validity_months":
                    sentences.append(
                        f"Passports must be valid for at least {value} months on entry to {destination} [{citation}]."
                    )
                elif slot == "vaccine_name":
                    sentences.append(f"The vaccine checked for {destination} is {value} [{citation}].")
                elif slot == "certificate_required":
                    phrase = "is required" if value else "is not required"
                    sentences.append(f"A vaccination certificate {phrase} to enter {destination} [{citation}].")
                elif slot == "duty_free_limit":
                    amount, currency = value
                    sentences.append(
                        f"The duty-free allowance for {destination} is {currency} {float(amount):,.0f} [{citation}]."
                    )
                elif slot == "max_stay_days":
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
                    phrase = "is required" if value else "is not required"
                    sentences.append(f"A customs declaration {phrase} in {destination} [{citation}].")
            text = " ".join(sentences) or "I couldn't find a supported answer in the saved evidence."
            if self.hallucinate:
                text += " Visa on arrival is available at all airports [E99]."
        return LLMResult(text=text, latency_ms=(time.perf_counter() - started) * 1000)


class OllamaLLM:
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60.0,
        api_key: str | None = None,
    ):
        """Local Ollama (no key) or Ollama Cloud (base_url https://ollama.com plus an API key)."""
        self.model, self.base_url, self.timeout_s, self.api_key = model, base_url.rstrip("/"), timeout_s, api_key

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
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
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
            except urllib.error.HTTPError as error:
                # 4xx (bad key, no credits, unknown model) will not succeed on retry.
                if 400 <= error.code < 500 and error.code != 429:
                    detail = error.read().decode("utf-8", errors="replace")[:200]
                    raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))
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
    def __init__(self, model: str = "claude-haiku-5-5"):
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
    if name == "ollama_cloud":
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set OLLAMA_API_KEY to use Ollama Cloud")
        return OllamaLLM(
            str(backend.get("model", "gpt-oss:20b")),
            str(backend.get("base_url", "https://ollama.com")),
            api_key=api_key,
        )
    if name == "openai":
        return OpenAILLM(str(backend.get("model", "gpt-4o-mini")))
    if name == "anthropic":
        return AnthropicLLM(str(backend.get("model", "claude-haiku-5-5")))
    raise ValueError(f"Unsupported LLM backend: {name}")


class FallbackLLM:
    """Use the chosen backend; if it cannot be reached, answer with the template generator.

    The template generator only restates the verified facts with citations, so falling back
    keeps the answer evidence-bound. `fallback_reason` records why, for the response flags.
    """

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.fallback_reason: str | None = None
        self._template = MockLLM()
        try:
            self._primary: LLM | None = build_llm(name, config)
        except Exception as error:  # missing package, bad config
            self._primary = None
            self.fallback_reason = type(error).__name__

    def generate(self, system: str, user: Any, **kwargs: Any) -> LLMResult:
        if self._primary is not None:
            try:
                return self._primary.generate(system, user, **kwargs)
            except Exception as error:  # unreachable server, missing key, rate limit
                self.fallback_reason = type(error).__name__
        return self._template.generate(system, user, **kwargs)

