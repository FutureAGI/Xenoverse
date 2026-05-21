import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, parse, request


class LLMRequestError(RuntimeError):
    pass


def _http_post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int = 120,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise LLMRequestError(f"HTTP {exc.code} from {url}: {body}") from exc
    except error.URLError as exc:
        raise LLMRequestError(f"Network error from {url}: {exc}") from exc


@dataclass
class ClientConfig:
    model: str
    temperature: float = 0.2
    max_tokens: int = 256
    timeout_sec: int = 120
    max_history_messages: int = 80


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    llm_calls: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.llm_calls += other.llm_calls

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "llm_calls": self.llm_calls,
        }


def _parse_openai_compatible_usage(body: Dict[str, Any]) -> TokenUsage:
    usage = body.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    details = usage.get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens") or 0)
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reasoning_tokens=reasoning,
        llm_calls=1,
    )


def _parse_anthropic_usage(body: Dict[str, Any]) -> TokenUsage:
    usage = body.get("usage") or {}
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        llm_calls=1,
    )


def _parse_gemini_usage(body: Dict[str, Any]) -> TokenUsage:
    usage = body.get("usageMetadata") or {}
    prompt = int(usage.get("promptTokenCount") or 0)
    completion = int(usage.get("candidatesTokenCount") or 0)
    total = int(usage.get("totalTokenCount") or (prompt + completion))
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        llm_calls=1,
    )


def _parse_openai_responses_usage(body: Dict[str, Any]) -> TokenUsage:
    usage = body.get("usage") or {}
    prompt = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    completion = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        llm_calls=1,
    )


class BaseLLMClient:
    def __init__(self) -> None:
        self.last_usage = TokenUsage()
        self.total_usage = TokenUsage()

    def _record_usage(self, usage: TokenUsage) -> None:
        self.last_usage = usage
        self.total_usage.add(usage)

    def reset_usage(self) -> None:
        self.last_usage = TokenUsage()
        self.total_usage = TokenUsage()

    def usage_summary(self) -> Dict[str, int]:
        return self.total_usage.to_dict()

    def reset_task(self, system_prompt: str, task_prompt: str) -> str:
        raise NotImplementedError

    def choose_action(self, step_prompt: str) -> str:
        raise NotImplementedError


class OpenAIResponsesClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str], config: ClientConfig, api_base: str = "https://api.openai.com"):
        super().__init__()
        self.api_key = api_key
        self.config = config
        self.api_base = api_base.rstrip("/")
        self.previous_response_id: Optional[str] = None
        self.system_prompt = ""

    def _request(self, user_input: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "input": user_input,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "instructions": self.system_prompt,
        }
        if self.previous_response_id is not None:
            payload["previous_response_id"] = self.previous_response_id

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = _http_post_json(
            url=f"{self.api_base}/v1/responses",
            payload=payload,
            headers=headers,
            timeout=self.config.timeout_sec,
        )
        self.previous_response_id = body.get("id")
        self._record_usage(_parse_openai_responses_usage(body))
        text = body.get("output_text", "")
        if not text:
            # Compatibility fallback for OpenAI-compatible /v1/responses variants.
            out_items = body.get("output") or []
            txt_parts: List[str] = []
            for item in out_items:
                for c in item.get("content", []):
                    if c.get("type") == "output_text" and c.get("text"):
                        txt_parts.append(c["text"])
            text = "\n".join(txt_parts).strip()
        if not text:
            raise LLMRequestError(f"Empty output_text from OpenAI response body: {body}")
        return text

    def reset_task(self, system_prompt: str, task_prompt: str) -> str:
        self.previous_response_id = None
        self.system_prompt = system_prompt
        return self._request(task_prompt)

    def choose_action(self, step_prompt: str) -> str:
        return self._request(step_prompt)


class AnthropicMessagesClient(BaseLLMClient):
    def __init__(self, api_key: str, config: ClientConfig, api_base: str = "https://api.anthropic.com"):
        super().__init__()
        self.api_key = api_key
        self.config = config
        self.api_base = api_base.rstrip("/")
        self.system_prompt = ""
        self.messages: List[Dict[str, Any]] = []

    def _request(self, user_input: str, cache_hint: bool = False) -> str:
        content_block: Dict[str, Any] = {"type": "text", "text": user_input}
        if cache_hint:
            content_block["cache_control"] = {"type": "ephemeral"}
        self.messages.append({"role": "user", "content": [content_block]})

        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": self.messages,
        }

        body = _http_post_json(
            url=f"{self.api_base}/v1/messages",
            payload=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout_sec,
        )

        txt_parts = []
        for blk in body.get("content", []):
            if blk.get("type") == "text":
                txt_parts.append(blk.get("text", ""))
        text = "\n".join([p for p in txt_parts if p]).strip()
        if not text:
            raise LLMRequestError(f"Empty text content from Anthropic response body: {body}")

        self._record_usage(_parse_anthropic_usage(body))
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        return text

    def reset_task(self, system_prompt: str, task_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.messages = []
        return self._request(task_prompt, cache_hint=True)

    def choose_action(self, step_prompt: str) -> str:
        return self._request(step_prompt)


class OpenAICompatibleChatClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str], config: ClientConfig, api_base: str):
        super().__init__()
        self.api_key = api_key
        self.config = config
        self.api_base = api_base.rstrip("/")
        self.messages: List[Dict[str, str]] = []

    def _trim_messages(self) -> None:
        if not self.messages:
            return
        keep = max(2, int(self.config.max_history_messages))
        if len(self.messages) <= keep:
            return
        system = self.messages[0] if self.messages[0].get("role") == "system" else None
        tail = self.messages[-(keep - (1 if system else 0)) :]
        self.messages = ([system] if system else []) + tail

    def _request(self) -> str:
        self._trim_messages()
        payload = {
            "model": self.config.model,
            "messages": self.messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = _http_post_json(
            url=f"{self.api_base}/v1/chat/completions",
            payload=payload,
            headers=headers,
            timeout=self.config.timeout_sec,
        )
        try:
            msg = body["choices"][0]["message"]
            text = msg.get("content") or ""
            # Reasoning models (e.g. glm-4.7-flash) may put all output in
            # reasoning_content while content is empty.
            if not text.strip():
                text = msg.get("reasoning_content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(f"Unexpected OpenAI-compatible response body: {body}") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMRequestError(f"Empty text in OpenAI-compatible response body: {body}")
        self._record_usage(_parse_openai_compatible_usage(body))
        self.messages.append({"role": "assistant", "content": text})
        return text

    def reset_task(self, system_prompt: str, task_prompt: str) -> str:
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        return self._request()

    def choose_action(self, step_prompt: str) -> str:
        self.messages.append({"role": "user", "content": step_prompt})
        return self._request()


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, config: ClientConfig, api_base: str = "https://generativelanguage.googleapis.com"):
        super().__init__()
        self.api_key = api_key
        self.config = config
        self.api_base = api_base.rstrip("/")
        self.system_prompt = ""
        self.contents: List[Dict[str, Any]] = []

    def _request(self, user_input: str) -> str:
        self.contents.append({"role": "user", "parts": [{"text": user_input}]})
        query = parse.urlencode({"key": self.api_key})
        payload = {
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "contents": self.contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        body = _http_post_json(
            url=f"{self.api_base}/v1beta/models/{self.config.model}:generateContent?{query}",
            payload=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.config.timeout_sec,
        )
        try:
            parts = body["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(f"Unexpected Gemini response body: {body}") from exc
        text = "\n".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()
        if not text:
            raise LLMRequestError(f"Empty text in Gemini response body: {body}")
        self._record_usage(_parse_gemini_usage(body))
        self.contents.append({"role": "model", "parts": [{"text": text}]})
        return text

    def reset_task(self, system_prompt: str, task_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.contents = []
        return self._request(task_prompt)

    def choose_action(self, step_prompt: str) -> str:
        return self._request(step_prompt)


def build_client(
    provider: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout_sec: int = 120,
    api_base: Optional[str] = None,
) -> BaseLLMClient:
    p = provider.lower()
    if p == "gpt":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set")
        cfg = ClientConfig(
            model=model or "gpt-5",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        return OpenAIResponsesClient(api_key=key, config=cfg, api_base=api_base or "https://api.openai.com")

    if p == "claude":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        cfg = ClientConfig(
            model=model or "claude-sonnet-4-5",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        return AnthropicMessagesClient(api_key=key, config=cfg, api_base=api_base or "https://api.anthropic.com")

    if p == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        cfg = ClientConfig(
            model=model or "deepseek-v4",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            max_history_messages=80,
        )
        return OpenAICompatibleChatClient(api_key=key, config=cfg, api_base=api_base or "https://api.deepseek.com")

    if p == "gemini":
        key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
        cfg = ClientConfig(
            model=model or "gemini-2.5-pro",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        return GeminiClient(api_key=key, config=cfg, api_base=api_base or "https://generativelanguage.googleapis.com")

    if p == "local":
        key = os.getenv("LOCAL_OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        local_max_history = int(os.getenv("LOCAL_MAX_HISTORY_MESSAGES", "40"))
        cfg = ClientConfig(
            model=model or "nvidia/nemotron-3-super",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            max_history_messages=local_max_history,
        )
        # Local LM Studio: prefer chat/completions for faster per-step latency.
        # Keep bounded history via message window trimming.
        return OpenAICompatibleChatClient(
            api_key=key if key else None,
            config=cfg,
            api_base=api_base or "http://127.0.0.1:1234",
        )

    raise ValueError(f"Unsupported provider: {provider}")
