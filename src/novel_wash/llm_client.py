"""Multi-format LLM router (OpenAI, Anthropic, Gemini) with rate limiting, failover, and cost tracking."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import httpx


class RouterConfigError(RuntimeError):
    """Raised on fatal model configuration, disabled model, or authentication errors."""


class RouterContentPolicyError(RuntimeError):
    """Raised when an upstream provider blocks a specific prompt under content moderation."""


SUPPORTED_FORMATS = frozenset({"openai", "anthropic", "gemini"})

_CONTENT_POLICY_MARKERS = (
    "content_policy",
    "content_policy_violation",
    "content_filter",
    "content exists risk",
    "content moderation",
    "flagged by moderation",
    "moderation_blocked",
    "data_inspection_failed",
    "不安全或敏感内容",
    "内容审计",
    "敏感内容",
    "安全审核",
)
_AUTH_ERROR_MARKERS = (
    "invalid_api_key",
    "incorrect api key",
    "authentication_error",
    "invalid x-api-key",
    "api_key_invalid",
    "api key not valid",
    "permission_denied",
    "unauthorized",
    "user location is not supported",
)
_AUTH_ERROR_TYPES = frozenset({"authentication_error", "invalid_api_key", "permission_error"})
_ZHIPU_POLICY_CODE_RE = re.compile(r'"code"\s*:\s*(?:1301(?![0-9.])|"1301")')
_GEMINI_BLOCK_FINISH_REASONS = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"})
_THINK_PREFIX_RE = re.compile(r"^\s*<think>[\s\S]*?</think>\s*", re.I)
_SENSITIVE_QUERY_KEYS = frozenset({"key", "api_key", "apikey", "token", "access_token", "auth"})

_FORMAT_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}

_FORMAT_KEY_ENVS = {
    "openai": ("NOVEL_WASH_API_KEY", "OPENAI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def _extract_error_obj(body: str) -> dict[str, Any]:
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _is_auth_error_response(status_code: int, body: str) -> bool:
    if status_code < 400:
        return False
    if status_code == 401:
        return True
    obj = _extract_error_obj(body)
    err = obj.get("error") if isinstance(obj.get("error"), dict) else obj
    err_type = str(err.get("type") or "").strip().lower()
    err_code = str(err.get("code") or "").strip().lower()
    if err_type in _AUTH_ERROR_TYPES or err_code in _AUTH_ERROR_TYPES:
        return True
    low = body.lower()
    return any(tok in low for tok in _AUTH_ERROR_MARKERS)


def _is_content_policy_response(body: str) -> bool:
    obj = _extract_error_obj(body)
    if obj:
        err = obj.get("error") if isinstance(obj.get("error"), dict) else obj
        code = err.get("code")
        if code == 1301 or (isinstance(code, str) and code.strip() == "1301"):
            return True
    low = body.lower()
    return any(tok in low for tok in _CONTENT_POLICY_MARKERS) or bool(_ZHIPU_POLICY_CODE_RE.search(body))


def _is_rate_limit_response(status_code: int, body: str) -> bool:
    if status_code in (429, 529):
        return True
    if status_code >= 400:
        low = body.lower()
        return "resource_exhausted" in low or "overloaded_error" in low or "rate_limit" in low
    return False


def _is_context_length_response(body: str) -> bool:
    low = body.lower()
    return any(
        tok in low
        for tok in ("context_length_exceeded", "prompt is too long", "exceeds the maximum number of tokens")
    )


def _normalize_format(fmt: str | None) -> str:
    if not fmt:
        return "openai"
    low = fmt.strip().lower().replace("-", "_")
    if low in ("openai", "openai_chat", "openai_compatible"):
        return "openai"
    if low in ("anthropic", "claude"):
        return "anthropic"
    if low in ("gemini", "google", "google_gemini"):
        return "gemini"
    raise RouterConfigError(f"Unsupported API format {fmt!r}; expected one of: openai, anthropic, gemini.")


def _infer_api_format(base_url: str | None) -> str:
    if not base_url:
        return "openai"
    try:
        host = (urlsplit(base_url.strip()).hostname or "").lower()
    except ValueError:
        return "openai"
    if "anthropic" in host:
        return "anthropic"
    if "generativelanguage.googleapis.com" in host or "aiplatform.googleapis.com" in host:
        return "gemini"
    return "openai"


def _infer_default_model(api_format: str, base_url: str) -> str:
    fmt = _normalize_format(api_format)
    if fmt == "anthropic":
        return "claude-sonnet-4-5"
    if fmt == "gemini":
        return "gemini-2.5-flash"
    try:
        host = (urlsplit(base_url.strip()).hostname or "").lower()
    except ValueError:
        host = ""
    if host == "api.openai.com" or host.endswith(".openai.com"):
        return "gpt-4o-mini"
    if host == "api.deepseek.com" or host.endswith(".deepseek.com"):
        return "deepseek-chat"
    if host in ("open.bigmodel.cn", "api.z.ai") or host.endswith((".bigmodel.cn", ".z.ai")):
        return "glm-5.3-flash"
    if host.endswith(("minimax.chat", "minimaxi.com", "minimax.io")):
        return "minimax-m3.1-flash"
    return ""


def _validate_base_url(base_url: str, provider_name: str) -> str:
    raw = base_url.strip()
    if not raw:
        raise RouterConfigError(f"Empty Base URL for provider '{provider_name}'.")
    scheme_sep = raw.find("://")
    if scheme_sep > 0:
        authority = raw[scheme_sep + 3 :].split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        if "@" in authority or "＠" in authority:
            raise RouterConfigError(
                f"Base URL for provider '{provider_name}' must not embed credentials; use env vars or --api-key."
            )
    try:
        parts = urlsplit(raw)
    except ValueError:
        raise RouterConfigError(
            f"Invalid Base URL for provider '{provider_name}': malformed URL."
        ) from None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise RouterConfigError(f"Invalid Base URL for provider '{provider_name}': expected http(s) URL.")
    if parts.username is not None or parts.password is not None:
        raise RouterConfigError(
            f"Base URL for provider '{provider_name}' must not embed credentials; use env vars or --api-key."
        )
    if parts.query:
        q_keys = {k.lower() for k, _ in parse_qsl(parts.query, keep_blank_values=True)}
        if q_keys & _SENSITIVE_QUERY_KEYS:
            raise RouterConfigError(
                f"Base URL for provider '{provider_name}' must not include query API keys (?key= etc.); "
                "pass keys via environment variable or --api-key."
            )
        raise RouterConfigError(f"Base URL for provider '{provider_name}' must not include a query string.")
    clean_path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, clean_path, "", ""))


def _strip_think_tags(text: str) -> str:
    return _THINK_PREFIX_RE.sub("", text, count=1) if text.lstrip().startswith("<think>") else text


def _deep_merge(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    if not extra:
        return out
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_CANONICAL_FAMILY_PREFIXES = (
    ("deepseek", "deepseek"),
    ("glm", "glm"),
    ("minimax", "minimax"),
    ("hunyuan", "hunyuan"),
    ("hy", "hunyuan"),
    ("kimi", "kimi"),
    ("qwen", "qwen"),
    ("mimo", "mimo"),
    ("claude", "claude"),
    ("gpt", "gpt"),
    ("gemini", "gemini"),
    ("llama", "llama"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("mistral", "mistral"),
    ("grok", "grok"),
    ("doubao", "doubao"),
    ("yi", "yi"),
)


def resolve_model_family(model: str | None, config: dict[str, Any] | None = None) -> str:
    """Resolve canonical model family fail-closed (returns '' for unrecognized models)."""
    if not model:
        return ""
    lookup = model.strip()
    if config:
        models_cfg = config.get("models", {})
        if lookup in models_cfg:
            fam = models_cfg[lookup].get("family")
            if isinstance(fam, str) and fam.strip():
                return fam.strip().lower()
        if ":" in lookup:
            prefix, rest = lookup.split(":", 1)
            known_providers = set(config.get("providers", {}).keys()) | set(config.get("groups", {}).keys())
            if prefix in known_providers and rest:
                lookup = rest
                if lookup in models_cfg:
                    fam = models_cfg[lookup].get("family")
                    if isinstance(fam, str) and fam.strip():
                        return fam.strip().lower()
    m = lookup.lower().rsplit("/", 1)[-1]
    for prefix, fam_name in _CANONICAL_FAMILY_PREFIXES:
        if prefix == "hy":
            if re.match(r"^hy(?:\d+|[-.]|$)", m):
                return fam_name
        elif re.match(rf"^{re.escape(prefix)}(?:\d|[-.]|$)", m):
            return fam_name
    return ""


def _build_request(
    api_format: str,
    base_url: str,
    api_key: str,
    upstream_model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int | None,
    temperature: float | None,
    reasoning_effort: str | None = None,
    max_tokens_param: str = "max_tokens",
    extra_body: dict[str, Any] | None = None,
    strip_reasoning: bool = False,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build URL, headers, and JSON body for openai, anthropic, or gemini formats."""
    fmt = _normalize_format(api_format)
    clean_base = _validate_base_url(base_url, fmt)
    parts = urlsplit(clean_base)
    path = parts.path.rstrip("/")
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "novel-wash/0.1",
    }

    if fmt == "openai":
        target_path = path if path.endswith("/chat/completions") else f"{path}/chat/completions"
        url = urlunsplit((parts.scheme, parts.netloc, target_path, "", ""))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload: dict[str, Any] = {
            "model": upstream_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if max_tokens is not None:
            payload[max_tokens_param or "max_tokens"] = int(max_tokens)
        if temperature is not None:
            payload["temperature"] = temperature
        if reasoning_effort and not strip_reasoning:
            payload["reasoning_effort"] = reasoning_effort
        merged = _deep_merge(payload, extra_body)
        if strip_reasoning:
            merged.pop("reasoning_effort", None)
        return url, headers, merged

    if fmt == "anthropic":
        if path.endswith("/messages"):
            target_path = path
        else:
            root_path = path[:-3] if path.endswith("/v1") else path
            target_path = f"{root_path}/v1/messages"
        url = urlunsplit((parts.scheme, parts.netloc, target_path, "", ""))
        if api_key:
            headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": upstream_model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": int(max_tokens if max_tokens is not None else 4096),
        }
        if temperature is not None and not (extra_body and "thinking" in extra_body):
            payload["temperature"] = min(max(float(temperature), 0.0), 1.0)
        merged = _deep_merge(payload, extra_body)
        if strip_reasoning:
            merged.pop("reasoning_effort", None)
        return url, headers, merged

    # fmt == "gemini"
    model_id = upstream_model[7:] if upstream_model.startswith("models/") else upstream_model
    if path.endswith(":generateContent"):
        target_path = path
    elif re.search(r"/v1(?:beta|alpha)?$", path):
        target_path = f"{path}/models/{model_id}:generateContent"
    else:
        target_path = f"{path}/v1beta/models/{model_id}:generateContent"
    url = urlunsplit((parts.scheme, parts.netloc, target_path, "", ""))
    if api_key:
        headers["x-goog-api-key"] = api_key
    gen_cfg: dict[str, Any] = {}
    if max_tokens is not None:
        gen_cfg["maxOutputTokens"] = int(max_tokens)
    if temperature is not None:
        gen_cfg["temperature"] = temperature
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
    }
    if gen_cfg:
        payload["generationConfig"] = gen_cfg
    merged = _deep_merge(payload, extra_body)
    if strip_reasoning:
        merged.pop("reasoning_effort", None)
    return url, headers, merged


def _parse_response(api_format: str, data: Any, route_key: str) -> tuple[str, str, int, int, int, int]:
    """Parse HTTP 200 payload into `(content, finish_reason, prompt_t, cached_t, completion_t, reasoning_t)`."""
    if not isinstance(data, dict):
        raise ValueError("HTTP 200 payload is not a JSON object")
    fmt = _normalize_format(api_format)

    if fmt == "openai":
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("HTTP 200 payload missing valid choices list")
        choice = choices[0]
        msg = choice.get("message")
        if not isinstance(msg, dict):
            raise ValueError("HTTP 200 choice missing message dict")
        refusal = msg.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise RouterContentPolicyError(f"OpenAI refusal on {route_key}: {refusal[:240]}")
        raw_fr = str(choice.get("finish_reason") or "")
        if raw_fr == "content_filter":
            raise RouterContentPolicyError(f"OpenAI finish_reason=content_filter on {route_key}")
        raw_content = msg.get("content")
        if isinstance(raw_content, list):
            parts_txt = [
                str(p.get("text") or "")
                for p in raw_content
                if isinstance(p, dict) and p.get("type") in (None, "text", "output_text")
            ]
            content = _strip_think_tags("".join(parts_txt))
        else:
            content = _strip_think_tags(str(raw_content or ""))
        finish_reason = "length" if raw_fr == "length" else ("stop" if raw_fr == "stop" else raw_fr)
        usage = data.get("usage")
        if usage is None:
            usage = {}
        elif not isinstance(usage, dict):
            raise ValueError("HTTP 200 payload usage must be an object")
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        pt_det = usage.get("prompt_tokens_details") or {}
        ct_det = usage.get("completion_tokens_details") or {}
        if not isinstance(pt_det, dict) or not isinstance(ct_det, dict):
            raise ValueError("HTTP 200 payload token details must be an object")
        cached_t = int(pt_det.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0)
        rt = int(ct_det.get("reasoning_tokens") or 0)
        return content, finish_reason, pt, cached_t, ct, rt

    if fmt == "anthropic":
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ValueError("Anthropic HTTP 200 payload missing content list")
        texts = [
            str(b.get("text") or "")
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        content = "".join(texts)
        raw_fr = str(data.get("stop_reason") or "")
        if raw_fr == "refusal":
            raise RouterContentPolicyError(f"Anthropic stop_reason=refusal on {route_key}")
        if raw_fr in ("max_tokens", "model_context_window_exceeded"):
            finish_reason = "length"
        elif raw_fr in ("end_turn", "stop_sequence"):
            finish_reason = "stop"
        else:
            finish_reason = raw_fr
        usage = data.get("usage")
        if usage is None:
            usage = {}
        elif not isinstance(usage, dict):
            raise ValueError("Anthropic payload usage must be an object")
        raw_in = int(usage.get("input_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_create = int(usage.get("cache_creation_input_tokens") or 0)
        pt = raw_in + cache_read + cache_create
        ct = int(usage.get("output_tokens") or 0)
        return content, finish_reason, pt, cache_read, ct, 0

    # fmt == "gemini"
    prompt_fb = data.get("promptFeedback")
    if isinstance(prompt_fb, dict) and prompt_fb.get("blockReason"):
        raise RouterContentPolicyError(
            f"Gemini promptFeedback blockReason={prompt_fb.get('blockReason')} on {route_key}"
        )
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise ValueError(f"Gemini HTTP 200 payload missing valid candidates list on {route_key}")
    cand = candidates[0]
    raw_fr = str(cand.get("finishReason") or "")
    if raw_fr in _GEMINI_BLOCK_FINISH_REASONS:
        raise RouterContentPolicyError(f"Gemini finishReason={raw_fr} on {route_key}")
    cand_content = cand.get("content") or {}
    parts = cand_content.get("parts") if isinstance(cand_content, dict) else []
    texts = [
        str(p.get("text") or "")
        for p in (parts if isinstance(parts, list) else [])
        if isinstance(p, dict) and p.get("thought") is not True
    ]
    content = "".join(texts)
    if raw_fr == "MAX_TOKENS":
        finish_reason = "length"
    elif raw_fr == "STOP":
        finish_reason = "stop"
    else:
        finish_reason = raw_fr.lower()
    um = data.get("usageMetadata")
    if um is None:
        um = {}
    elif not isinstance(um, dict):
        raise ValueError("Gemini payload usageMetadata must be an object")
    pt = int(um.get("promptTokenCount") or 0)
    cached_t = int(um.get("cachedContentTokenCount") or 0)
    cand_out = int(um.get("candidatesTokenCount") or 0)
    rt = int(um.get("thoughtsTokenCount") or 0)
    ct = cand_out + rt
    return content, finish_reason, pt, cached_t, ct, rt


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (v.startswith('"') and v.endswith('"') and len(v) >= 2) or (
            v.startswith("'") and v.endswith("'") and len(v) >= 2
        ):
            v = v[1:-1]
        else:
            v = re.sub(r"\s+#.*$", "", v).strip()
        if k and k not in os.environ:
            os.environ[k] = v


class _GroupRateLimiter:
    """Thread-safe spacing rate limiter + concurrency gate for a provider."""

    def __init__(
        self,
        rpm: float,
        *,
        max_concurrency: int = 4,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_interval = 60.0 / max(rpm, 0.5)
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max(1, int(max_concurrency)))
        self._next_allowed = 0.0
        self._cooldown_until = 0.0
        self._penalty_until = 0.0
        self._monotonic = monotonic_fn
        self._sleep = sleep_fn

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self._monotonic()
                interval = self._base_interval * (2.0 if now < self._penalty_until else 1.0)
                wait_until = max(self._next_allowed, self._cooldown_until)
                if now >= wait_until:
                    self._next_allowed = now + interval
                    return
                delay = wait_until - now
            self._sleep(min(delay, 5.0))

    def backoff(self, seconds: float) -> None:
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, self._monotonic() + seconds)

    def halve_rpm(self, duration_s: float = 60.0) -> None:
        with self._lock:
            self._penalty_until = max(self._penalty_until, self._monotonic() + duration_s)

    def in_cooldown(self) -> bool:
        with self._lock:
            return self._monotonic() < self._cooldown_until

    def slot(self) -> threading.Semaphore:
        return self._sem


class _RouteBreaker:
    """Per-(provider, model) circuit breaker with exponential backoff and half-open probe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.consecutive_failures = 0
        self.open_until = 0.0
        self.cooldown_s = 60.0
        self.half_open_in_flight = False
        self._probe_token: int | None = None
        self._next_token: int = 1
        self._min_valid_token: int = 0
        self.permanently_disabled = False

    def can_try(self, now: float) -> bool:
        """Read-only availability check (does not consume a half-open probe slot)."""
        with self._lock:
            return not self.permanently_disabled and now >= self.open_until and not self.half_open_in_flight

    def acquire_dispatch(self, now: float) -> tuple[bool, int | None]:
        """Atomically reserve dispatch slot, returning `(True, unique_dispatch_token)`."""
        with self._lock:
            if self.permanently_disabled or now < self.open_until:
                return False, None
            if self.open_until > 0.0:
                if self.half_open_in_flight:
                    return False, None
                tok = self._next_token
                self._next_token += 1
                self._probe_token = tok
                self.half_open_in_flight = True
                return True, tok
            tok = self._next_token
            self._next_token += 1
            return True, tok

    def acquire_probe(self, now: float) -> bool:
        """Atomically reserve dispatch slot (consuming the single half-open probe when open_until > 0)."""
        ok, _ = self.acquire_dispatch(now)
        return ok

    def release_probe(self, token: int | None = None) -> None:
        """Release a reserved half-open probe only when the caller owns `_probe_token`."""
        with self._lock:
            if token is not None and self._probe_token == token:
                self._probe_token = None
                self.half_open_in_flight = False

    def allow(self, now: float) -> bool:
        return self.acquire_probe(now)

    def record_success(self, token: int | None = None, *, is_deferred: bool = False) -> None:
        with self._lock:
            if token is not None and token < self._min_valid_token:
                return
            if self._probe_token is not None and token != self._probe_token:
                return
            if is_deferred and self.open_until > 0.0 and token is None:
                return
            self.consecutive_failures = 0
            self.open_until = 0.0
            self.cooldown_s = 60.0
            self._probe_token = None
            self.half_open_in_flight = False

    def record_failure(
        self,
        now: float,
        *,
        permanent: bool = False,
        threshold: int = 2,
        token: int | None = None,
        is_deferred: bool = False,
    ) -> None:
        with self._lock:
            if token is not None and token < self._min_valid_token:
                return
            if self._probe_token is not None and token != self._probe_token:
                return
            was_half_open = (self._probe_token is not None) and (token == self._probe_token)
            if token == self._probe_token:
                self._probe_token = None
                self.half_open_in_flight = False
            if permanent:
                self.permanently_disabled = True
                self._min_valid_token = self._next_token
                return
            self.consecutive_failures += 1
            if was_half_open or self.consecutive_failures >= threshold:
                self.open_until = now + self.cooldown_s
                self.cooldown_s = min(self.cooldown_s * 2.0, 180.0)
                self._min_valid_token = self._next_token


@dataclass
class CallMetrics:
    model: str
    group: str
    family: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_cny: float = 0.0
    latency_s: float = 0.0
    finish_reason: str = ""
    probe_token: int | None = None


@dataclass
class CostLedger:
    calls: int = 0
    fallbacks: int = 0
    by_route: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, m: CallMetrics, *, is_fallback: bool = False) -> None:
        key = f"{m.group}/{m.model}"
        with self._lock:
            self.calls += 1
            if is_fallback:
                self.fallbacks += 1
            bucket = self.by_route.setdefault(
                key,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost_cny": 0.0,
                    "latency_s": 0.0,
                },
            )
            bucket["calls"] += 1
            bucket["prompt_tokens"] += m.prompt_tokens
            bucket["cached_tokens"] += m.cached_tokens
            bucket["completion_tokens"] += m.completion_tokens
            bucket["reasoning_tokens"] += m.reasoning_tokens
            bucket["cost_cny"] = round(bucket["cost_cny"] + m.cost_cny, 6)
            bucket["latency_s"] = round(bucket["latency_s"] + m.latency_s, 3)

    def total_cost_cny(self) -> float:
        with self._lock:
            return round(sum(v["cost_cny"] for v in self.by_route.values()), 6)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "fallbacks": self.fallbacks,
                "total_cost_cny": round(sum(v["cost_cny"] for v in self.by_route.values()), 6),
                "by_route": {k: dict(v) for k, v in self.by_route.items()},
            }


def _synthesize_zero_config(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    api_format: str | None = None,
) -> dict[str, Any]:
    """Build an in-memory single-provider config when `config/models.json` is absent."""
    env_base = (
        base_url
        or os.environ.get("NOVEL_WASH_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or os.environ.get("GEMINI_BASE_URL")
        or ""
    ).strip()
    raw_fmt = api_format or os.environ.get("NOVEL_WASH_API_FORMAT")
    fmt_hint = _normalize_format(raw_fmt) if raw_fmt else "openai"
    resolved_base = _validate_base_url(env_base or _FORMAT_DEFAULT_BASE_URLS[fmt_hint], "default")
    fmt = _normalize_format(raw_fmt) if raw_fmt else _infer_api_format(resolved_base)

    primary = os.environ.get("NOVEL_WASH_PRIMARY_MODEL", "").strip() or _infer_default_model(fmt, resolved_base) or None
    secondary = os.environ.get("NOVEL_WASH_SECONDARY_MODEL", "").strip() or None

    provider_entry: dict[str, Any] = {
        "format": fmt,
        "base_url": resolved_base,
        "api_key_envs": list(dict.fromkeys([*_FORMAT_KEY_ENVS[fmt], "NOVEL_WASH_API_KEY"])),
        "rpm_limit": 30.0,
        "pool_rpm_limit": 30.0,
        "max_concurrency": 4,
    }
    if api_key is not None:
        provider_entry["api_key"] = api_key

    return {
        "base_url": resolved_base,
        "providers": {"default": provider_entry},
        "models": {},
        "pools": {},
        "default_routing": {"primary": primary, "secondary": secondary},
    }


class LLMRouter:
    """Routes LLM calls across configured providers (OpenAI, Anthropic, Gemini) with failover."""

    def __init__(
        self,
        project_root: Path,
        *,
        config_path: Path | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_format: str | None = None,
        allow_welfare_fallback: bool = True,
        http_client: httpx.Client | None = None,
        monotonic_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        load_dotenv(project_root / ".env")
        chosen_cfg = config_path or (project_root / "config" / "models.json")
        if chosen_cfg.exists():
            raw_cfg = json.loads(chosen_cfg.read_text(encoding="utf-8"))
        elif config_path is not None:
            raise RouterConfigError(f"Model config file not found: {chosen_cfg}")
        else:
            raw_cfg = _synthesize_zero_config()

        self.config: dict[str, Any] = copy.deepcopy(raw_cfg)
        self._explicit_base_url = _validate_base_url(base_url, "_cli") if base_url else None
        self._explicit_api_key = api_key if api_key is not None else None
        self._explicit_api_format = _normalize_format(api_format) if api_format else None
        self._has_cli_override = bool(
            self._explicit_base_url is not None
            or self._explicit_api_key is not None
            or self._explicit_api_format is not None
        )
        self.allow_welfare_fallback = allow_welfare_fallback
        self.ledger = CostLedger()
        self._monotonic = monotonic_fn or time.monotonic
        self._sleep = sleep_fn or time.sleep
        self._limiters: dict[str, _GroupRateLimiter] = {}
        self._pool_limiters: dict[str, _GroupRateLimiter] = {}
        self._keys: dict[str, str] = {}
        self._providers: dict[str, dict[str, Any]] = {}
        self._key_hints: dict[str, list[str]] = {}

        is_legacy_schema = "groups" in self.config and "providers" not in self.config
        raw_providers = self.config.get("providers") or self.config.get("groups") or {}

        top_env_base = (
            os.environ.get("NOVEL_WASH_BASE_URL")
            or (os.environ.get("XCM_BASE_URL") if is_legacy_schema else None)
            or os.environ.get("OPENAI_BASE_URL")
        )
        top_cfg_base = self.config.get("base_url")
        top_env_fmt = os.environ.get("NOVEL_WASH_API_FORMAT")

        for pname, pcfg in raw_providers.items():
            p_copy = dict(pcfg)
            p_base_env_name = str(p_copy.get("base_url_env") or "").strip()
            p_env_base = os.environ.get(p_base_env_name, "").strip() if p_base_env_name else ""
            raw_cfg_fmt = p_copy.get("format")

            if is_legacy_schema:
                raw_base = p_env_base or p_copy.get("base_url") or top_env_base or top_cfg_base or _FORMAT_DEFAULT_BASE_URLS["openai"]
                fmt = _normalize_format(raw_cfg_fmt) if raw_cfg_fmt else "openai"
            elif pname == "default":
                raw_base = (
                    p_env_base
                    or os.environ.get("NOVEL_WASH_BASE_URL")
                    or os.environ.get("OPENAI_BASE_URL")
                    or p_copy.get("base_url")
                    or top_cfg_base
                    or _FORMAT_DEFAULT_BASE_URLS["openai"]
                )
                if top_env_fmt:
                    fmt = _normalize_format(top_env_fmt)
                elif p_env_base or os.environ.get("NOVEL_WASH_BASE_URL"):
                    inferred = _infer_api_format(str(raw_base))
                    fmt = inferred if inferred != "openai" else (_normalize_format(raw_cfg_fmt) if raw_cfg_fmt else "openai")
                else:
                    fmt = _normalize_format(raw_cfg_fmt) if raw_cfg_fmt else _infer_api_format(str(raw_base))
            else:
                fmt = _normalize_format(raw_cfg_fmt) if raw_cfg_fmt else _infer_api_format(str(p_env_base or p_copy.get("base_url") or ""))
                raw_base = p_env_base or p_copy.get("base_url") or _FORMAT_DEFAULT_BASE_URLS[fmt]

            p_base = _validate_base_url(str(raw_base), pname)

            mc = int(p_copy.get("max_concurrency", 4))
            base_rpm = float(p_copy.get("rpm_limit", 30.0 if not is_legacy_schema else 5.0))
            pool_rpm = float(p_copy.get("pool_rpm_limit", base_rpm))
            self._limiters[pname] = _GroupRateLimiter(
                base_rpm, max_concurrency=mc, monotonic_fn=self._monotonic, sleep_fn=self._sleep
            )
            self._pool_limiters[pname] = _GroupRateLimiter(
                pool_rpm, max_concurrency=mc, monotonic_fn=self._monotonic, sleep_fn=self._sleep
            )

            env_names: list[str] = []
            for key_field in ("api_key_envs", "fallback_api_key_envs"):
                val_list = p_copy.get(key_field)
                if isinstance(val_list, list):
                    env_names.extend(str(x).strip() for x in val_list if str(x).strip())
            for single_field in ("api_key_env", "env_key"):
                s_val = str(p_copy.get(single_field) or "").strip()
                if s_val and s_val not in env_names:
                    env_names.insert(0, s_val)
            if not env_names:
                env_names = list(_FORMAT_KEY_ENVS[fmt])

            self._key_hints[pname] = env_names
            resolved_key = ""
            for ename in env_names:
                ev = os.environ.get(ename, "").strip()
                if ev:
                    resolved_key = ev
                    break
            if not resolved_key and p_copy.get("api_key"):
                resolved_key = str(p_copy["api_key"]).strip()
            if resolved_key:
                self._keys[pname] = resolved_key

            fb_prov = p_copy.get("fallback_provider") or p_copy.get("fallback_group")
            self._providers[pname] = {
                "name": pname,
                "format": fmt,
                "base_url": p_base,
                "models": p_copy.get("models") if isinstance(p_copy.get("models"), list) else None,
                "fallback_provider": str(fb_prov) if fb_prov else None,
                "price_multiplier": float(p_copy.get("price_multiplier", p_copy.get("discount_multiplier", 1.0))),
            }

        # Build isolated virtual `_cli` route when explicit CLI credentials/endpoint/format were given
        if self._has_cli_override:
            if self._explicit_api_format:
                cli_fmt = self._explicit_api_format
            elif self._explicit_base_url:
                cli_fmt = _infer_api_format(self._explicit_base_url)
            elif top_env_fmt:
                cli_fmt = _normalize_format(top_env_fmt)
            else:
                cli_fmt = _infer_api_format(os.environ.get("NOVEL_WASH_BASE_URL"))

            cli_base_raw = (
                self._explicit_base_url
                or (self._providers["default"]["base_url"] if ("default" in self._providers and not self._explicit_api_format) else None)
                or (os.environ.get("NOVEL_WASH_BASE_URL") if not self._explicit_api_format else None)
                or _FORMAT_DEFAULT_BASE_URLS[cli_fmt]
            )
            cli_base = _validate_base_url(str(cli_base_raw), "_cli")
            cli_key_envs = list(dict.fromkeys([*_FORMAT_KEY_ENVS[cli_fmt], "NOVEL_WASH_API_KEY"]))
            cli_key = ""
            if self._explicit_api_key is not None:
                cli_key = self._explicit_api_key.strip()
            else:
                for ename in cli_key_envs:
                    ev = os.environ.get(ename, "").strip()
                    if ev:
                        cli_key = ev
                        break
            self._limiters["_cli"] = _GroupRateLimiter(30.0, max_concurrency=4, monotonic_fn=self._monotonic, sleep_fn=self._sleep)
            self._pool_limiters["_cli"] = _GroupRateLimiter(30.0, max_concurrency=4, monotonic_fn=self._monotonic, sleep_fn=self._sleep)
            self._key_hints["_cli"] = cli_key_envs
            if cli_key:
                self._keys["_cli"] = cli_key
            self._providers["_cli"] = {
                "name": "_cli",
                "format": cli_fmt,
                "base_url": cli_base,
                "models": None,
                "fallback_provider": None,
                "price_multiplier": 1.0,
            }

        # Keep config["providers"] & config["groups"] populated for downstream helpers/tests
        self.config.setdefault("providers", raw_providers)
        if "groups" not in self.config:
            self.config["groups"] = raw_providers
        self.config.setdefault("models", {})
        self.config.setdefault("pools", {})
        self.config.setdefault("default_routing", {})

        active_ref = self._providers.get("_cli") or self._providers.get("default") or (next(iter(self._providers.values())) if self._providers else None)
        self.base_url = str(active_ref["base_url"]) if active_ref else _FORMAT_DEFAULT_BASE_URLS["openai"]

        self._breakers: dict[str, _RouteBreaker] = {}
        self._breaker_lock = threading.Lock()
        self._unsupported_reasoning_effort: set[str] = set()
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=30.0),
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=30.0),
        )

    def _active_default_provider(self, explicit_provider: str | None = None) -> dict[str, Any] | None:
        if explicit_provider and explicit_provider in self._providers:
            return self._providers[explicit_provider]
        if self._has_cli_override and "_cli" in self._providers:
            return self._providers["_cli"]
        return self._providers.get("default") or (next(iter(self._providers.values())) if self._providers else None)

    def resolve_default_primary(self, provider: str | None = None) -> str:
        env_p = os.environ.get("NOVEL_WASH_PRIMARY_MODEL", "").strip()
        if env_p:
            return env_p
        use_cfg_defaults = (provider is None) and (not self._has_cli_override)
        if use_cfg_defaults:
            cfg_p = str(self.config.get("default_routing", {}).get("primary") or "").strip()
            if cfg_p:
                return cfg_p
            models_cfg = self.config.get("models") or {}
            for mname, mcfg in models_cfg.items():
                if isinstance(mcfg, dict) and mcfg.get("enabled", True):
                    return mname
        p = self._active_default_provider(provider)
        if p is not None:
            inferred = _infer_default_model(p["format"], p["base_url"])
            if inferred:
                return inferred
        raise RouterConfigError(
            "No primary model configured for this endpoint. Set NOVEL_WASH_PRIMARY_MODEL in .env or pass --primary <model>."
        )

    @property
    def default_primary(self) -> str:
        return self.resolve_default_primary()

    @property
    def default_secondary(self) -> str | None:
        raw_env = os.environ.get("NOVEL_WASH_SECONDARY_MODEL")
        if raw_env is not None:
            s = raw_env.strip()
            return None if s.lower() in ("", "none", "off", "null") else s
        if self._has_cli_override:
            return None
        cfg_s = self.config.get("default_routing", {}).get("secondary")
        if isinstance(cfg_s, str):
            s = cfg_s.strip()
            return None if s.lower() in ("", "none", "off", "null") else s
        return None

    def _default_provider_name(self) -> str:
        if self._has_cli_override and "_cli" in self._providers:
            return "_cli"
        if "default" in self._providers:
            return "default"
        for pname in self._providers:
            if pname != "_cli" and pname in self._keys:
                return pname
        return next(iter(self._providers)) if self._providers else "default"

    def _split_provider_model(self, model: str) -> tuple[str | None, str]:
        """Split `provider:model` ONLY when the prefix matches a configured provider name."""
        if ":" in model:
            prefix, rest = model.split(":", 1)
            if prefix in self._providers and prefix != "_cli" and rest:
                return prefix, rest
        return None, model

    def _resolve_model_entry(self, model: str, provider_override: str | None = None) -> tuple[str | None, bool, str, dict[str, Any]]:
        prefix_prov, lookup_name = self._split_provider_model(model)
        explicit_prov = provider_override or prefix_prov
        using_cli = False
        if explicit_prov is None and self._has_cli_override and "_cli" in self._providers:
            explicit_prov = "_cli"
            using_cli = True

        mcfg = self.config.get("models", {}).get(lookup_name)
        if mcfg is None and model in self.config.get("models", {}):
            lookup_name = model
            mcfg = self.config["models"][model]

        if mcfg is not None:
            if not mcfg.get("enabled", True):
                raise RouterConfigError(f"Model {lookup_name} is disabled in config/models.json")
            entry = dict(mcfg)
            base_m = lookup_name.lower().rsplit("/", 1)[-1]
            if re.match(r"^(?:o1|o3|o4|gpt-5)(?:\d|[-.]|$)", base_m):
                entry.setdefault("max_tokens_param", "max_completion_tokens")
            return explicit_prov, using_cli, lookup_name, entry

        # Unlisted model: synthesize sane defaults routed to the explicit or default provider
        target_prov = explicit_prov or self._default_provider_name()
        fam = resolve_model_family(lookup_name, self.config)
        base_m = lookup_name.lower().rsplit("/", 1)[-1]
        is_openai_reasoning = bool(re.match(r"^(?:o1|o3|o4|gpt-5)(?:\d|[-.]|$)", base_m))
        synth_cfg: dict[str, Any] = {
            "enabled": True,
            "family": fam,
            "provider": target_prov,
            "max_tokens": 4096,
            "temperature": None if is_openai_reasoning else 0.0,
            "max_tokens_param": "max_completion_tokens" if is_openai_reasoning else "max_tokens",
        }
        return explicit_prov, using_cli, lookup_name, synth_cfg

    def _provider_allows_model(self, pname: str, lookup_name: str, is_explicit_registered: bool) -> bool:
        if pname not in self._providers or pname not in self._keys:
            return False
        allow = self._providers[pname].get("models")
        if isinstance(allow, list) and is_explicit_registered:
            return lookup_name in allow
        return True

    def supports_model(self, model: str, provider: str | None = None) -> bool:
        try:
            routes = self._candidate_routes(model, provider, raise_on_missing=False)
            return bool(routes)
        except RouterConfigError:
            return False

    def describe_route(self, model: str, provider: str | None = None) -> dict[str, Any]:
        try:
            eff_override, using_cli, lookup_name, mcfg = self._resolve_model_entry(model, provider)
        except RouterConfigError as exc:
            return {
                "provider": provider or self._default_provider_name(),
                "model": model,
                "upstream_model": model,
                "format": "openai",
                "url": "",
                "family": resolve_model_family(model, self.config),
                "has_key": False,
                "eligible": False,
                "expected_key_envs": [],
                "error": str(exc),
            }
        routes, diag_err = self._candidate_routes_with_diag(model, eff_override, mcfg=mcfg, using_cli=using_cli)
        chosen = routes[0] if routes else str(eff_override or mcfg.get("provider") or mcfg.get("preferred_group") or self._default_provider_name())
        pcfg = self._providers.get(chosen, {"format": "openai", "base_url": self.base_url})
        upstream = str(mcfg.get("model") or lookup_name)
        mtok = int(mcfg.get("max_tokens") or mcfg.get("max_output_tokens") or 4096)
        url, _, _ = _build_request(
            pcfg["format"],
            pcfg["base_url"],
            "",
            upstream,
            "",
            "",
            max_tokens=mtok,
            temperature=mcfg.get("temperature"),
            reasoning_effort=mcfg.get("reasoning_effort"),
            max_tokens_param=str(mcfg.get("max_tokens_param") or "max_tokens"),
            extra_body=mcfg.get("extra_body"),
        )
        return {
            "provider": chosen,
            "model": lookup_name,
            "upstream_model": upstream,
            "format": pcfg["format"],
            "url": url,
            "family": resolve_model_family(lookup_name, self.config),
            "has_key": chosen in self._keys,
            "eligible": bool(routes),
            "expected_key_envs": self._key_hints.get(chosen, []),
            "error": None if routes else diag_err,
        }

    def _breaker(self, route_key: str) -> _RouteBreaker:
        with self._breaker_lock:
            br = self._breakers.get(route_key)
            if br is None:
                br = _RouteBreaker()
                self._breakers[route_key] = br
            return br

    def report_bad_output(self, group: str, model: str, *, probe_token: int | None = None) -> None:
        self._breaker(f"{group}/{model}").record_failure(self._monotonic(), token=probe_token, is_deferred=True)

    def report_good_output(self, group: str, model: str, *, probe_token: int | None = None) -> None:
        self._breaker(f"{group}/{model}").record_success(token=probe_token, is_deferred=True)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _compute_cost(self, model: str, group: str, prompt_t: int, cached_t: int, comp_t: int) -> float:
        mcfg = self.config["models"].get(model, {})
        prices = mcfg.get("pricing_cny_per_1m", {}).get(group)
        if not prices:
            return 0.0
        uncached_in = max(0, prompt_t - cached_t)
        cost = (
            (uncached_in * float(prices["input"]))
            + (cached_t * float(prices.get("cached_input", prices["input"])))
            + (comp_t * float(prices["output"]))
        ) / 1_000_000.0
        return round(cost, 8)

    def _candidate_routes_with_diag(
        self,
        model: str,
        group: str | None,
        *,
        mcfg: dict[str, Any] | None = None,
        using_cli: bool = False,
    ) -> tuple[list[str], str]:
        if mcfg is None:
            eff_override, using_cli, lookup_name, mcfg = self._resolve_model_entry(model, group)
        else:
            _, lookup_name = self._split_provider_model(model)
            eff_override = group
        is_reg = lookup_name in self.config.get("models", {})
        routes: list[str] = []
        primary_g = str(eff_override or mcfg.get("provider") or mcfg.get("preferred_group") or self._default_provider_name())
        if self._provider_allows_model(primary_g, lookup_name, is_reg):
            routes.append(primary_g)
        if not using_cli and eff_override is None:
            fb = (
                mcfg.get("fallback_provider")
                or mcfg.get("fallback_group")
                or self._providers.get(primary_g, {}).get("fallback_provider")
            )
            if (
                self.allow_welfare_fallback
                and fb
                and str(fb) not in routes
                and self._provider_allows_model(str(fb), lookup_name, is_reg)
            ):
                routes.append(str(fb))

        if routes:
            return routes, ""
        hints = self._key_hints.get(primary_g) or ["NOVEL_WASH_API_KEY"]
        if primary_g not in self._providers:
            return [], f"Unknown provider '{primary_g}' for model {lookup_name}."
        if primary_g not in self._keys:
            return [], (
                f"Missing API key for provider '{primary_g}' (model {lookup_name}). "
                f"Set {' or '.join(hints)} in .env or environment (or pass --api-key)."
            )
        allow = self._providers[primary_g].get("models")
        if isinstance(allow, list) and is_reg and lookup_name not in allow:
            return [], f"Model '{lookup_name}' is not in provider '{primary_g}' allowlist."
        return [], f"No configured provider route supports model {lookup_name}."

    def _candidate_routes(
        self,
        model: str,
        group: str | None,
        *,
        raise_on_missing: bool = True,
    ) -> list[str]:
        routes, diag = self._candidate_routes_with_diag(model, group)
        if not routes and raise_on_missing:
            raise RouterConfigError(diag)
        return routes

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        group: str | None = None,
        reasoning_effort: str | None = "USE_CONFIG",
        max_tokens: int | None = None,
        temperature: float | None = "USE_CONFIG",  # type: ignore[assignment]
        max_retries: int = 4,
    ) -> tuple[str, CallMetrics]:
        eff_override, using_cli, lookup_name, mcfg = self._resolve_model_entry(model, group)
        eff = mcfg.get("reasoning_effort") if reasoning_effort == "USE_CONFIG" else reasoning_effort
        mtok = max_tokens if max_tokens is not None else int(mcfg.get("max_tokens") or mcfg.get("max_output_tokens") or 4096)
        temp: float | None = mcfg.get("temperature") if temperature == "USE_CONFIG" else temperature  # type: ignore[assignment]
        upstream_model = str(mcfg.get("model") or lookup_name)
        max_tok_param = str(mcfg.get("max_tokens_param") or "max_tokens")
        extra_body = mcfg.get("extra_body") if isinstance(mcfg.get("extra_body"), dict) else None

        routes, diag = self._candidate_routes_with_diag(model, eff_override, mcfg=mcfg, using_cli=using_cli)
        if not routes:
            raise RouterConfigError(diag)
        last_err: Exception | None = None
        retried_reasoning = False

        for attempt in range(max_retries):
            chosen_group = routes[0]
            for g in routes:
                if not self._limiters[g].in_cooldown():
                    chosen_group = g
                    break
            else:
                chosen_group = routes[-1]

            pcfg = self._providers[chosen_group]
            is_fallback = chosen_group != routes[0]
            self._limiters[chosen_group].acquire()

            strip_reasoning = (lookup_name in self._unsupported_reasoning_effort) or retried_reasoning
            active_eff = None if (not eff or strip_reasoning) else eff
            url, headers, payload = _build_request(
                pcfg["format"],
                pcfg["base_url"],
                self._keys.get(chosen_group, ""),
                upstream_model,
                system_prompt,
                user_prompt,
                max_tokens=mtok,
                temperature=temp,
                reasoning_effort=active_eff,
                max_tokens_param=max_tok_param,
                extra_body=extra_body,
                strip_reasoning=strip_reasoning,
            )

            t0 = self._monotonic()
            try:
                resp = self._client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_err = exc
                self._sleep(min(2.0 * (attempt + 1), 10.0))
                continue

            dt = max(0.0, self._monotonic() - t0)
            if (
                not retried_reasoning
                and resp.status_code == 400
                and "reasoning" in resp.text.lower()
                and "reasoning_effort" in payload
            ):
                self._unsupported_reasoning_effort.add(lookup_name)
                retried_reasoning = True
                continue

            if _is_auth_error_response(resp.status_code, resp.text):
                raise RouterConfigError(f"HTTP {resp.status_code} on {chosen_group}/{lookup_name}: {resp.text[:400]}")

            if resp.status_code in (400, 403) and _is_content_policy_response(resp.text):
                raise RouterContentPolicyError(
                    f"HTTP {resp.status_code} content policy on {chosen_group}/{lookup_name}: {resp.text[:300]}"
                )

            if resp.status_code == 400 and _is_context_length_response(resp.text):
                raise RuntimeError(f"Context length exceeded on {chosen_group}/{lookup_name}: {resp.text[:300]}")

            if resp.status_code in (403, 404):
                raise RouterConfigError(f"HTTP {resp.status_code} on {chosen_group}/{lookup_name}: {resp.text[:400]}")

            if _is_rate_limit_response(resp.status_code, resp.text) or resp.status_code in (500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {resp.status_code} on {chosen_group}/{lookup_name}: {resp.text[:240]}")
                retry_after = 0.0
                if "retry-after" in resp.headers:
                    try:
                        retry_after = min(max(float(resp.headers["retry-after"]), 0.5), 60.0)
                    except ValueError:
                        retry_after = 0.0
                self._limiters[chosen_group].backoff(
                    retry_after or (16.0 if _is_rate_limit_response(resp.status_code, resp.text) else 10.0)
                )
                if len(routes) > 1 and chosen_group == routes[0]:
                    continue
                self._sleep(retry_after or min(4.0 * (attempt + 1), 20.0))
                continue

            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} on {chosen_group}/{lookup_name}: {resp.text[:400]}")

            try:
                content, finish_reason, pt, cached_t, ct, rt = _parse_response(
                    pcfg["format"], resp.json(), f"{chosen_group}/{lookup_name}"
                )
            except RouterContentPolicyError:
                raise
            except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
                last_err = RuntimeError(f"Malformed HTTP 200 JSON envelope on {chosen_group}/{lookup_name}: {exc}")
                if attempt + 1 < max_retries:
                    self._sleep(min(2.0 * (attempt + 1), 10.0))
                continue
            cost = self._compute_cost(lookup_name, chosen_group, pt, cached_t, ct)
            metrics = CallMetrics(
                model=lookup_name,
                group=chosen_group,
                family=resolve_model_family(lookup_name, self.config),
                prompt_tokens=pt,
                cached_tokens=cached_t,
                completion_tokens=ct,
                reasoning_tokens=rt,
                cost_cny=cost,
                latency_s=round(dt, 3),
                finish_reason=finish_reason,
            )
            self.ledger.record(metrics, is_fallback=is_fallback)
            return content, metrics

        raise RuntimeError(f"All {max_retries} attempts failed for {lookup_name}: {last_err}")

    def _eligible_pool_routes(
        self,
        pool_name: str,
        *,
        exclude_families: frozenset[str] = frozenset(),
        exclude_routes: frozenset[str] = frozenset(),
        respect_breaker: bool = True,
    ) -> list[tuple[str, str]]:
        raw_routes = self.config.get("pools", {}).get(pool_name, [])
        now = self._monotonic()
        out: list[tuple[str, str]] = []
        for r in raw_routes:
            if "/" not in r:
                continue
            g, m = r.split("/", 1)
            mcfg = self.config.get("models", {}).get(m)
            if not mcfg or not mcfg.get("enabled", True):
                continue
            if not self._provider_allows_model(g, m, True):
                continue
            if resolve_model_family(m, self.config) in exclude_families:
                continue
            if r in exclude_routes:
                continue
            br = self._breaker(r)
            if br.permanently_disabled:
                continue
            if respect_breaker and not br.can_try(now):
                continue
            out.append((g, m))
        return out

    def pool_has_eligible_route(self, pool_name: str, *, exclude_families: frozenset[str] = frozenset()) -> bool:
        return bool(self._eligible_pool_routes(pool_name, exclude_families=exclude_families, respect_breaker=False))

    def pool_concurrency(self, pool_name: str, *, exclude_families: frozenset[str] = frozenset()) -> int:
        routes = self._eligible_pool_routes(pool_name, exclude_families=exclude_families, respect_breaker=True)
        if not routes:
            routes = self._eligible_pool_routes(pool_name, exclude_families=exclude_families, respect_breaker=False)
        if not routes:
            return 1
        g = routes[0][0]
        prov_map = self.config.get("providers") or self.config.get("groups") or {}
        return max(1, int(prov_map.get(g, {}).get("max_concurrency", 4)))

    def chat_pool(
        self,
        pool_name: str,
        system_prompt: str,
        user_prompt: str,
        *,
        exclude_families: frozenset[str] = frozenset(),
        exclude_routes: frozenset[str] = frozenset(),
        max_tokens: int | None = None,
        temperature: float | None = None,
        allow_low_censorship_fallback: bool = True,
        defer_success: bool = False,
        _attempted: set[str] | None = None,
    ) -> tuple[str, CallMetrics]:
        """Route a chat call across an ordered `(provider/model)` pool with circuit breaking & policy failover."""
        attempted_routes: set[str] = _attempted if _attempted is not None else set()
        effective_excl = exclude_routes | attempted_routes
        eligible = self._eligible_pool_routes(
            pool_name, exclude_families=exclude_families, exclude_routes=effective_excl
        )
        if not eligible:
            raise RuntimeError(f"No open healthy routes left in pool '{pool_name}'")

        policy_err: RouterContentPolicyError | None = None
        last_err: Exception | None = None

        for idx, (group, model) in enumerate(eligible):
            route_key = f"{group}/{model}"
            if route_key in attempted_routes:
                continue
            pcfg = self._providers[group]
            mcfg = self.config["models"][model]
            eff = mcfg.get("reasoning_effort")
            mtok = max_tokens if max_tokens is not None else int(mcfg.get("max_tokens") or mcfg.get("max_output_tokens") or 4096)
            temp = temperature if temperature is not None else mcfg.get("temperature")
            upstream_model = str(mcfg.get("model") or model)
            max_tok_param = str(mcfg.get("max_tokens_param") or "max_tokens")
            extra_body = mcfg.get("extra_body") if isinstance(mcfg.get("extra_body"), dict) else None
            limiter = self._pool_limiters[group]
            br = self._breaker(route_key)
            ok_disp, probe_tok = br.acquire_dispatch(self._monotonic())
            if not ok_disp:
                continue
            attempted_routes.add(route_key)
            settled = False

            try:
                sub_attempts = 2 if idx == len(eligible) - 1 else 1
                attempt_i = 0
                retried_io = False
                retried_reasoning = False
                strip_reasoning = model in self._unsupported_reasoning_effort
                while attempt_i < sub_attempts:
                    attempt_i += 1
                    limiter.acquire()
                    active_eff = None if (not eff or strip_reasoning) else eff
                    url, headers, payload = _build_request(
                        pcfg["format"],
                        pcfg["base_url"],
                        self._keys.get(group, ""),
                        upstream_model,
                        system_prompt,
                        user_prompt,
                        max_tokens=mtok,
                        temperature=temp,
                        reasoning_effort=active_eff,
                        max_tokens_param=max_tok_param,
                        extra_body=extra_body,
                        strip_reasoning=strip_reasoning,
                    )

                    t0 = self._monotonic()
                    try:
                        with limiter.slot():
                            resp = self._client.post(url, headers=headers, json=payload)
                    except httpx.HTTPError as exc:
                        last_err = exc
                        if not retried_io:
                            retried_io = True
                            sub_attempts += 1
                            self._sleep(0.5)
                            continue
                        br.record_failure(self._monotonic(), threshold=4, token=probe_tok)
                        settled = True
                        break

                    dt = max(0.0, self._monotonic() - t0)
                    if (
                        not retried_reasoning
                        and resp.status_code == 400
                        and "reasoning" in resp.text.lower()
                        and "reasoning_effort" in payload
                    ):
                        self._unsupported_reasoning_effort.add(model)
                        strip_reasoning = True
                        retried_reasoning = True
                        sub_attempts += 1
                        continue

                    if _is_auth_error_response(resp.status_code, resp.text):
                        last_err = RuntimeError(f"HTTP {resp.status_code} auth error on {route_key}: {resp.text[:240]}")
                        br.record_failure(self._monotonic(), permanent=True, token=probe_tok)
                        settled = True
                        break

                    if resp.status_code in (400, 403) and _is_content_policy_response(resp.text):
                        br.release_probe(probe_tok)
                        settled = True
                        policy_err = RouterContentPolicyError(
                            f"HTTP {resp.status_code} content policy on {route_key}: {resp.text[:240]}"
                        )
                        last_err = policy_err
                        if allow_low_censorship_fallback and pool_name != "low_censorship":
                            combined_excl = exclude_routes | attempted_routes
                            if self._eligible_pool_routes(
                                "low_censorship", exclude_families=exclude_families, exclude_routes=combined_excl
                            ):
                                try:
                                    return self.chat_pool(
                                        "low_censorship",
                                        system_prompt,
                                        user_prompt,
                                        exclude_families=exclude_families,
                                        exclude_routes=exclude_routes,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        allow_low_censorship_fallback=False,
                                        defer_success=defer_success,
                                        _attempted=attempted_routes,
                                    )
                                except RouterContentPolicyError as pe:
                                    policy_err = pe
                                    last_err = pe
                                except RuntimeError:
                                    pass
                        break

                    if resp.status_code in (403, 404):
                        last_err = RuntimeError(f"HTTP {resp.status_code} on {route_key}: {resp.text[:240]}")
                        br.record_failure(self._monotonic(), permanent=True, token=probe_tok)
                        settled = True
                        break

                    if _is_rate_limit_response(resp.status_code, resp.text) or resp.status_code in (500, 502, 503, 504) or resp.status_code >= 400:
                        last_err = RuntimeError(f"HTTP {resp.status_code} on {route_key}: {resp.text[:240]}")
                        if _is_rate_limit_response(resp.status_code, resp.text):
                            limiter.halve_rpm(60.0)
                            if idx == len(eligible) - 1 and attempt_i < sub_attempts:
                                self._sleep(2.0)
                                continue
                        fail_thresh = (
                            4
                            if (_is_rate_limit_response(resp.status_code, resp.text) or resp.status_code in (500, 502, 503, 504))
                            else 2
                        )
                        br.record_failure(self._monotonic(), threshold=fail_thresh, token=probe_tok)
                        settled = True
                        break

                    try:
                        content, finish_reason, pt, cached_t, ct, rt = _parse_response(
                            pcfg["format"], resp.json(), route_key
                        )
                    except RouterContentPolicyError as pe:
                        br.release_probe(probe_tok)
                        settled = True
                        policy_err = pe
                        last_err = pe
                        if allow_low_censorship_fallback and pool_name != "low_censorship":
                            combined_excl = exclude_routes | attempted_routes
                            if self._eligible_pool_routes(
                                "low_censorship", exclude_families=exclude_families, exclude_routes=combined_excl
                            ):
                                try:
                                    return self.chat_pool(
                                        "low_censorship",
                                        system_prompt,
                                        user_prompt,
                                        exclude_families=exclude_families,
                                        exclude_routes=exclude_routes,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        allow_low_censorship_fallback=False,
                                        defer_success=defer_success,
                                        _attempted=attempted_routes,
                                    )
                                except RouterContentPolicyError as pe2:
                                    policy_err = pe2
                                    last_err = pe2
                                except RuntimeError:
                                    pass
                        break
                    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
                        last_err = RuntimeError(f"Malformed HTTP 200 JSON envelope on {route_key}: {exc}")
                        br.record_failure(self._monotonic(), threshold=2, token=probe_tok)
                        settled = True
                        break

                    cost = self._compute_cost(model, group, pt, cached_t, ct)
                    if not defer_success:
                        br.record_success(token=probe_tok, is_deferred=False)
                    settled = True
                    metrics = CallMetrics(
                        model=model,
                        group=group,
                        family=resolve_model_family(model, self.config),
                        prompt_tokens=pt,
                        cached_tokens=cached_t,
                        completion_tokens=ct,
                        reasoning_tokens=rt,
                        cost_cny=cost,
                        latency_s=round(dt, 3),
                        finish_reason=finish_reason,
                        probe_token=probe_tok if defer_success else None,
                    )
                    self.ledger.record(metrics, is_fallback=(idx > 0 or not allow_low_censorship_fallback))
                    return content, metrics
            finally:
                if not settled and probe_tok is not None:
                    br.release_probe(probe_tok)

        if policy_err is not None:
            raise policy_err
        raise RuntimeError(f"All {len(eligible)} routes failed in pool '{pool_name}': {last_err}")
