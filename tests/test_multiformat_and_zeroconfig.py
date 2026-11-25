"""Request-level tests for multi-format adapters, zero-config .env, CLI overrides, security, and doctor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from novel_wash import cli as cli_mod
from novel_wash.llm_client import LLMRouter, RouterConfigError, RouterContentPolicyError


def test_zero_config_env_only_inline_comments_and_endpoint_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "XCM_BASE_URL",
        "NOVEL_WASH_BASE_URL",
        "OPENAI_BASE_URL",
        "NOVEL_WASH_API_KEY",
        "OPENAI_API_KEY",
        "NOVEL_WASH_API_FORMAT",
        "NOVEL_WASH_PRIMARY_MODEL",
        "NOVEL_WASH_SECONDARY_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)

    # Unquoted inline comments in copied .env snippets are stripped cleanly
    (tmp_path / ".env").write_text(
        "NOVEL_WASH_BASE_URL=https://api.openai.com/v1 # inline comment\n"
        "NOVEL_WASH_API_KEY=sk-user-openai\n",
        encoding="utf-8",
    )

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode("utf-8"))
        seen.append({"url": str(req.url), "auth": req.headers.get("Authorization"), "body": body})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '<think>internal</think>{"L1":{"del":[1],"keep":[]}}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 12, "prompt_cache_hit_tokens": 10},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = LLMRouter(tmp_path, http_client=client)
    try:
        # Two-variable OpenAI quickstart selects an actual OpenAI model (gpt-4o-mini), not DeepSeek!
        assert router.default_primary == "gpt-4o-mini"
        assert router.default_secondary is None

        txt, metrics = router.chat(router.default_primary, "sys", "usr")
        assert txt == '{"L1":{"del":[1],"keep":[]}}'
        assert metrics.cached_tokens == 10
        assert seen[0]["url"] == "https://api.openai.com/v1/chat/completions"
        assert seen[0]["auth"] == "Bearer sk-user-openai"
        assert seen[0]["body"]["model"] == "gpt-4o-mini"

        # Reasoning-only OpenAI models omit temperature & use max_completion_tokens; Ollama colons stay intact
        router.chat("o3-mini", "sys", "usr")
        assert "temperature" not in seen[1]["body"]
        assert "max_completion_tokens" in seen[1]["body"] and "max_tokens" not in seen[1]["body"]
        router.chat("qwen3:8b", "sys", "usr")
        assert seen[2]["body"]["model"] == "qwen3:8b"
    finally:
        client.close()

    # Unknown custom gateway without explicit primary model raises clear RouterConfigError
    custom_dir = tmp_path / "custom_gw"
    custom_dir.mkdir()
    r_custom = LLMRouter(custom_dir, base_url="https://custom.relay.example/v1", api_key="sk-x")
    try:
        with pytest.raises(RouterConfigError, match="NOVEL_WASH_PRIMARY_MODEL"):
            _ = r_custom.default_primary
    finally:
        r_custom.close()


def test_anthropic_gemini_isolation_versioned_urls_and_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret")
    monkeypatch.setenv("NOVEL_WASH_API_KEY", "generic-gateway-key")
    monkeypatch.setenv("NOVEL_WASH_BASE_URL", "https://my-generic-gateway.example/v1")

    (tmp_path / "config").mkdir()
    cfg = json.loads((Path(__file__).resolve().parents[1] / "config" / "models.example.json").read_text())
    cfg["models"]["claude-sonnet-4-5"] = {
        "enabled": True,
        "family": "claude",
        "provider": "anthropic",
        "max_tokens": 1024,
        "temperature": 0.2,
    }
    cfg["models"]["gemini-2.5-flash"] = {
        "enabled": True,
        "family": "gemini",
        "provider": "gemini",
        "model": "models/gemini-2.5-flash",
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    (tmp_path / "config" / "models.json").write_text(json.dumps(cfg), encoding="utf-8")

    # 1. Credential isolation: clearing ANTHROPIC_API_KEY must NOT fall back to NOVEL_WASH_API_KEY!
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    r_iso = LLMRouter(tmp_path)
    try:
        info = r_iso.describe_route("claude-sonnet-4-5")
        assert info["eligible"] is False
        assert info["expected_key_envs"] == ["ANTHROPIC_API_KEY"]
        with pytest.raises(RouterConfigError, match="ANTHROPIC_API_KEY"):
            r_iso.chat("claude-sonnet-4-5", "sys", "hi")
    finally:
        r_iso.close()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-secret")

    # 2. Live mock dispatch: check host, path, headers, usage math, and finish_reason normalization
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append({"url": str(req.url), "headers": dict(req.headers), "body": json.loads(req.content.decode("utf-8"))})
        if "/messages" in str(req.url):
            if "POLICY_REFUSAL" in json.dumps(seen[-1]["body"]):
                return httpx.Response(200, json={"content": [], "stop_reason": "refusal"})
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": '{"ok":true}'}],
                    "stop_reason": "max_tokens",
                    "usage": {"input_tokens": 10, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 5, "output_tokens": 8},
                },
            )
        if ":generateContent" in str(req.url):
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "finishReason": "MAX_TOKENS",
                            "content": {"parts": [{"text": "t", "thought": True}, {"text": '{"g":1}'}]},
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 40, "cachedContentTokenCount": 12, "candidatesTokenCount": 6, "thoughtsTokenCount": 14},
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"openai":true}'}, "finish_reason": "stop"}], "usage": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = LLMRouter(tmp_path, http_client=client)
    try:
        txt_a, met_a = router.chat("anthropic:claude-sonnet-4-5", "sys prompt", "hello")
        assert txt_a == '{"ok":true}' and met_a.finish_reason == "length"
        assert met_a.prompt_tokens == 45 and met_a.cached_tokens == 30
        assert seen[0]["url"] == "https://api.anthropic.com/v1/messages"
        assert seen[0]["headers"]["x-api-key"] == "ant-secret"

        with pytest.raises(RouterContentPolicyError):
            router.chat("claude-sonnet-4-5", "sys", "POLICY_REFUSAL")

        txt_g, met_g = router.chat("gemini-2.5-flash", "gem sys", "hi")
        assert txt_g == '{"g":1}' and met_g.finish_reason == "length"
        assert met_g.completion_tokens == 20 and met_g.reasoning_tokens == 14
        assert seen[-1]["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        assert "?key=" not in seen[-1]["url"]
        assert seen[-1]["headers"]["x-goog-api-key"] == "gem-secret"
    finally:
        client.close()

    # 3. CLI --base-url https://api.anthropic.com infers `anthropic` adapter on `_cli` route,
    #    while explicit `--provider default` still targets `default` (`openai`)!
    client2 = httpx.Client(transport=httpx.MockTransport(handler))
    seen.clear()
    r_cli = LLMRouter(tmp_path, base_url="https://api.anthropic.com", api_key="sk-cli-ant", http_client=client2)
    try:
        r_cli.chat("minimax-m3.1-flash", "sys", "via cli override")
        assert seen[0]["url"] == "https://api.anthropic.com/v1/messages"
        assert seen[0]["headers"]["x-api-key"] == "sk-cli-ant"

        r_cli.chat("minimax-m3.1-flash", "sys", "explicit default", group="default")
        assert seen[1]["url"] == "https://my-generic-gateway.example/v1/chat/completions"
        assert seen[1]["headers"]["authorization"] == "Bearer generic-gateway-key"
    finally:
        client2.close()


def test_legacy_xcm_host_and_gemini_url_security(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Legacy groups preserve XCM_BASE_URL destination host without overwriting to api.openai.com
    monkeypatch.setenv("XCM_BASE_URL", "https://legacy-relay.example/v1")
    monkeypatch.setenv("LEGACY_KEY", "sk-leg")
    monkeypatch.delenv("NOVEL_WASH_BASE_URL", raising=False)
    leg_dir = tmp_path / "leg"
    (leg_dir / "config").mkdir(parents=True)
    (leg_dir / "config" / "models.json").write_text(
        json.dumps(
            {
                "groups": {"std": {"env_key": "LEGACY_KEY", "models": ["glm-5.3-flash"]}},
                "models": {"glm-5.3-flash": {"enabled": True, "family": "glm", "preferred_group": "std"}},
            }
        ),
        encoding="utf-8",
    )
    seen_urls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})

    c_leg = httpx.Client(transport=httpx.MockTransport(handler))
    r_leg = LLMRouter(leg_dir, http_client=c_leg)
    try:
        r_leg.chat("glm-5.3-flash", "s", "u")
        assert seen_urls == ["https://legacy-relay.example/v1/chat/completions"]
    finally:
        c_leg.close()

    # 2. Query credential (?key=...) in base URL is rejected WITHOUT echoing secret; /v1 Gemini version stays /v1
    with pytest.raises(RouterConfigError) as exc_info:
        LLMRouter(tmp_path, base_url="https://generativelanguage.googleapis.com/v1beta?key=topsecret_xyz", api_key="k")
    assert "topsecret_xyz" not in str(exc_info.value)

    r_v1 = LLMRouter(tmp_path, base_url="https://generativelanguage.googleapis.com/v1", api_key="gem-k", api_format="gemini")
    try:
        info = r_v1.describe_route("gemini-2.5-flash")
        assert info["url"] == "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
    finally:
        r_v1.close()


def test_policy_vs_auth_bounded_reasoning_retry_and_pool_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("K1", "key-1")
    monkeypatch.setenv("K2", "key-2")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "p1": {"format": "openai", "base_url": "https://p1.example/v1", "api_key_envs": ["K1"], "models": ["m-a"]},
                    "p2": {"format": "gemini", "base_url": "https://p2.example", "api_key_envs": ["K2"], "models": ["m-b"]},
                },
                "models": {
                    "m-a": {"enabled": True, "family": "minimax", "provider": "p1", "extra_body": {"reasoning_effort": "high"}},
                    "m-b": {"enabled": True, "family": "gemini", "provider": "p2"},
                },
                "pools": {
                    "primary": ["p1/m-b", "p1/m-a"],  # p1/m-b violates p1's allowlist!
                    "low_censorship": ["p2/m-b"],
                },
            }
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode("utf-8"))
        calls.append({"url": str(req.url), "body": body})
        u = str(req.url)
        if "p1.example" in u:
            txt_u = json.dumps(body)
            if "ERR_1301_BIG" in txt_u:
                return httpx.Response(400, json={"error": {"code": 1301000000, "message": "transient error"}})
            if "AUTH_MOD" in txt_u:
                return httpx.Response(403, json={"error": {"code": "invalid_api_key", "message": "bad key on moderation endpoint"}})
            if "BAD_REASONING" in txt_u:
                return httpx.Response(400, json={"error": {"message": "reasoning_effort unsupported"}})
        # Gemini missing candidates without blockReason -> ValueError, NOT policy fallback
        return httpx.Response(200, json={"usageMetadata": {"promptTokenCount": 5}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = LLMRouter(tmp_path, http_client=client)
    try:
        # 1. Pool allowlist parity rejects p1/m-b (only p1/m-a eligible)
        assert router._eligible_pool_routes("primary") == [("p1", "m-a")]

        # 2. code=1301000000 must NOT raise RouterContentPolicyError
        with pytest.raises(RuntimeError) as e_1301:
            router.chat("m-a", "s", "ERR_1301_BIG", max_retries=1)
        assert not isinstance(e_1301.value, RouterContentPolicyError)

        # 3. invalid_api_key mentioning "moderation endpoint" must raise RouterConfigError, NOT policy
        with pytest.raises(RouterConfigError):
            router.chat("m-a", "s", "AUTH_MOD")

        # 4. Bounded reasoning strip with extra_body: exactly 2 calls total (1st has reasoning_effort, 2nd stripped!), no loop!
        calls.clear()
        with pytest.raises(RuntimeError):
            router.chat_pool("primary", "s", "BAD_REASONING")
        assert len(calls) == 2
        assert "reasoning_effort" in calls[0]["body"]
        assert "reasoning_effort" not in calls[1]["body"]

        # 5. Gemini HTTP 200 without candidates or blockReason trips breaker (not policy fallback!)
        with pytest.raises(RuntimeError, match="missing valid candidates"):
            router.chat("m-b", "s", "MALFORMED_GEMINI", max_retries=1)

        with pytest.raises(RuntimeError, match="Malformed HTTP 200 JSON envelope") as e_pool_gem:
            router.chat_pool("low_censorship", "s", "MALFORMED_GEMINI")
        assert not isinstance(e_pool_gem.value, RouterContentPolicyError)
        assert router._breaker("p2/m-b").consecutive_failures >= 1
    finally:
        client.close()


def test_doctor_eligibility_and_sample_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("NOVEL_WASH_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # 1. Doctor --dry-run rejects allowlist-excluded model even when provider has key
    proj = tmp_path / "doc_proj"
    (proj / "config").mkdir(parents=True)
    (proj / "config" / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "only_a": {
                        "format": "openai",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "sk-local-dummy",
                        "models": ["gpt-4o-mini"],
                    }
                },
                "models": {"other-model": {"enabled": True, "provider": "only_a"}},
            }
        ),
        encoding="utf-8",
    )
    rc_bad = cli_mod.main(
        ["doctor", "--project-root", str(proj), "--provider", "only_a", "--primary", "other-model", "--dry-run"]
    )
    assert rc_bad == 1
    out_bad = json.loads(capsys.readouterr().out)
    assert out_bad["ok"] is False and "allowlist" in out_bad["checks"][0]["error"]

    # 2. Bundled offline example cleans polluted fixture with zero API keys
    sample_path = Path(__file__).resolve().parents[1] / "examples" / "sample_polluted.txt"
    out_dir = tmp_path / "sample_out"
    rc_wash = cli_mod.main(["wash", str(sample_path), "--no-model", "--no-epub", "--out", str(out_dir)])
    capsys.readouterr()
    assert rc_wash == 0
    reader_text = (out_dir / "reader" / "sample_polluted.reader.txt").read_text(encoding="utf-8")
    assert "月漪" not in reader_text and "234567891" not in reader_text and "首发网址" not in reader_text
    assert "九百五十三，九百五十四。" in reader_text

    # 3. Live doctor check: rejects empty / truncated probe replies cleanly and omits temperature on o3-mini
    live_bodies: list[dict[str, Any]] = []

    def doc_handler(req: httpx.Request) -> httpx.Response:
        b = json.loads(req.content.decode("utf-8"))
        live_bodies.append(b)
        if len(live_bodies) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "length"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})

    orig_client_cls = httpx.Client

    class _PatchedClient(orig_client_cls):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(doc_handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _PatchedClient)
    rc_trunc = cli_mod.main(
        ["doctor", "--project-root", str(tmp_path), "--base-url", "https://api.openai.com/v1", "--api-key", "sk-t", "--primary", "o3-mini"]
    )
    assert rc_trunc == 1
    out_trunc = json.loads(capsys.readouterr().out)
    assert out_trunc["ok"] is False and "truncated" in out_trunc["checks"][0]["error"]
    assert "temperature" not in live_bodies[0] and "max_completion_tokens" in live_bodies[0]

    rc_empty = cli_mod.main(
        ["doctor", "--project-root", str(tmp_path), "--base-url", "https://api.openai.com/v1", "--api-key", "sk-t", "--primary", "gpt-4o-mini"]
    )
    assert rc_empty == 1
    out_empty = json.loads(capsys.readouterr().out)
    assert out_empty["ok"] is False and "empty" in out_empty["checks"][0]["error"]


def test_codex_repair_regressions_zero_config_anthropic_http200_url_and_strip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from novel_wash.llm_client import _build_request

    # 1. Zero-config two-variable quickstart (NOVEL_WASH_BASE_URL + NOVEL_WASH_API_KEY) works for Anthropic & Gemini,
    #    and CLI overrides stay isolated on `_cli` without mutating `default`!
    monkeypatch.setenv("NOVEL_WASH_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("NOVEL_WASH_API_KEY", "sk-ant-two-var")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append({"url": str(req.url), "headers": dict(req.headers), "body": json.loads(req.content.decode("utf-8"))})
        if "gemini" in str(req.url) and "FAIL_ONCE" in json.dumps(seen[-1]["body"]):
            if len([x for x in seen if "gemini" in x["url"]]) == 1:
                return httpx.Response(400, json={"error": {"message": "reasoning_effort unsupported"}})
            return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "{}"}]}}]})
        if "/messages" in str(req.url):
            return httpx.Response(200, json={"content": [{"type": "text", "text": "unauthorized user quote"}], "stop_reason": "end_turn"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "unauthorized prose ok"}, "finish_reason": "stop"}]})

    c = httpx.Client(transport=httpx.MockTransport(handler))
    zc_dir = tmp_path / "zc_isolated"
    zc_dir.mkdir()
    r = LLMRouter(zc_dir, base_url="https://custom.openai.test/v1", api_key="sk-cli-only", api_format="openai", http_client=c)
    try:
        # Calling explicit `default` still uses two-variable Anthropic env (NOT CLI overrides!) and succeeds on HTTP 200 containing "unauthorized"
        txt_d, _ = r.chat("claude-sonnet-4-5", "s", "u", group="default")
        assert txt_d == "unauthorized user quote"
        assert seen[0]["url"] == "https://api.anthropic.com/v1/messages"
        assert seen[0]["headers"]["x-api-key"] == "sk-ant-two-var"

        # Default CLI route uses `_cli` without touching `default`
        txt_c, _ = r.chat("gpt-4o-mini", "s", "u")
        assert txt_c == "unauthorized prose ok"
        assert seen[1]["url"] == "https://custom.openai.test/v1/chat/completions"
        assert seen[1]["headers"]["authorization"] == "Bearer sk-cli-only"

        # 2. Full-width `＠` authority credential rejection MUST NOT echo password in direct or zero-config paths
        with pytest.raises(RouterConfigError) as e_fw:
            LLMRouter(tmp_path, base_url="https://user:secret_fw_pass＠api.openai.com/v1", api_key="k")
        assert "secret_fw_pass" not in str(e_fw.value)

        monkeypatch.setenv("NOVEL_WASH_BASE_URL", "https://user:secret_zc_pass＠example.invalid/v1")
        with pytest.raises(RouterConfigError) as e_zc_fw:
            LLMRouter(zc_dir)
        assert "secret_zc_pass" not in str(e_zc_fw.value)

        # 3. Non-exact policy codes (`"1301suffix"`, `1301.5`) are NOT misclassified as policy
        from novel_wash.llm_client import _is_content_policy_response
        assert _is_content_policy_response('{"error":{"code":"1301suffix"}}') is False
        assert _is_content_policy_response('{"error":{"code":1301.5}}') is False
        assert _is_content_policy_response('{"error":{"code":1301}}') is True

        # 4. `strip_reasoning=True` strips `extra_body.reasoning_effort` for Anthropic and Gemini too!
        _, _, body_a = _build_request(
            "anthropic", "https://api.anthropic.com", "k", "claude-sonnet-4-5", "s", "u",
            max_tokens=100, temperature=0.0, extra_body={"reasoning_effort": "high"}, strip_reasoning=True,
        )
        assert "reasoning_effort" not in body_a
        _, _, body_g = _build_request(
            "gemini", "https://generativelanguage.googleapis.com", "k", "gemini-2.5-flash", "s", "u",
            max_tokens=100, temperature=0.0, extra_body={"reasoning_effort": "high"}, strip_reasoning=True,
        )
        assert "reasoning_effort" not in body_g
    finally:
        c.close()
