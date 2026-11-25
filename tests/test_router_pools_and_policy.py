"""Tests for model pools, route circuit breaker, runaway cap, Tier-C family exclusion, and policy_scan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from novel_wash.inline_rules import DirtyLineTask, InlineAnalysisResult, InlineCandidateSite
from novel_wash.llm_cleaner import run_llm_inline_cleaning
from novel_wash.llm_client import LLMRouter, RouterContentPolicyError
from novel_wash.policy_scan import scan_book_policy


def _make_pool_project(tmp_path: Path, *, pools: dict[str, list[str]] | None = None) -> Path:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    cfg = {
        "base_url": "https://mock.gateway.test/v1",
        "groups": {
            "welfare": {
                "env_key": "TEST_WELFARE_KEY",
                "discount_multiplier": 0.03,
                "rpm_limit": 1000.0,
                "max_concurrency": 1,
                "fallback_group": "stable",
                "models": ["deepseek-v4.1-flash-1017", "glm-5.3-flash", "hy4-preview"],
            },
            "stable": {
                "env_key": "TEST_STABLE_KEY",
                "discount_multiplier": 0.08,
                "rpm_limit": 1000.0,
                "max_concurrency": 4,
                "fallback_group": None,
                "models": ["minimax-m3.1-flash"],
            },
        },
        "models": {
            "minimax-m3.1-flash": {
                "enabled": True,
                "family": "minimax",
                "preferred_group": "stable",
                "fallback_group": None,
                "reasoning_effort": "none",
                "max_tokens": 8192,
                "temperature": 0,
                "pricing_cny_per_1m": {"stable": {"input": 0.08, "output": 0.32, "cached_input": 0.008}},
            },
            "deepseek-v4.1-flash-1017": {
                "enabled": True,
                "family": "deepseek",
                "preferred_group": "welfare",
                "fallback_group": "stable",
                "reasoning_effort": "none",
                "max_tokens": 4096,
                "temperature": 0,
                "pricing_cny_per_1m": {"welfare": {"input": 0.06, "output": 0.24, "cached_input": 0.0012}},
            },
            "glm-5.3-flash": {
                "enabled": True,
                "family": "glm",
                "preferred_group": "welfare",
                "fallback_group": "stable",
                "reasoning_effort": "low",
                "max_tokens": 8192,
                "temperature": 0,
                "pricing_cny_per_1m": {"welfare": {"input": 0.024, "output": 0.084, "cached_input": 0.0069}},
            },
            "hy4-preview": {
                "enabled": True,
                "family": "hunyuan",
                "preferred_group": "welfare",
                "fallback_group": None,
                "reasoning_effort": "none",
                "max_tokens": 4096,
                "temperature": 0,
                "pricing_cny_per_1m": {"welfare": {"input": 0.18, "output": 0.54, "cached_input": 0.009}},
            },
        },
        "pools": pools
        or {
            "primary": ["stable/minimax-m3.1-flash", "welfare/glm-5.3-flash", "welfare/hy4-preview"],
            "secondary": ["welfare/glm-5.3-flash", "welfare/hy4-preview", "stable/minimax-m3.1-flash"],
            "low_censorship": ["stable/minimax-m3.1-flash", "welfare/deepseek-v4.1-flash-1017"],
        },
    }
    (tmp_path / "config" / "models.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / ".env").write_text("TEST_WELFARE_KEY=wk\nTEST_STABLE_KEY=sk\n", encoding="utf-8")
    return tmp_path


def test_breaker_trips_half_open_recovers_and_permanent_disables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj = _make_pool_project(tmp_path)
    monkeypatch.delenv("TEST_WELFARE_KEY", raising=False)
    monkeypatch.delenv("TEST_STABLE_KEY", raising=False)

    clock = [100.0]
    seen: list[str] = []
    fail_minimax = [True]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        seen.append(model)
        if model == "minimax-m3.1-flash" and fail_minimax[0]:
            return httpx.Response(400, text='{"error":"invalid request parameters"}')
        if model == "glm-5.3-flash":
            return httpx.Response(404, text='{"error":"model_not_found"}')
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok":1}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = LLMRouter(
        proj,
        http_client=client,
        monotonic_fn=lambda: clock[0],
        sleep_fn=lambda sec: clock.__setitem__(0, clock[0] + sec),
    )
    try:
        # Call 1: minimax 400 (1st failure), glm-5.3-flash 404 (permanent!), hy4-preview succeeds
        txt1, m1 = router.chat_pool("primary", "sys", "u1")
        assert txt1 == '{"ok":1}' and m1.model == "hy4-preview"

        # Call 2: minimax 400 (2nd failure -> breaker opens for 60s), glm-5.3-flash skipped (perm disabled), hy4-preview succeeds
        txt2, m2 = router.chat_pool("primary", "sys", "u2")
        assert txt2 == '{"ok":1}' and m2.model == "hy4-preview"
        assert seen.count("glm-5.3-flash") == 1

        # Call 3 at t=110s (< 160s): minimax breaker is OPEN so skipped completely!
        clock[0] = 110.0
        seen.clear()
        txt3, m3 = router.chat_pool("primary", "sys", "u3")
        assert m3.model == "hy4-preview"
        assert "minimax-m3.1-flash" not in seen

        # Call 4 at t=170s (>= 160s): half-open probe restores minimax cleanly!
        clock[0] = 170.0
        fail_minimax[0] = False
        seen.clear()
        txt4, m4 = router.chat_pool("primary", "sys", "u4")
        assert m4.model == "minimax-m3.1-flash"
        assert seen == ["minimax-m3.1-flash"]
    finally:
        router.close()


def test_content_policy_403_no_breaker_and_low_censorship_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj = _make_pool_project(
        tmp_path,
        pools={
            "primary": ["welfare/glm-5.3-flash", "welfare/hy4-preview"],
            "low_censorship": ["stable/minimax-m3.1-flash"],
        },
    )
    monkeypatch.delenv("TEST_WELFARE_KEY", raising=False)
    monkeypatch.delenv("TEST_STABLE_KEY", raising=False)

    minimax_refuses = [False]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "glm-5.3-flash" and "adult" in request.content.decode():
            return httpx.Response(403, text='{"error":"content_policy violation"}')
        if model in ("minimax-m3.1-flash", "hy4-preview") and minimax_refuses[0]:
            return httpx.Response(403, text='{"error":"content_policy"}')
        if model == "minimax-m3.1-flash":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"adult":"ok"}'}, "finish_reason": "stop"}], "usage": {}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"clean":"ok"}'}, "finish_reason": "stop"}], "usage": {}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = LLMRouter(proj, http_client=client, sleep_fn=lambda _: None)
    try:
        # 403 on glm-5.3-flash fails over straight to low_censorship (minimax-m3.1-flash)
        txt, m = router.chat_pool("primary", "sys", "adult scene")
        assert txt == '{"adult":"ok"}' and m.model == "minimax-m3.1-flash"
        assert calls == ["glm-5.3-flash", "minimax-m3.1-flash"]

        # Breaker on glm-5.3-flash was NOT tripped by 403 -> next clean call still uses glm-5.3-flash
        calls.clear()
        txt2, m2 = router.chat_pool("primary", "sys", "clean scene")
        assert m2.model == "glm-5.3-flash"

        # If low_censorship also refuses, propagates RouterContentPolicyError
        minimax_refuses[0] = True
        with pytest.raises(RouterContentPolicyError):
            router.chat_pool("primary", "sys", "adult scene")
    finally:
        router.close()


def test_truncation_failover_cross_family_exclusion_and_legacy_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = _make_pool_project(tmp_path)
    monkeypatch.delenv("TEST_WELFARE_KEY", raising=False)
    monkeypatch.delenv("TEST_STABLE_KEY", raising=False)

    lines = ["甚至衫咝有些单薄。"]
    task = DirtyLineTask(
        line_num=1,
        orig_line=lines[0],
        pre_peeled_line=lines[0],
        peeled_to_orig_offsets=list(range(len(lines[0]))),
        tagged_text="甚至⟦1衫⟧⟦2咝⟧有些单薄。",
        candidates=[
            InlineCandidateSite(1, 2, 3, "衫", "C", "test"),
            InlineCandidateSite(2, 3, 4, "咝", "C", "test"),
        ],
    )
    inline_res = InlineAnalysisResult([], [], {"衫", "咝"}, [task], {1})
    cids = sorted(c.cid for c in task.candidates)
    ln = task.line_num

    seen_calls: list[tuple[str, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        seen_calls.append((model, body["max_tokens"]))
        # 1st primary route (minimax) simulates runaway truncation -> must fail over to glm without splitting!
        if model == "minimax-m3.1-flash":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"L1":{"del":['}, "finish_reason": "length"}], "usage": {}},
            )
        # Fallback primary (glm-5.3-flash) succeeds; then Tier-C MUST exclude family="glm" and call minimax? Wait:
        # If minimax fails in test only on 1st call, let's verify cross-family exclusion cleanly below!
        payload = json.dumps({f"L{ln}": {"del": cids, "keep": []}})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": payload}, "finish_reason": "stop"}], "usage": {}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = LLMRouter(proj, http_client=client, sleep_fn=lambda _: None)
    try:
        ckpt = tmp_path / "ckpt.json"
        # Use a custom pools config where minimax truncates once then succeeds on secondary OR glm->minimax cross-family
        router.config["pools"]["secondary"] = ["welfare/glm-5.3-flash", "stable/minimax-m3.1-flash"]
        edits, review, stats = run_llm_inline_cleaning(
            inline_res,
            lines,
            router,
            primary_model="deepseek-v4.1-flash-1017",
            secondary_model="glm-5.3-flash",
            source_sha256="abc123",
            checkpoint_path=ckpt,
            use_pools=True,
        )
        # Verify runaway cap was bounded (< 4096) and truncated minimax did NOT split; glm-5.3-flash answered primary
        assert seen_calls[0] == ("minimax-m3.1-flash", 2048)
        assert seen_calls[1][0] == "glm-5.3-flash"
        # Tier-C for glm-primary MUST NOT call glm; it MUST pick minimax (or mark secondary_unresolved if minimax also fails)!
        tier_c_models = [m for m, _ in seen_calls[2:]]
        assert all(m != "glm-5.3-flash" for m in tier_c_models)
        assert stats["secondary_unresolved"] == 1
        assert stats["llm_approved_edits"] == 0
        assert all(e.status == "review" for e in edits)  # Tier-C failed closed to review, never approved!

        # Blocker 5: Pooled checkpoint entry missing "fam" MUST be rejected (cache miss, never relabeled!)
        ckpt_data = json.loads(ckpt.read_text(encoding="utf-8"))
        assert str(ln) in ckpt_data["primary"]
        ckpt_data["primary"][str(ln)].pop("fam", None)
        ckpt.write_text(json.dumps(ckpt_data), encoding="utf-8")

        seen_calls.clear()
        router.config["pools"]["primary"] = ["stable/minimax-m3.1-flash"]
        router.config["pools"]["secondary"] = ["welfare/glm-5.3-flash"]

        def handler_ok(request: httpx.Request) -> httpx.Response:
            model = json.loads(request.content)["model"]
            seen_calls.append((model, 0))
            payload = json.dumps({f"L{ln}": {"del": cids, "keep": []}})
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": payload}, "finish_reason": "stop"}], "usage": {}},
            )

        router._breakers.clear()
        router._client = httpx.Client(transport=httpx.MockTransport(handler_ok))
        edits2, _, stats2 = run_llm_inline_cleaning(
            inline_res,
            lines,
            router,
            primary_model="deepseek-v4.1-flash-1017",
            secondary_model="glm-5.3-flash",
            source_sha256="abc123",
            checkpoint_path=ckpt,
            use_pools=True,
        )
        # Missing "fam" entry in pooled mode triggered cache miss -> primary ran freshly on minimax, Tier-C on glm!
        assert seen_calls == [("minimax-m3.1-flash", 0), ("glm-5.3-flash", 0)]
        assert stats2["llm_approved_edits"] >= 1
        assert all(e.source == "llm:consensus(minimax-m3.1-flash+glm-5.3-flash)" for e in edits2)

        # Genuine legacy single-model checkpoint (explicit mode="single") migrates cleanly into pooled mode
        ckpt_legacy = tmp_path / "ckpt_legacy.json"
        legacy_meta = dict(ckpt_data["metadata"])
        legacy_meta["mode"] = "single"
        ckpt_legacy.write_text(
            json.dumps(
                {
                    "metadata": legacy_meta,
                    "primary": {str(ln): {"fp": ckpt_data["primary"][str(ln)]["fp"], "del": cids}},
                    "secondary": {},
                    "unresolved": {"primary": {}, "secondary": {}},
                }
            ),
            encoding="utf-8",
        )
        seen_calls.clear()
        edits3, _, stats3 = run_llm_inline_cleaning(
            inline_res,
            lines,
            router,
            primary_model="deepseek-v4.1-flash-1017",
            secondary_model="glm-5.3-flash",
            source_sha256="abc123",
            checkpoint_path=ckpt_legacy,
            use_pools=True,
        )
        # Primary replayed as deepseek from single-mode metadata; Tier-C ran on glm & approved!
        assert seen_calls == [("glm-5.3-flash", 0)]
        assert stats3["llm_approved_edits"] >= 1
    finally:
        router.close()


def test_policy_scan_political_and_fadeout_hints(tmp_path: Path) -> None:
    from novel_wash import cli as cli_mod

    fantasy_lines = [
        "第1章 王朝风云",
        "修仙宗门长老面见皇帝，朝廷与帝国议和。",
        "他轻轻吻了吻她的红唇，耳垂微红，轻颤着靠在怀里。",
        "……",
        "此处省略一万字，读者自行脑补。",
    ]
    res_f = scan_book_policy(fantasy_lines)
    assert res_f["level"] == "none"
    assert res_f["fadeout_hints"]["explicit_count"] == 1
    assert res_f["fadeout_hints"]["ellipsis_count"] == 1

    # Blocker 8: same Tier-B terms tested in modern (hit), fantasy (suppressed), and neutral (not hit)
    assert scan_book_policy(["网友在2023年讨论意识形态问题。", "媒体报道公安部发布2023年公告。"])["hit_count"] == 2
    assert scan_book_policy(["修仙宗门长老在2023年穿越后谈论意识形态与公安部。"])["hit_count"] == 0
    assert scan_book_policy(["房间里讨论了意识形态与公安部。"])["hit_count"] == 0

    pol_lines = [
        "1989年的北京街头出现六四事件相关讨论。",
        "网友在2023年提到了某地省委书记和公安部通知。",
    ]
    res_p = scan_book_policy(pol_lines)
    assert res_p["level"] == "high"
    assert res_p["tier_a_hits"] == 1
    assert res_p["tier_b_hits"] >= 1
    assert len(res_p["samples"]) >= 1

    # CLI read-only scan-policy creates no output editions
    sample_txt = tmp_path / "sample.txt"
    sample_txt.write_text("\n".join(pol_lines), encoding="utf-8")
    out_json = tmp_path / "pol.json"
    assert cli_mod.main(["scan-policy", str(sample_txt), "--out", str(out_json)]) == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data[0]["level"] == "high"
    assert not (tmp_path / "output").exists()


def test_codex_blockers_3_4_breaker_probe_and_policy_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj = _make_pool_project(
        tmp_path,
        pools={
            "primary": ["stable/minimax-m3.1-flash", "welfare/glm-5.3-flash"],
            "low_censorship": ["stable/minimax-m3.1-flash"],
        },
    )
    monkeypatch.delenv("TEST_WELFARE_KEY", raising=False)
    monkeypatch.delenv("TEST_STABLE_KEY", raising=False)

    # Blocker 3: deferred success trips breaker on bad outputs, pool_concurrency does not steal probe, bad 200 envelope fails over
    clock = [10.0]
    mode = ["malformed_200"]
    seen: list[str] = []

    def h3(request: httpx.Request) -> httpx.Response:
        m = json.loads(request.content)["model"]
        seen.append(m)
        if m == "minimax-m3.1-flash":
            if mode[0] == "malformed_200":
                return httpx.Response(200, text="[]")
            if mode[0] == "ok":
                return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":1}'}, "finish_reason": "stop"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":2}'}, "finish_reason": "stop"}]})

    router = LLMRouter(
        proj,
        http_client=httpx.Client(transport=httpx.MockTransport(h3)),
        monotonic_fn=lambda: clock[0],
        sleep_fn=lambda s: clock.__setitem__(0, clock[0] + s),
    )
    try:
        # Call 1 & Call 2 return malformed JSON [] -> trips minimax breaker cleanly and fails over to glm!
        _, m1 = router.chat_pool("primary", "sys", "u1", defer_success=True)
        assert m1.model == "glm-5.3-flash"
        router.report_good_output(m1.group, m1.model)
        _, m2 = router.chat_pool("primary", "sys", "u2", defer_success=True)
        assert m2.model == "glm-5.3-flash"
        assert router._breaker("stable/minimax-m3.1-flash").open_until > clock[0]

        # Advance past cooldown; pool_concurrency MUST NOT consume half-open probe!
        clock[0] += 70.0
        assert router.pool_concurrency("primary") == 4
        assert router._breaker("stable/minimax-m3.1-flash").half_open_in_flight is False
        mode[0] = "ok"
        seen.clear()
        _, m3 = router.chat_pool("primary", "sys", "u3", defer_success=True)
        assert m3.model == "minimax-m3.1-flash"
        assert seen == ["minimax-m3.1-flash"]
        router.report_good_output(m3.group, m3.model)
    finally:
        router.close()

    # Blocker 4: 403 on minimax with low_censorship only having minimax falls through to healthy glm route, and preserves 403 if glm also refuses
    glm_refuses = [False]
    seen4: list[str] = []

    def h4(request: httpx.Request) -> httpx.Response:
        m = json.loads(request.content)["model"]
        seen4.append(m)
        if m == "minimax-m3.1-flash":
            return httpx.Response(403, text='{"error":"content_policy"}')
        if glm_refuses[0]:
            return httpx.Response(403, text='{"error":"content_policy"}')
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":1}'}, "finish_reason": "stop"}]})

    r4 = LLMRouter(proj, http_client=httpx.Client(transport=httpx.MockTransport(h4)), sleep_fn=lambda _: None)
    try:
        txt, m = r4.chat_pool("primary", "sys", "adult")
        assert txt == '{"ok":1}' and m.model == "glm-5.3-flash"
        assert seen4 == ["minimax-m3.1-flash", "glm-5.3-flash"]

        glm_refuses[0] = True
        with pytest.raises(RouterContentPolicyError):
            r4.chat_pool("primary", "sys", "adult")
    finally:
        r4.close()


def test_codex_re_review_blockers_4_5_6_9_10(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zipfile
    import novel_wash.cli as cli_mod
    from novel_wash.inline_rules import DirtyLineTask, InlineAnalysisResult, InlineCandidateSite
    from novel_wash.llm_cleaner import run_llm_inline_cleaning, task_fingerprint
    from novel_wash.llm_client import LLMRouter
    from novel_wash.policy_scan import scan_path_policy

    monkeypatch.delenv("TEST_WELFARE_KEY", raising=False)
    monkeypatch.delenv("TEST_STABLE_KEY", raising=False)

    # 1. Blocker 4 (Codex): stale deferred OR pre-trip closed-state token cannot clear active probe; usage 1e309 fails over cleanly
    proj = _make_pool_project(
        tmp_path / "r4",
        pools={
            "primary": ["stable/minimax-m3.1-flash", "welfare/glm-5.3-flash", "welfare/deepseek-v4.1-flash-1017"],
            "low_censorship": ["welfare/glm-5.3-flash"],
        },
    )
    clock = [10.0]
    mode = ["ok"]
    seen_req: list[str] = []

    def h_tok(request: httpx.Request) -> httpx.Response:
        m = json.loads(request.content)["model"]
        seen_req.append(m)
        if m == "minimax-m3.1-flash":
            if mode[0] == "inf_usage":
                return httpx.Response(200, text='{"choices":[{"message":{"content":"{\\"ok\\":1}"}}],"usage":{"prompt_tokens":1e309}}')
            if mode[0] == "fail":
                return httpx.Response(503, text="err")
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":1}'}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":2}'}}]})

    router = LLMRouter(
        proj,
        http_client=httpx.Client(transport=httpx.MockTransport(h_tok)),
        monotonic_fn=lambda: clock[0],
        sleep_fn=lambda s: clock.__setitem__(0, clock[0] + s),
    )
    try:
        _, stale_m = router.chat_pool("primary", "sys", "old", defer_success=True)
        br = router._breaker("stable/minimax-m3.1-flash")
        br.record_failure(clock[0], threshold=1)
        clock[0] = br.open_until + 1.0
        _, probe_m = router.chat_pool("primary", "sys", "probe", defer_success=True)
        assert br.half_open_in_flight is True
        # Stale call reporting good/bad/release MUST NOT clear the active half-open probe reservation
        router.report_good_output(stale_m.group, stale_m.model, probe_token=stale_m.probe_token)
        br.release_probe(None)
        br.record_success(stale_m.probe_token, is_deferred=False)
        br.record_failure(clock[0], token=stale_m.probe_token, is_deferred=False)
        assert br.half_open_in_flight is True
        _, concurrent_m = router.chat_pool("primary", "sys", "concurrent", defer_success=True)
        assert concurrent_m.model == "glm-5.3-flash"
        # Settling actual probe clears half_open_in_flight
        router.report_good_output(probe_m.group, probe_m.model, probe_token=probe_m.probe_token)
        assert br.half_open_in_flight is False

        # Late pre-trip sync permanent=True or regular failure CANNOT disable or extend cooldown after recovery
        pre_ok, pre_tok = br.acquire_dispatch(clock[0])
        assert pre_ok and pre_tok is not None
        br.record_failure(clock[0], threshold=1)
        prev_open = br.open_until
        clock[0] = prev_open + 1.0
        probe2_ok, probe2_tok = br.acquire_dispatch(clock[0])
        assert probe2_ok and br.half_open_in_flight is True
        br.record_success(probe2_tok, is_deferred=False)
        br.record_failure(clock[0] + 10.0, permanent=True, token=pre_tok, is_deferred=False)
        assert br.permanently_disabled is False and br.open_until == 0.0

        # Overflowing usage (`1e309` -> OverflowError) on half-open probe trips breaker, releases probe, and fails over cleanly
        br.record_failure(clock[0], threshold=1)
        clock[0] = br.open_until + 1.0
        mode[0] = "inf_usage"
        seen_req.clear()
        _, fallback_m = router.chat_pool("primary", "sys", "bad_u")
        assert fallback_m.model == "glm-5.3-flash"
        assert br.half_open_in_flight is False
        assert seen_req == ["minimax-m3.1-flash", "glm-5.3-flash"]
    finally:
        router.close()

    # 2. Blocker 5 (Codex): A=403, B=503, C=200 with low_censorship=[B] executes A -> B -> C (never retries B)
    seen_fb: list[str] = []

    def h_fb(request: httpx.Request) -> httpx.Response:
        m = json.loads(request.content)["model"]
        seen_fb.append(m)
        if m == "minimax-m3.1-flash":
            return httpx.Response(403, text='{"error":"content_policy"}')
        if m == "glm-5.3-flash":
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":3}'}}]})

    r_fb = LLMRouter(proj, http_client=httpx.Client(transport=httpx.MockTransport(h_fb)), sleep_fn=lambda _: None)
    try:
        _, m_fb = r_fb.chat_pool("primary", "sys", "u")
        assert m_fb.model == "deepseek-v4.1-flash-1017"
        assert seen_fb == ["minimax-m3.1-flash", "glm-5.3-flash", "deepseek-v4.1-flash-1017"]
    finally:
        r_fb.close()

    # 3. Blocker 6 (Codex): unmarked v3 checkpoint rejected, spoofed (mod, fam) rejected, {} nonempty batch trips breaker
    proj_c = _make_pool_project(
        tmp_path / "r6",
        pools={
            "primary": ["stable/minimax-m3.1-flash", "welfare/deepseek-v4.1-flash-1017"],
            "secondary": ["welfare/glm-5.3-flash"],
        },
    )
    line_txt = "测试一行ɑ水印刷。"
    task = DirtyLineTask(1, line_txt, line_txt, list(range(len(line_txt))), "测试一行⟦1ɑ⟧水印刷。", [InlineCandidateSite(1, 4, 5, "ɑ", "C", "c1")])
    inline_res = InlineAnalysisResult([], [], {"ɑ"}, [task], {1})
    fp_val = task_fingerprint(task)
    sys_p_sha = __import__("hashlib").sha256(
        __import__("novel_wash.llm_cleaner", fromlist=["build_system_prompt"]).build_system_prompt([]).encode("utf-8")
    ).hexdigest()[:16]
    unmarked_meta = {
        "schema_version": 3,
        "source_sha256": "s1",
        "primary_model": "deepseek-v4.1-flash-1017",
        "secondary_model": "glm-5.3-flash",
        "primary_family": "deepseek",
        "secondary_family": "glm",
        "group_override": None,
        "prompt_sha256": sys_p_sha,
        "task_count": 1,
    }
    ckpt_unmarked = tmp_path / "ckpt_unmarked.json"
    ckpt_unmarked.write_text(
        json.dumps({"metadata": unmarked_meta, "primary": {"1": {"fp": fp_val, "del": [1]}}, "secondary": {}}),
        encoding="utf-8",
    )
    seen_c: list[str] = []

    def h_empty_then_ok(request: httpx.Request) -> httpx.Response:
        m = json.loads(request.content)["model"]
        seen_c.append(m)
        if m == "minimax-m3.1-flash":
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"L1":{"del":[1],"keep":[]}}'}}]})

    rc = LLMRouter(proj_c, http_client=httpx.Client(transport=httpx.MockTransport(h_empty_then_ok)), sleep_fn=lambda _: None)
    try:
        edits_c, _, _ = run_llm_inline_cleaning(
            inline_res,
            [line_txt],
            rc,
            primary_model="deepseek-v4.1-flash-1017",
            secondary_model="glm-5.3-flash",
            source_sha256="s1",
            checkpoint_path=ckpt_unmarked,
            use_pools=True,
        )
        # Unmarked v3 checkpoint ignored -> minimax returned {} (marked bad & failed over) -> deepseek + glm consensus!
        assert seen_c == ["minimax-m3.1-flash", "deepseek-v4.1-flash-1017", "glm-5.3-flash"]
        assert rc._breaker("stable/minimax-m3.1-flash").consecutive_failures >= 1
        assert len(edits_c) == 1

        # Spoofed (mod, fam) OR unconfigured model (when only `glm-5.3-flash` is configured) MUST be rejected!
        ckpt_spoof = tmp_path / "ckpt_spoof.json"
        ckpt_spoof.write_text(
            json.dumps(
                {
                    "metadata": {**unmarked_meta, "mode": "pooled"},
                    "primary": {"1": {"fp": fp_val, "del": [1], "fam": "glm", "mod": "deepseek-v4.1-flash-1017"}},
                    "secondary": {"1": {"fp": fp_val, "del": [1], "fam": "glm", "mod": "glm-unconfigured-legacy"}},
                }
            ),
            encoding="utf-8",
        )
        seen_c.clear()
        rc._breakers.clear()
        run_llm_inline_cleaning(
            inline_res,
            [line_txt],
            rc,
            primary_model="deepseek-v4.1-flash-1017",
            secondary_model="glm-5.3-flash",
            source_sha256="s1",
            checkpoint_path=ckpt_spoof,
            use_pools=True,
        )
        assert "deepseek-v4.1-flash-1017" in seen_c and "glm-5.3-flash" in seen_c
    finally:
        rc.close()

    # 4. Blocker 9 (Codex): EPUB scan catches inline formatting-newline tags, numeric entities, GB18030, cross-member isolation, invalid --terms, hard-link --out
    ep_pol = tmp_path / "pol.epub"
    ch_inline = (
        '<?xml version="1.0" encoding="utf-8"?><html><body>'
        '<p>他谈起文化\n<em>大革命</em>往事。</p>'
        '<p>另有&#25991;&#21270;\n<em>&#22823;&#38761;&#21629;</em>记录。</p>'
        '<p>文化</p><p>大革命</p>'
        '<x:p xmlns:x="http://www.w3.org/1999/xhtml">文化</x:p><x:p xmlns:x="http://www.w3.org/1999/xhtml">大革命</x:p>'
        '<x.y:html xmlns:x.y="http://www.w3.org/1999/xhtml" xmlns:p.foo="http://www.w3.org/1999/xhtml"><x.y:head><x.y:script>文化大革命</x.y:script></x.y:head>'
        '<x.y:body><x.y:p>文化</x.y:p><x.y:p>大革命</x.y:p>'
        '<x.y:script src="external.js"/><x.y:p>文化大革命</x.y:p><x.y:script>hidden</x.y:script>'
        '<p>文化<p.foo:span>大革命</p.foo:span></p></x.y:body></x.y:html>'
        '<章:p xmlns:章="http://www.w3.org/1999/xhtml">文化</章:p><章:p xmlns:章="http://www.w3.org/1999/xhtml">大革命</章:p>'
        '</body></html>'
    ).encode("utf-8")
    ch_gb = '<?xml version="1.0" encoding="gb18030"?><html><body><p>这是关于文化大革命的旧档。</p></body></html>'.encode("gb18030")
    ch_b1 = '<?xml version="1.0" encoding="utf-8"?><html><body><p>街头出现公安人员巡逻。</p></body></html>'.encode("utf-8")
    ch_b2 = '<?xml version="1.0" encoding="utf-8"?><html><body><p>2024年的现代新闻报道。</p></body></html>'.encode("utf-8")
    with zipfile.ZipFile(ep_pol, "w") as zf:
        zf.writestr("OEBPS/1.xhtml", ch_inline)
        zf.writestr("OEBPS/2.xhtml", ch_gb)
        zf.writestr("OEBPS/3.xhtml", ch_b1)
        zf.writestr("OEBPS/4.xhtml", ch_b2)
    pol_res = scan_path_policy(ep_pol)
    # 5 Tier-A hits (inline+newline <em>, numeric+newline, self-closing script prose, p.foo:span inline, gb18030); splits stay 0!
    assert pol_res["tier_a_hits"] == 5 and pol_res["tier_b_hits"] == 0
    with pytest.raises(FileNotFoundError):
        scan_path_policy(ep_pol, terms_path=tmp_path / "nonexistent_terms.json")
    bad_terms = tmp_path / "bad_terms.json"
    bad_terms.write_text('{"tier_a_exact": "文化大革命"}', encoding="utf-8")
    with pytest.raises(ValueError):
        scan_path_policy(ep_pol, terms_path=bad_terms)

    src_txt = tmp_path / "book.txt"
    src_txt.write_text("第一章 正文\n普通句子。\n", encoding="utf-8")
    hard_link = tmp_path / "book_hardlink.txt"
    __import__("os").link(src_txt, hard_link)
    rc_cli = cli_mod.main(["scan-policy", str(src_txt), "--out", str(hard_link)])
    assert rc_cli != 0
    assert src_txt.read_text(encoding="utf-8") == "第一章 正文\n普通句子。\n"

    # 5. Open-source config/models.example.json uses clean multi-format providers & approved recommended models only
    repo_root = Path(__file__).resolve().parents[1]
    cfg_p = repo_root / "config" / "models.example.json"
    if not cfg_p.exists():
        raise AssertionError("config/models.example.json must exist")
    raw_cfg = cfg_p.read_text(encoding="utf-8")
    assert "welfare" not in raw_cfg.lower() and "discount_multiplier" not in raw_cfg
    cfg_data = json.loads(raw_cfg)
    provs = cfg_data.get("providers", {})
    assert {"openai", "anthropic", "gemini"}.issubset({p.get("format") for p in provs.values()})
    assert "api.openai.com" not in str(provs.get("default", {}).get("base_url", ""))
    assert "NOVEL_WASH_API_KEY" not in provs.get("anthropic", {}).get("api_key_envs", [])
    assert "NOVEL_WASH_API_KEY" not in provs.get("gemini", {}).get("api_key_envs", [])
    approved_models = {
        "minimax-m3.1-flash",
        "deepseek-v4.1-flash-1017",
        "deepseek-v4.1-flash",
        "glm-5.3-flash",
        "hy4-preview",
        "hy3",
    }
    configured_models = set(cfg_data.get("models", {}).keys())
    for routes in cfg_data.get("pools", {}).values():
        for r in routes:
            configured_models.add(r.split("/", 1)[-1])
    configured_models.update(cfg_data.get("default_routing", {}).values())
    assert configured_models and configured_models.issubset(approved_models)

