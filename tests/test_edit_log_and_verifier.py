from pathlib import Path

import pytest

from novel_wash.edit_log import (
    EditOp,
    apply_edits_to_line,
    render_ai_edition,
    render_reader_edition,
)
from novel_wash.inline_rules import DirtyLineTask, InlineCandidateSite
from novel_wash.verifier import verify_line_decisions


def test_apply_edits_g4_and_error_guards():
    orig = "所以三条败犬希霖望趁着巴最后"
    edits = [
        EditOp(line=1, start=7, end=8, op="del", before="霖", after="", tier="llm_tier_n", reason="test", source="test"),
        EditOp(line=1, start=11, end=12, op="del", before="巴", after="", tier="llm_tier_n", reason="test", source="test"),
    ]
    assert apply_edits_to_line(orig, edits) == "所以三条败犬希望趁着最后"

    # Overlapping edit must fail
    bad_overlap = edits + [
        EditOp(line=1, start=7, end=8, op="del", before="霖", after="", tier="llm_tier_n", reason="dup", source="test")
    ]
    with pytest.raises(ValueError, match="Overlapping"):
        apply_edits_to_line(orig, bad_overlap)

    # Mismatched before text must fail
    bad_mismatch = [
        EditOp(line=1, start=4, end=5, op="del", before="X", after="", tier="llm_tier_n", reason="bad", source="test")
    ]
    with pytest.raises(ValueError, match="Source mismatch"):
        apply_edits_to_line(orig, bad_mismatch)


def test_pointwise_verify_partial_acceptance_protects_tier_c():
    orig = "所以三条败犬希霖望"
    task = DirtyLineTask(
        line_num=1,
        orig_line=orig,
        pre_peeled_line=orig,
        peeled_to_orig_offsets=list(range(len(orig))),
        tagged_text="所以⟦1三⟧条败犬希⟦2霖⟧望",
        candidates=[
            InlineCandidateSite(cid=1, orig_start=2, orig_end=3, char="三", tier="C", reason="soft_protected_lexicon"),
            InlineCandidateSite(cid=2, orig_start=7, orig_end=8, char="霖", tier="N", reason="eligible_watermark_homophone"),
        ],
    )
    # Primary wrongly votes to delete both 1 and 2; secondary keeps 1 and deletes 2
    outcome = verify_line_decisions(
        task,
        primary_del_cids={1, 2},
        primary_model="deepseek-v4.1-flash-1017",
        secondary_del_cids={2},
        secondary_model="minimax-m3.1-flash",
    )
    approved = [e for e in outcome.edits if e.status == "approved"]
    review = [e for e in outcome.edits if e.status == "review"]
    assert len(approved) == 1 and approved[0].before == "霖"
    assert len(review) == 1 and review[0].before == "三"

    # Conservative apply leaves '三' intact while still deleting '霖'!
    assert apply_edits_to_line(orig, outcome.edits, include_review=False) == "所以三条败犬希望"


def test_dual_renderers_preserve_ai_lines_and_collapse_reader_lines():
    orig_lines = [
        "第一章 开始",
        "饵衫*丝捂}⑹qi八久医",
        "“你好。”",
        "本文件来自尚香书院。",
        "正文第二行。",
    ]
    edits = [
        EditOp(line=2, start=0, end=len(orig_lines[1]), op="clear_line", before=orig_lines[1], after="", tier="soup", reason="soup", source="rule"),
        EditOp(line=4, start=0, end=len(orig_lines[3]), op="clear_line", before=orig_lines[3], after="", tier="ad", reason="ad", source="rule"),
    ]
    ai = render_ai_edition(orig_lines, edits)
    assert len(ai.splitlines()) == 5
    assert ai.splitlines()[1] == ""
    assert ai.splitlines()[2] == "“你好。”"

    reader = render_reader_edition(orig_lines, edits)
    assert "饵衫" not in reader
    assert "尚香书院" not in reader
    assert "第一章 开始\n\n“你好。”\n正文第二行。\n" == reader


def test_llm_cleaner_splits_truncated_batches_and_recovers(tmp_path):
    from novel_wash.llm_cleaner import _run_model_over_tasks, build_system_prompt

    class _DummyLedger:
        def total_cost_cny(self) -> float:
            return 0.0

    class _StubRouter:
        def __init__(self) -> None:
            self.ledger = _DummyLedger()
            self.calls: list[int] = []

        def chat(self, model, sys_prompt, user_prompt, **kwargs):
            lines = [l for l in user_prompt.splitlines() if l.startswith("[L")]
            self.calls.append(len(lines))
            if len(lines) > 4:
                return '{"L1": {"del":', None  # Simulate truncated output on large batch
            resp = {}
            for l in lines:
                ln = l.split("]", 1)[0][1:]
                resp[ln] = {"del": [1], "keep": []}
            import json
            return json.dumps(resp), None

    orig = "测试霖文字。"
    tasks = [
        DirtyLineTask(
            line_num=i,
            orig_line=orig,
            pre_peeled_line=orig,
            peeled_to_orig_offsets=list(range(len(orig))),
            tagged_text="测试⟦1霖⟧文字。",
            candidates=[InlineCandidateSite(1, 2, 3, "霖", "N", "test")],
        )
        for i in range(1, 9)
    ]
    peeled = [orig] * 8
    router = _StubRouter()
    res, unresolved = _run_model_over_tasks(
        tasks,
        peeled,
        build_system_prompt([]),
        router,  # type: ignore[arg-type]
        "stub",
        batch_size=8,
        checkpoint_data={"primary": {}},
        checkpoint_key="primary",
        checkpoint_path=tmp_path / "ckpt.json",
    )
    assert router.calls[0] == 8
    assert all(c <= 4 for c in router.calls[1:])
    assert unresolved == set()
    assert all(res[i] == {1} for i in range(1, 9))


def test_llm_cleaner_concurrent_stable_execution(tmp_path):
    import json
    import threading
    from novel_wash.llm_cleaner import _run_model_over_tasks, build_system_prompt

    class _DummyLedger:
        def total_cost_cny(self) -> float:
            return 0.0

    class _ConcurrentStubRouter:
        def __init__(self) -> None:
            self.ledger = _DummyLedger()
            self.config = {
                "groups": {"tier_b": {"max_concurrency": 4}},
                "models": {"stub": {"preferred_group": "tier_b"}},
            }
            self.lock = threading.Lock()
            self.thread_names: set[str] = set()

        def chat(self, model, sys_prompt, user_prompt, **kwargs):
            with self.lock:
                self.thread_names.add(threading.current_thread().name)
            resp = {}
            for l in user_prompt.splitlines():
                if l.startswith("[L"):
                    ln = l.split("]", 1)[0][1:]
                    resp[ln] = {"del": [1], "keep": []}
            return json.dumps(resp), None

    orig = "测试霖文字。"
    tasks = [
        DirtyLineTask(
            line_num=i,
            orig_line=orig,
            pre_peeled_line=orig,
            peeled_to_orig_offsets=list(range(len(orig))),
            tagged_text="测试⟦1霖⟧文字。",
            candidates=[InlineCandidateSite(1, 2, 3, "霖", "N", "test")],
        )
        for i in range(1, 25)
    ]
    peeled = [orig] * 24
    router = _ConcurrentStubRouter()
    res, unresolved = _run_model_over_tasks(
        tasks,
        peeled,
        build_system_prompt([]),
        router,  # type: ignore[arg-type]
        "stub",
        batch_size=4,
        checkpoint_data={"primary": {}},
        checkpoint_key="primary",
        checkpoint_path=tmp_path / "ckpt.json",
    )
    assert unresolved == set()
    assert len(res) == 24
    assert all(res[i] == {1} for i in range(1, 25))
    ckpt = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
    assert len(ckpt["primary"]) == 24


def test_blocker2_tier_c_cross_family_and_strict_decision_parser():
    from novel_wash.llm_cleaner import parse_batch_decisions

    orig = "所以三条败犬希霖望"
    task = DirtyLineTask(
        line_num=1,
        orig_line=orig,
        pre_peeled_line=orig,
        peeled_to_orig_offsets=list(range(len(orig))),
        tagged_text="所以⟦1三⟧条败犬希⟦2霖⟧望",
        candidates=[
            InlineCandidateSite(cid=1, orig_start=2, orig_end=3, char="三", tier="C", reason="soft"),
            InlineCandidateSite(cid=2, orig_start=7, orig_end=8, char="霖", tier="N", reason="elig"),
        ],
    )
    # Same-family models (deepseek-v4.1-flash + deepseek-v4.1-flash-1017) MUST NOT approve Tier-C
    same_fam = verify_line_decisions(
        task,
        primary_del_cids={1, 2},
        primary_model="deepseek-v4.1-flash-1017",
        secondary_del_cids={1, 2},
        secondary_model="deepseek-v4.1-flash",
    )
    tier_c_same = [e for e in same_fam.edits if e.tier == "llm_tier_c"]
    assert len(tier_c_same) == 1 and tier_c_same[0].status == "review"
    assert same_fam.review_rows[0]["reason"] == "tier_c_same_family_disallowed"

    # Distinct families DO approve Tier-C on consensus
    diff_fam = verify_line_decisions(
        task,
        primary_del_cids={1, 2},
        primary_model="deepseek-v4.1-flash-1017",
        secondary_del_cids={1, 2},
        secondary_model="glm-5.3-flash",
    )
    assert all(e.status == "approved" for e in diff_fam.edits)

    # Strict parser rejects bool True, empty {}, overlap, and incomplete partitions
    for bad_json in (
        '{"L1": {"del": [true], "keep": [2]}}',
        '{"L1": {}}',
        '{"L1": {"del": [1], "keep": [1, 2]}}',
        '{"L1": {"del": [1], "keep": []}}',
    ):
        dm, missing = parse_batch_decisions(bad_json, [task])
        assert dm == {} and len(missing) == 1


def test_blocker3_unresolved_retry_recursive_split_and_fingerprint(tmp_path):
    import json
    from novel_wash.llm_cleaner import build_system_prompt, run_llm_inline_cleaning
    from novel_wash.inline_rules import InlineAnalysisResult
    from novel_wash.llm_client import RouterConfigError

    class _DummyLedger:
        def total_cost_cny(self) -> float:
            return 0.0

    orig1 = "测试霖文字。"
    orig2 = "测试巴文字。"
    t1 = DirtyLineTask(1, orig1, orig1, list(range(len(orig1))), "测试⟦1霖⟧文字。", [InlineCandidateSite(1, 2, 3, "霖", "N", "t")])
    t2 = DirtyLineTask(2, orig2, orig2, list(range(len(orig2))), "测试⟦1巴⟧文字。", [InlineCandidateSite(1, 2, 3, "巴", "N", "t")])
    ir = InlineAnalysisResult(pre_edits=[], proper_nouns=[], eligible_cjk={"霖", "巴"}, tasks=[t1, t2], dirty_line_nums={1, 2})

    class _PartialFailRouter:
        def __init__(self, fail_l2: bool) -> None:
            self.ledger = _DummyLedger()
            self.config = {
                "groups": {"tier_a": {"max_concurrency": 1}},
                "models": {
                    "deepseek-v4.1-flash-1017": {"family": "deepseek", "preferred_group": "tier_a"},
                    "glm-5.3-flash": {"family": "glm", "preferred_group": "tier_a"},
                },
            }
            self.fail_l2 = fail_l2
            self.seen_lns: list[list[str]] = []

        def chat(self, model, sys_prompt, user_prompt, **kwargs):
            lns = [l.split("]", 1)[0][1:] for l in user_prompt.splitlines() if l.startswith("[L")]
            self.seen_lns.append(lns)
            if lns == ["L1", "L2"]:
                return '{"L1": {"del": [1], "keep": []}}', None
            if self.fail_l2 and lns == ["L2"]:
                return "not-json", None
            return json.dumps({ln: {"del": [1], "keep": []} for ln in lns}), None

    ckpt = tmp_path / "ckpt_v2.json"
    edits1, rev1, st1 = run_llm_inline_cleaning(
        ir, [orig1, orig2], _PartialFailRouter(fail_l2=True),  # type: ignore[arg-type]
        primary_model="deepseek-v4.1-flash-1017",
        secondary_model="glm-5.3-flash",
        source_sha256="abc",
        checkpoint_path=ckpt,
    )
    assert st1["incomplete"] is True and st1["primary_unresolved"] == 1
    assert any(r["reason"] == "llm_primary_unresolved" for r in rev1)
    assert len(edits1) == 1 and edits1[0].line == 1
    saved = json.loads(ckpt.read_text())
    assert "2" not in saved["primary"] and "2" in saved["unresolved"]["primary"]

    # Next run MUST retry L2 (L1 reused from fingerprint cache, L2 retried cleanly)
    r2 = _PartialFailRouter(fail_l2=False)
    edits2, rev2, st2 = run_llm_inline_cleaning(
        ir, [orig1, orig2], r2,  # type: ignore[arg-type]
        primary_model="deepseek-v4.1-flash-1017",
        secondary_model="glm-5.3-flash",
        source_sha256="abc",
        checkpoint_path=ckpt,
    )
    assert r2.seen_lns == [["L2"]]
    assert st2["incomplete"] is False and st2["primary_unresolved"] == 0
    assert len(edits2) == 2

    # Changing candidate char invalidates per-task fingerprint even with same metadata/line number!
    t2_changed = DirtyLineTask(2, orig2, orig2, list(range(len(orig2))), "测试⟦1木⟧文字。", [InlineCandidateSite(1, 2, 3, "木", "N", "t")])
    ir_changed = InlineAnalysisResult(pre_edits=[], proper_nouns=[], eligible_cjk={"霖", "木"}, tasks=[t1, t2_changed], dirty_line_nums={1, 2})
    r3 = _PartialFailRouter(fail_l2=False)
    run_llm_inline_cleaning(
        ir_changed, [orig1, orig2], r3,  # type: ignore[arg-type]
        primary_model="deepseek-v4.1-flash-1017",
        secondary_model="glm-5.3-flash",
        source_sha256="abc",
        checkpoint_path=ckpt,
    )
    assert r3.seen_lns == [["L2"]]

    # Fatal RouterConfigError must propagate, never silently stored as unresolved
    class _FatalRouter(_PartialFailRouter):
        def chat(self, model, sys_prompt, user_prompt, **kwargs):
            raise RouterConfigError("disabled")

    with pytest.raises(RouterConfigError):
        run_llm_inline_cleaning(
            ir, [orig1, orig2], _FatalRouter(fail_l2=False),  # type: ignore[arg-type]
            primary_model="deepseek-v4.1-flash-1017",
            secondary_model="glm-5.3-flash",
            source_sha256="new",
            checkpoint_path=tmp_path / "fatal.json",
        )


def test_blocker5_bounds_newlines_precedence_and_pua_ad_composition(tmp_path):
    from novel_wash.cli import _merge_non_overlapping, wash_book
    from novel_wash.edit_log import apply_and_verify

    # Out-of-range line and newline in replacement must raise ValueError
    with pytest.raises(ValueError, match="out of range"):
        apply_and_verify(["正常行"], [EditOp(2, 0, 1, "del", "正", "", "soup", "oob", "test")])
    with pytest.raises(ValueError, match="Line break"):
        apply_and_verify(["正常行"], [EditOp(1, 0, 1, "sub", "正", "a\nb", "soup", "nl", "test")])

    # Approval precedence: approved beats review on overlapping spans
    rev = EditOp(1, 2, 3, "del", "三", "", "llm_tier_c", "rev", "test", status="review")
    app = EditOp(1, 2, 3, "del", "三", "", "llm_tier_n", "app", "test", status="approved")
    merged = _merge_non_overlapping([rev, app])
    assert len(merged) == 1 and merged[0].status == "approved"

    # Cross-layer PUA + inline ad composition: PUA decode completes "记住网址https://abc.com"
    import json
    from collections import Counter
    from novel_wash.pua_solver import _fingerprint_codepoints, _source_pua_fingerprint

    book = tmp_path / "pua_ad.txt"
    pua_line = "\ue3e8住网址https://abc.com \ue3e9暗里他\ue3ea现了。\n"
    book.write_text(pua_line * 3, encoding="utf-8")
    (tmp_path / "cache" / "pua_maps").mkdir(parents=True)
    lines = book.read_text(encoding="utf-8").splitlines()
    cps = ["\ue3e8", "\ue3e9", "\ue3ea"]
    fp = _fingerprint_codepoints(cps)
    sfp = _source_pua_fingerprint(lines, cps, Counter({c: 3 for c in cps}))
    (tmp_path / "cache" / "pua_maps" / f"pua_map_{fp}_{sfp[:8]}.json").write_text(
        json.dumps(
            {
                "fingerprint": fp,
                "source_fingerprint": sfp,
                "char_map": {"\ue3e8": "记", "\ue3e9": "阴", "\ue3ea": "发"},
                "uncertain": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    res = wash_book(book, project_root=tmp_path, out_dir=tmp_path / "out", use_model=False, render_epub=False)
    assert Path(res["ai_path"]).read_text(encoding="utf-8").splitlines()[0] == " 阴暗里他发现了。"


def _legacy_groups_fixture() -> dict:
    return {
        "base_url": "https://legacy-gateway.example/v1",
        "groups": {
            "tier_a": {
                "env_key": "LEGACY_PRIMARY_API_KEY",
                "discount_multiplier": 0.03,
                "rpm_limit": 4.5,
                "max_concurrency": 2,
                "fallback_group": "tier_b",
                "models": ["deepseek-v4.1-flash-1017", "glm-5.3-flash"],
            },
            "tier_b": {
                "env_key": "LEGACY_FALLBACK_API_KEY",
                "discount_multiplier": 0.08,
                "rpm_limit": 20.0,
                "pool_rpm_limit": 120.0,
                "max_concurrency": 10,
                "fallback_group": None,
                "models": ["minimax-m3.1-flash", "deepseek-v4.1-flash-1017", "glm-5.3-flash"],
            },
        },
        "models": {
            "minimax-m3.1-flash": {
                "enabled": True,
                "family": "minimax",
                "preferred_group": "tier_b",
                "fallback_group": None,
                "max_tokens": 8192,
                "temperature": 0,
            },
            "deepseek-v4.1-flash-1017": {
                "enabled": True,
                "family": "deepseek",
                "preferred_group": "tier_a",
                "fallback_group": "tier_b",
                "max_tokens": 4096,
                "temperature": 0,
            },
            "glm-5.3-flash": {
                "enabled": True,
                "family": "glm",
                "preferred_group": "tier_a",
                "fallback_group": "tier_b",
                "max_tokens": 8192,
                "temperature": 0,
            },
        },
    }


def test_router_mock_http_429_failover_and_disabled_model(tmp_path, monkeypatch):
    import json
    import httpx
    from novel_wash.llm_client import LLMRouter, RouterConfigError

    (tmp_path / "config").mkdir()
    cfg = _legacy_groups_fixture()
    cfg["models"]["kimi-k3.1-pro"] = {"enabled": False, "family": "kimi", "preferred_group": "tier_b"}
    (tmp_path / "config" / "models.json").write_text(json.dumps(cfg))
    monkeypatch.delenv("NOVEL_WASH_BASE_URL", raising=False)
    monkeypatch.delenv("XCM_BASE_URL", raising=False)
    monkeypatch.setenv("LEGACY_PRIMARY_API_KEY", "test-p-key")
    monkeypatch.setenv("LEGACY_FALLBACK_API_KEY", "test-fb-key")

    seen_auths: list[str] = []
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auths.append(request.headers.get("Authorization", ""))
        seen_urls.append(str(request.url))
        if len(seen_auths) == 1:
            return httpx.Response(429, json={"error": "rate_limited"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"L1":{"del":[1],"keep":[]}}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    fake_now = [0.0]
    sleeps: list[float] = []

    def fake_mono() -> float:
        return fake_now[0]

    def fake_sleep(sec: float) -> None:
        sleeps.append(sec)
        fake_now[0] += sec

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = LLMRouter(tmp_path, http_client=client, monotonic_fn=fake_mono, sleep_fn=fake_sleep)
    try:
        with pytest.raises(RouterConfigError, match="disabled"):
            router.chat("kimi-k3.1-pro", "sys", "usr")

        txt, metrics = router.chat("deepseek-v4.1-flash-1017", "sys", "usr")
        assert '"del"' in txt
        assert metrics.group == "tier_b"
        assert router.ledger.fallbacks == 1
        assert seen_auths == ["Bearer test-p-key", "Bearer test-fb-key"]
        assert seen_urls == [
            "https://legacy-gateway.example/v1/chat/completions",
            "https://legacy-gateway.example/v1/chat/completions",
        ]

        # Second call on tier_b enforces 20 RPM interval (3.0s spacing via fake clock)
        sleeps.clear()
        router.chat("deepseek-v4.1-flash-1017", "sys", "usr", group="tier_b")
        assert any(s >= 2.9 for s in sleeps)

        # Missing credentials raises RouterConfigError
        monkeypatch.delenv("LEGACY_FALLBACK_API_KEY", raising=False)
        r_nokey = LLMRouter(tmp_path, http_client=client, monotonic_fn=fake_mono, sleep_fn=fake_sleep)
        with pytest.raises(RouterConfigError):
            r_nokey.chat("minimax-m3.1-flash", "sys", "usr")
    finally:
        client.close()


def test_content_policy_403_splits_batch_and_isolates_singleton(tmp_path, monkeypatch):
    import json
    import httpx
    from novel_wash.inline_rules import InlineAnalysisResult
    from novel_wash.llm_cleaner import run_llm_inline_cleaning
    from novel_wash.llm_client import LLMRouter, RouterConfigError, RouterContentPolicyError

    (tmp_path / "config").mkdir()
    cfg = _legacy_groups_fixture()
    (tmp_path / "config" / "models.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("LEGACY_PRIMARY_API_KEY", "test-p-key")

    # 1. Verify LLMRouter distinguishes 403 content policy from 403 invalid key immediately
    def policy_then_auth(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        usr = body["messages"][1]["content"]
        if "AUTH_FAIL" in usr:
            return httpx.Response(403, json={"error": {"message": "invalid_api_key"}})
        return httpx.Response(
            403,
            json={"error": {"message": "内容审计命中风险规则", "type": "content_policy_violation"}},
        )

    c_test = httpx.Client(transport=httpx.MockTransport(policy_then_auth))
    r_test = LLMRouter(tmp_path, allow_welfare_fallback=False, http_client=c_test)
    try:
        with pytest.raises(RouterContentPolicyError):
            r_test.chat("deepseek-v4.1-flash-1017", "sys", "sensitive text")
        with pytest.raises(RouterConfigError):
            r_test.chat("deepseek-v4.1-flash-1017", "sys", "AUTH_FAIL")
    finally:
        c_test.close()

    # 2. Verify batch splits around content-blocked lines, retries without [ctx] on singleton, and isolates L3 only
    peeled_lines = [
        "测试霖一。",
        "敏感上下文ctx。",  # line 2 (ctx above L3, below L1? Actually ctx around L2 is line 2? Make line 2 non-dirty ctx!)
        "测试巴三。",
        "测试巴四。",
        "测试巴五。",
    ]
    t1 = DirtyLineTask(1, peeled_lines[0], peeled_lines[0], [0, 1, 2, 3, 4], "测试⟦1霖⟧一。", [InlineCandidateSite(1, 2, 3, "霖", "N", "t")])
    t3 = DirtyLineTask(3, peeled_lines[2], peeled_lines[2], [0, 1, 2, 3, 4], "测试⟦1巴⟧三。", [InlineCandidateSite(1, 2, 3, "巴", "N", "t")])
    t4 = DirtyLineTask(4, peeled_lines[3], peeled_lines[3], [0, 1, 2, 3, 4], "永远违禁词⟦1巴⟧四。", [InlineCandidateSite(1, 2, 3, "巴", "N", "t")])
    t5 = DirtyLineTask(5, peeled_lines[4], peeled_lines[4], [0, 1, 2, 3, 4], "测试⟦1巴⟧五。", [InlineCandidateSite(1, 2, 3, "巴", "N", "t")])
    ir = InlineAnalysisResult([], [], {"霖", "巴"}, [t1, t3, t4, t5], {1, 3, 4, 5})

    class _PolicyStubRouter:
        def __init__(self) -> None:
            self.ledger = type("L", (), {"total_cost_cny": staticmethod(lambda: 0.0)})()
            self.config = cfg

        def chat(self, model, sys_prompt, user_prompt, **kwargs):
            if "敏感上下文" in user_prompt or "永远违禁词" in user_prompt:
                raise RouterContentPolicyError("HTTP 403 content_policy_violation")
            resp = {}
            for line in user_prompt.splitlines():
                if line.startswith("[L"):
                    ln = line.split("]", 1)[0][1:]
                    resp[ln] = {"del": [1], "keep": []}
            return json.dumps(resp), None

    ckpt = tmp_path / "policy_ckpt.json"
    edits, rev, stats = run_llm_inline_cleaning(
        ir,
        peeled_lines,
        _PolicyStubRouter(),  # type: ignore[arg-type]
        primary_model="deepseek-v4.1-flash-1017",
        secondary_model="glm-5.3-flash",
        source_sha256="p403",
        checkpoint_path=ckpt,
    )
    assert stats["primary_unresolved"] == 1
    assert stats["incomplete"] is True
    assert {e.line for e in edits} == {1, 3, 5}
    saved = json.loads(ckpt.read_text())
    assert set(saved["primary"].keys()) == {"1", "3", "5"}
    assert "4" not in saved["primary"] and "4" in saved["unresolved"]["primary"]
    assert any(r["line"] == 4 and r["reason"] == "llm_primary_unresolved" for r in rev)


def test_round2_verifier_parser_checkpoint_and_replay_edge_cases(tmp_path):
    import json
    from novel_wash.cli import _compose_pua_and_stage_b
    from novel_wash.edit_log import apply_and_verify
    from novel_wash.inline_rules import InlineAnalysisResult
    from novel_wash.llm_cleaner import (
        _run_batch_recursive,
        _run_model_over_tasks,
        parse_batch_decisions,
        run_llm_inline_cleaning,
        task_fingerprint,
    )
    from novel_wash.llm_client import RouterContentPolicyError

    orig = "甲木乙"
    c1 = InlineCandidateSite(1, 1, 2, "木", "C", "t")
    task = DirtyLineTask(2, orig, orig, [0, 1, 2], "甲⟦1木⟧乙", [c1])

    # 1. Fail-closed family resolution: vendor alias & unknown model pairs cannot approve Tier-C
    for m_sec in ("vendor/deepseek-v4-flash", "unknown-alias-v1"):
        out_rev = verify_line_decisions(
            task,
            primary_del_cids={1},
            primary_model="deepseek-v4-flash",
            secondary_del_cids={1},
            secondary_model=m_sec,
        )
        assert out_rev.edits[0].status == "review"
    out_unk = verify_line_decisions(
        task, primary_del_cids={1}, primary_model="foo-1", secondary_del_cids={1}, secondary_model="bar-2"
    )
    assert out_unk.edits[0].status == "review"

    # 2. Duplicate contradictory JSON keys rejected
    with pytest.raises(json.JSONDecodeError, match="Duplicate"):
        parse_batch_decisions('{"L2":{"del":[],"keep":[1]},"L2":{"del":[1],"keep":[]}}', [task])

    # 3. Invalid matching-fp checkpoint vote del:[999] triggers retry instead of coerced all-keep
    class _RetryRouter:
        def __init__(self) -> None:
            self.ledger = type("L", (), {"total_cost_cny": staticmethod(lambda: 0.0)})()
            self.calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            return '{"L2": {"del": [1], "keep": []}}', None

    peeled = ["前一行正文", orig, "后一行正文"]
    fp_ctx = task_fingerprint(task, prev_ctx="前一行正文", next_ctx="后一行正文")
    ckpt_bad = {"primary": {"2": {"fp": fp_ctx, "del": [999]}}, "unresolved": {"primary": {}}}
    rr = _RetryRouter()
    res, unresolved = _run_model_over_tasks(
        [task], peeled, "", rr, "deepseek-v4.1-flash-1017", batch_size=5, checkpoint_data=ckpt_bad, checkpoint_key="primary", checkpoint_path=None  # type: ignore[arg-type]
    )
    assert rr.calls == 1 and res == {2: {1}} and unresolved == set()

    # 4. Changing neighbor [ctx] invalidates cached vote
    peeled_changed = ["被修改的前一行", orig, "后一行正文"]
    rr2 = _RetryRouter()
    res2, _ = _run_model_over_tasks(
        [task], peeled_changed, "", rr2, "deepseek-v4.1-flash-1017", batch_size=5, checkpoint_data=ckpt_bad, checkpoint_key="primary", checkpoint_path=None  # type: ignore[arg-type]
    )
    assert rr2.calls == 1 and res2 == {2: {1}}

    # 5. Bare singleton retry after RouterContentPolicyError propagates fatal ValueError
    class _FatalBareRouter:
        def chat(self, model, sp, usr, **kwargs):
            if "[ctx]" in usr:
                raise RouterContentPolicyError("blocked")
            raise ValueError("fatal router bug")

    with pytest.raises(ValueError, match="fatal router bug"):
        _run_batch_recursive([task], peeled, "", _FatalBareRouter(), "m", None)  # type: ignore[arg-type]

    # 6. Review clear_line preserved on unrelated line in composition, never trumps approved span, and invalid op rejected
    pua_op = EditOp(1, 0, 1, "sub", "甲", "木", "pua", "x", "x")
    rev_clear_2 = EditOp(2, 0, len(orig), "clear_line", orig, "", "soup", "rev_clr", "x", status="review")
    composed = _compose_pua_and_stage_b(["甲", orig], [pua_op], [rev_clear_2], [])
    assert rev_clear_2 in composed

    app_sub_1 = EditOp(1, 0, 1, "sub", "甲", "木", "pua", "x", "x", status="approved")
    rev_clear_1 = EditOp(1, 0, 2, "clear_line", "甲乙", "", "soup", "rev_clr", "x", status="review")
    replayed_inc_rev, _ = apply_and_verify(["甲乙"], [app_sub_1, rev_clear_1], include_review=True)
    assert replayed_inc_rev == ["木乙"]

    bad_op = EditOp(1, 0, 1, "bogus", "甲", "\nx", "x", "t", "x")
    with pytest.raises(ValueError):
        apply_and_verify(["甲乙"], [EditOp(1, 0, 2, "clear_line", "甲乙", "", "x", "t", "x"), bad_op])


def test_round4_ai_whitespace_preservation_json_array_reject_and_chapter_headings():
    import json
    from novel_wash.edit_log import apply_and_verify, is_chapter_heading
    from novel_wash.llm_cleaner import parse_batch_decisions

    # 1. Deleting only [2,3) preserves untouched leading fullwidth spaces in AI edition, while collapsing in reader edition
    src_lines = ["\u3000\u3000甲"]
    ops = [EditOp(1, 2, 3, "del", "甲", "", "llm_tier_n", "test_del", "test")]
    out_lines, cleared = apply_and_verify(src_lines, ops)
    assert out_lines == ["\u3000\u3000"]
    assert cleared == {1}
    assert render_ai_edition(src_lines, ops) == "\u3000\u3000\n"
    assert render_reader_edition(src_lines, ops) == ""

    # 2. Top-level JSON arrays (including prose-prefixed arrays) must be rejected by _extract_json / parse_batch_decisions
    task = DirtyLineTask(
        line_num=2,
        orig_line="乙a",
        pre_peeled_line="乙a",
        peeled_to_orig_offsets=[0, 1],
        tagged_text="乙⟦1a⟧",
        candidates=[InlineCandidateSite(cid=1, orig_start=1, orig_end=2, char="a", tier="C", reason="junk")],
    )
    with pytest.raises(json.JSONDecodeError):
        parse_batch_decisions('[{"L2":{"del":[1],"keep":[]}}]', [task])
    with pytest.raises(json.JSONDecodeError):
        parse_batch_decisions('Result:\n[{"L2":{"del":[1],"keep":[]}}]', [task])

    # 3. Chapter headings: reject prose colon sentence, keep real nested & numbered chapter headings
    assert is_chapter_heading("第三章：他走了。") is False
    assert is_chapter_heading("第三章: 他走了。") is False
    assert is_chapter_heading("第三章 ：他走了。") is False
    assert is_chapter_heading("第318章: 第三百一十五章 观察结果。") is True
    assert is_chapter_heading("1.第1章 诈尸") is True
    assert is_chapter_heading("4359.第4359章 毁灭，天道弥补（全书完）") is True


def test_round5_extract_json_bracketed_prose_vs_real_arrays():
    import json
    from novel_wash.llm_cleaner import parse_batch_decisions

    task = DirtyLineTask(
        line_num=2,
        orig_line="乙a",
        pre_peeled_line="乙a",
        peeled_to_orig_offsets=[0, 1],
        tagged_text="乙⟦1a⟧",
        candidates=[InlineCandidateSite(cid=1, orig_start=1, orig_end=2, char="a", tier="C", reason="junk")],
    )
    # Bracketed non-JSON prose prefix before valid JSON object MUST be accepted
    del_map, missing = parse_batch_decisions('[Result]\n{"L2":{"del":[1],"keep":[]}}', [task])
    assert missing == []
    assert del_map == {2: {1}}

    # Real JSON arrays (direct, prose-wrapped, truncated, or empty) MUST still be rejected
    for bad_payload in (
        '[{"L2":{"del":[1],"keep":[]}}]',
        'Result:\n[{"L2":{"del":[1],"keep":[]}}]',
        '[{"L2":{"del":[1],"keep":[]}}',
        'Result:\n[]',
        'Result:\n[null,{"L2":{"del":[1],"keep":[]}}',
        '[1,{"L2":{"del":[1],"keep":[]}}',
        '[Result,{"L2":{"del":[1],"keep":[]}}',
    ):
        with pytest.raises(json.JSONDecodeError):
            parse_batch_decisions(bad_payload, [task])

    del_map2, missing2 = parse_batch_decisions('[Step 1] [Result]\n{"L2":{"del":[1],"keep":[]}}', [task])
    assert missing2 == [] and del_map2 == {2: {1}}


def test_round7_extract_json_bracket_depth_and_numbered_prose_prefix():
    import json
    from novel_wash.llm_cleaner import _extract_json

    for bad_payload in (
        'Result:\n[Result, [],{"L2":{"del":[1],"keep":[]}}',
        'Result:\n[[{"L2":{"del":[1],"keep":[]}}]]',
        'Result:\n[1], {"L2":{"del":[1],"keep":[]}}',
    ):
        with pytest.raises(json.JSONDecodeError):
            _extract_json(bad_payload)

    parsed = _extract_json('[1. Result]\n{"L2":{"del":[1],"keep":[]}}')
    assert parsed == {"L2": {"del": [1], "keep": []}}


def test_round8_extract_json_rejects_scalar_arrays_truncated_wrappers_and_trailing_arrays():
    import json
    from novel_wash.llm_cleaner import parse_batch_decisions

    task = DirtyLineTask(
        line_num=2,
        orig_line="乙a",
        pre_peeled_line="乙a",
        peeled_to_orig_offsets=[0, 1],
        tagged_text="乙⟦1a⟧",
        candidates=[InlineCandidateSite(cid=1, orig_start=1, orig_end=2, char="a", tier="C", reason="junk")],
    )
    for bad_payload in (
        '[1]\n{"L2":{"del":[1],"keep":[]}}',
        '{"result":{"L2":{"del":[1],"keep":[]}}',
        'Result:\n{"result":{"L2":{"del":[1],"keep":[]}}',
        '{"L2":{"del":[1],"keep":[]}}\n[1]',
        '{"L2":{"del":[1],"keep":[]}} extra [1]',
        '}\n{"L2":{"del":[1],"keep":[]}}',
        '[1, ]\n{"L2":{"del":[1],"keep":[]}}',
    ):
        with pytest.raises(json.JSONDecodeError):
            parse_batch_decisions(bad_payload, [task])

    for ok_payload in (
        '[Result]\n{"L2":{"del":[1],"keep":[]}}',
        '[1. Result]\n{"L2":{"del":[1],"keep":[]}}',
    ):
        del_map, missing = parse_batch_decisions(ok_payload, [task])
        assert del_map == {2: {1}} and missing == []


def test_round9_extract_json_strict_whitelist_envelopes():
    import json
    from novel_wash.llm_cleaner import parse_batch_decisions

    task = DirtyLineTask(
        line_num=2,
        orig_line="乙a",
        pre_peeled_line="乙a",
        peeled_to_orig_offsets=[0, 1],
        tagged_text="乙⟦1a⟧",
        candidates=[InlineCandidateSite(cid=1, orig_start=1, orig_end=2, char="a", tier="C", reason="junk")],
    )
    for bad_payload in (
        '[1 2]\n{"L2":{"del":[1],"keep":[]}}',
        'null\n{"L2":{"del":[1],"keep":[]}}',
        'true\n{"L2":{"del":[1],"keep":[]}}',
        '42\n{"L2":{"del":[1],"keep":[]}}',
        '[null]\n{"L2":{"del":[1],"keep":[]}}',
        '{"L2":{"del":[1],"keep":[]}}\n42',
        '{"L2":{"del":[1],"keep":[]}}\nnull',
    ):
        with pytest.raises(json.JSONDecodeError):
            parse_batch_decisions(bad_payload, [task])

    for ok_payload in (
        'Result:\n{"L2":{"del":[1],"keep":[]}}',
        '[Result]\n{"L2":{"del":[1],"keep":[]}}',
        '[1. Result]\n{"L2":{"del":[1],"keep":[]}}',
        '[Step 1] [Result]\n{"L2":{"del":[1],"keep":[]}}',
        'Result:\n```json\n{"L2":{"del":[1],"keep":[]}}\n```',
    ):
        del_map, missing = parse_batch_decisions(ok_payload, [task])
        assert del_map == {2: {1}} and missing == []


def test_quality_fix_poisoned_heading_verifier_and_json_constants():
    import json
    from novel_wash.llm_cleaner import _extract_json

    # 1. Poisoned heading allows title-tail deletions while protecting `第216章 ` prefix
    orig = "第216章 球在猎户座之下"
    peeled = "第216章 球在猎户座之下"
    c_prefix = InlineCandidateSite(1, 0, 1, "第", "C", "prefix")
    c_tail = InlineCandidateSite(2, 6, 7, "球", "N", "tail")
    task_dirty = DirtyLineTask(1, orig + "\u200c", peeled, list(range(len(peeled))), "", [c_prefix, c_tail])
    out = verify_line_decisions(task_dirty, primary_del_cids={1, 2}, primary_model="deepseek-v4.1-flash-1017")
    assert out.rejected_count == 1  # prefix char rejected!
    assert len(out.edits) == 1 and out.edits[0].before == "球"

    # 2. Unclosed fence, NaN, and [null:] rejected by _extract_json
    for bad in (
        '```json\n{"L2":{"del":[1],"keep":[]}}',
        '{"L2":{"del":[NaN],"keep":[]}}',
        '[null:]\n{"L2":{"del":[1],"keep":[]}}',
    ):
        with pytest.raises(json.JSONDecodeError):
            _extract_json(bad)



