"""Unified novel-wash pipeline & CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from .edit_log import (
    BookEditLog,
    EditOp,
    apply_and_verify,
    render_ai_edition,
    render_reader_edition,
)
from .encoding import DecodedFile, read_novel_file
from .epub_export import build_epub, infer_title_and_author
from .epub_fix import check_and_repair_epub
from .group_soup import detect_explicit_ads_only, detect_soup_and_ads
from .harmony_restore import restore_known_slots
from .inline_rules import WORD_INTERNAL_CENSOR_RE, analyze_inline_pollution, cleanup_post_edit_artifacts
from .llm_cleaner import run_llm_inline_cleaning
from .llm_client import LLMRouter, RouterContentPolicyError
from .policy_scan import scan_book_policy, scan_path_policy
from .pua_solver import solve_pua_cipher
from .triage import triage_decoded, triage_file


def _merge_non_overlapping(edits: list[EditOp]) -> list[EditOp]:
    """Merge edits across layers with strict approval precedence (`approved` beats `review`) and no overlaps."""
    by_line: dict[int, list[EditOp]] = {}
    for e in edits:
        by_line.setdefault(e.line, []).append(e)

    merged: list[EditOp] = []
    for ln in sorted(by_line):
        ops = by_line[ln]
        approved_clears = [e for e in ops if e.op == "clear_line" and e.status == "approved"]
        if approved_clears:
            merged.append(approved_clears[0])
            continue
        review_clears = [e for e in ops if e.op == "clear_line"]
        if review_clears and not any(e.status == "approved" for e in ops):
            merged.append(review_clears[0])
            continue
        span_ops = [e for e in ops if e.op != "clear_line"]
        # Approved edits always claim spans before review edits; longer spans win ties at same start.
        span_ops.sort(key=lambda e: (0 if e.status == "approved" else 1, e.start, -(e.end - e.start)))
        claimed: list[tuple[int, int]] = []
        for e in span_ops:
            if any(not (e.end <= st or e.start >= en) for st, en in claimed):
                continue
            merged.append(e)
            claimed.append((e.start, e.end))
    return merged


def _compose_pua_and_stage_b(
    orig_lines: list[str],
    pua_edits: list[EditOp],
    merged_stage_b: list[EditOp],
    review_queue: list[dict[str, Any]],
) -> list[EditOp]:
    """Compose Stage A (1-to-1 PUA subs) and Stage B edits cleanly onto original line coordinates."""
    if not pua_edits:
        return merged_stage_b

    pua_by_line: dict[int, list[EditOp]] = {}
    for e in pua_edits:
        pua_by_line.setdefault(e.line, []).append(e)

    b_by_line: dict[int, list[EditOp]] = {}
    for e in merged_stage_b:
        b_by_line.setdefault(e.line, []).append(e)

    composed: list[EditOp] = []
    for ln in sorted(set(pua_by_line) | set(b_by_line)):
        orig_line = orig_lines[ln - 1]
        p_ops = pua_by_line.get(ln, [])
        b_ops = b_by_line.get(ln, [])

        approved_clears = [e for e in b_ops if e.op == "clear_line" and e.status == "approved"]
        if approved_clears:
            c = approved_clears[0]
            composed.append(
                EditOp(
                    line=ln,
                    start=0,
                    end=len(orig_line),
                    op="clear_line",
                    before=orig_line,
                    after="",
                    tier=c.tier,
                    reason=c.reason,
                    source=c.source,
                    status="approved",
                    confidence=c.confidence,
                )
            )
            continue

        suppressed_pua: set[int] = set()
        approved_spans: list[tuple[int, int]] = []
        for e in b_ops:
            if e.status != "approved" or e.op == "clear_line":
                continue
            orig_before = orig_line[e.start : e.end]
            rebased_reason = f"{e.reason}+rebased_pua" if orig_before != e.before else e.reason
            composed.append(
                EditOp(
                    line=ln,
                    start=e.start,
                    end=e.end,
                    op=e.op,
                    before=orig_before,
                    after=e.after,
                    tier=e.tier,
                    reason=rebased_reason,
                    source=e.source,
                    status="approved",
                    confidence=e.confidence,
                )
            )
            approved_spans.append((e.start, e.end))
            for idx_p, p in enumerate(p_ops):
                if not (p.end <= e.start or p.start >= e.end):
                    suppressed_pua.add(idx_p)

        kept_pua_spans: list[tuple[int, int]] = []
        for idx_p, p in enumerate(p_ops):
            if idx_p not in suppressed_pua:
                composed.append(p)
                kept_pua_spans.append((p.start, p.end))

        blocked_spans = approved_spans + kept_pua_spans
        review_clears = [e for e in b_ops if e.op == "clear_line" and e.status == "review"]
        if review_clears:
            rc = review_clears[0]
            if not p_ops and not any(e.status == "approved" for e in b_ops):
                composed.append(rc)
            else:
                review_queue.append(
                    {
                        "line": ln,
                        "start": 0,
                        "end": len(orig_line),
                        "before_decoded": rc.before,
                        "reason": f"{rc.reason}_deferred_for_approved_overlap",
                        "original": orig_line,
                    }
                )
        for e in b_ops:
            if e.status != "review" or e.op == "clear_line":
                continue
            if any(not (e.end <= st or e.start >= en) for st, en in blocked_spans):
                review_queue.append(
                    {
                        "line": ln,
                        "start": e.start,
                        "end": e.end,
                        "before_decoded": e.before,
                        "reason": f"{e.reason}_deferred_for_pua_overlap",
                        "original": orig_line,
                    }
                )
                continue
            orig_before = orig_line[e.start : e.end]
            if orig_before == e.before:
                composed.append(e)

    return composed


def _safe_slug(path: Path) -> str:
    s = path.stem.strip()
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", s)
    return s[:80] or "book"


def wash_book(
    src_path: Path,
    *,
    project_root: Path,
    out_dir: Path,
    use_model: bool = True,
    primary_model: str | None = None,
    secondary_model: str | None = "USE_DEFAULT",
    group_override: str | None = None,
    force_full_rules: bool = False,
    render_epub: bool = True,
    restore_harmony_with_llm: bool = False,
    use_pools: bool = False,
    router: LLMRouter | None = None,
    config_path: Path | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_format: str | None = None,
) -> dict[str, Any]:
    """Wash a single novel file end-to-end and write edit-log, AI/Reader TXTs, and EPUB3."""
    decoded = read_novel_file(src_path)
    lines = decoded.lines
    slug = _safe_slug(src_path)
    cache_dir = project_root / "cache" / "pua_maps"
    ckpt_dir = project_root / "runs" / ".checkpoints"

    own_router = False
    if use_model and router is None:
        router = LLMRouter(
            project_root,
            config_path=config_path,
            base_url=base_url,
            api_key=api_key,
            api_format=api_format,
        )
        own_router = True

    eff_primary = primary_model or (router.default_primary if router is not None else "deepseek-v4.1-flash-1017")
    if secondary_model == "USE_DEFAULT":
        eff_secondary: str | None = router.default_secondary if router is not None else None
    elif secondary_model is None or str(secondary_model).strip().lower() in ("", "none", "off", "null"):
        eff_secondary = None
    else:
        eff_secondary = secondary_model

    cost_before = router.ledger.total_cost_cny() if router else 0.0
    all_review: list[dict[str, Any]] = []

    try:
        # Layer 1: PUA Cipher Solver (before other rules so decoded lines enter soup/inline passes!)
        pua_cand = (
            str(router.config.get("default_routing", {}).get("pua_solver") or "").strip()
            if router is not None
            else ""
        )
        pua_primary = (
            pua_cand
            if (router is not None and pua_cand and router.supports_model(pua_cand, group_override))
            else eff_primary
        )
        arb_cand = (
            str(router.config.get("default_routing", {}).get("arbitrator") or "").strip()
            if router is not None
            else ""
        )
        if eff_primary != pua_primary and (router is None or router.supports_model(eff_primary, group_override)):
            pua_secondary: str | None = eff_primary
        elif eff_secondary and eff_secondary != pua_primary and (router is None or router.supports_model(eff_secondary, group_override)):
            pua_secondary = eff_secondary
        elif router is not None and arb_cand and arb_cand != pua_primary and router.supports_model(arb_cand, group_override):
            pua_secondary = arb_cand
        else:
            pua_secondary = None

        def _pua_llm(sys_p: str, usr_p: str, model_name: str | None) -> str:
            assert router is not None
            chosen_m = model_name or pua_primary
            eff_grp = group_override if (group_override and router.supports_model(chosen_m, group_override)) else None
            txt, _ = router.chat(chosen_m, sys_p, usr_p, group=eff_grp)
            return txt

        pua_res = solve_pua_cipher(
            lines,
            cache_dir,
            llm_call=_pua_llm if use_model else None,
            primary_model=pua_primary,
            secondary_model=pua_secondary,
        )
        if pua_res.edits:
            working_lines, _ = apply_and_verify(lines, pua_res.edits)
        else:
            working_lines = list(lines)

        work_decoded = (
            DecodedFile(
                path=decoded.path,
                sha256=decoded.sha256,
                encoding=decoded.encoding,
                text="\n".join(working_lines),
                lines=working_lines,
                line_ending=decoded.line_ending,
                is_mixed=decoded.is_mixed,
            )
            if pua_res.edits
            else decoded
        )
        post_pua_triage = triage_decoded(work_decoded, file_label=str(src_path))
        is_clean_safe = (not force_full_rules) and (
            post_pua_triage.grade in ("clean", "encoding_only")
            or (
                pua_res.pua_count > 50
                and post_pua_triage.circled_lines < 5
                and post_pua_triage.hard_cjk_lines < 5
                and post_pua_triage.yueyi_block_lines < 5
            )
        )

        clusters: list[str] = []
        cleared_lines: set[int] = set()
        proper_nouns_out: list[str] = []
        rule_pre_count = 0
        llm_tasks_count = 0
        llm_stats: dict[str, Any] = {
            "primary_tasks": 0,
            "tier_c_crosscheck_tasks": 0,
            "rejected_ops": 0,
            "primary_unresolved": 0,
            "secondary_unresolved": 0,
            "incomplete": False,
            "clean_safe_mode": is_clean_safe,
        }
        stage_b_edits: list[EditOp] = []

        if is_clean_safe:
            # Conservative mode: preserve clean prose/kaomoji/percentages; strip explicit site ads & word-internal ^/` only
            ad_edits = detect_explicit_ads_only(working_lines)
            stage_b_edits.extend(ad_edits)
            cleared_lines = {e.line for e in ad_edits if e.op == "clear_line"}
            censor_edits: list[EditOp] = []
            for idx, wline in enumerate(working_lines, start=1):
                if idx in cleared_lines:
                    continue
                for m in WORD_INTERNAL_CENSOR_RE.finditer(wline):
                    censor_edits.append(
                        EditOp(
                            line=idx,
                            start=m.start(),
                            end=m.end(),
                            op="del",
                            before=m.group(),
                            after="",
                            tier="rule_sym",
                            reason="word_internal_censor_symbol",
                            source="rule:clean_safe",
                            status="approved",
                        )
                    )
            stage_b_edits.extend(censor_edits)
            rule_pre_count = len(censor_edits)
            harmony_skip = ({e.line for e in pua_res.edits} if pua_res.edits else set()) | cleared_lines | {e.line for e in censor_edits}
        else:
            # Layer 2: Standalone group-number soup & site banners
            soup_edits, soup_review, clusters, soup_nums = detect_soup_and_ads(working_lines)
            stage_b_edits.extend(soup_edits)
            all_review.extend(soup_review)

            # Layer 3: Inline anti-censor & watermark pre-peeling + self-learning + candidate tagging
            inline_res = analyze_inline_pollution(working_lines, soup_edits, soup_nums)
            proper_nouns_out = inline_res.proper_nouns
            rule_pre_count = len(inline_res.pre_edits)
            llm_tasks_count = len(inline_res.tasks)
            stage_b_edits.extend(inline_res.pre_edits)
            peeled_lines, _ = apply_and_verify(working_lines, soup_edits + inline_res.pre_edits)

            # Layer 4 & 5: Batched LLM numbered-candidate cleaning + point-wise verify
            if use_model and inline_res.tasks and router is not None:
                ckpt_path = ckpt_dir / f"{slug}_{decoded.sha256[:12]}.json"
                llm_edits, llm_review, llm_stats = run_llm_inline_cleaning(
                    inline_res,
                    peeled_lines,
                    router,
                    primary_model=eff_primary,
                    secondary_model=eff_secondary,
                    source_sha256=decoded.sha256,
                    checkpoint_path=ckpt_path,
                    group_override=group_override,
                    use_pools=use_pools,
                )
                llm_stats["clean_safe_mode"] = False
                stage_b_edits.extend(llm_edits)
                all_review.extend(llm_review)

            pua_lines = {e.line for e in pua_res.edits}
            cleared_lines = {e.line for e in soup_edits if e.op == "clear_line"}
            harmony_skip = pua_lines | cleared_lines | {t.line_num for t in inline_res.tasks}

        merged_stage_b = _merge_non_overlapping(stage_b_edits)
        expected_after_b, _ = apply_and_verify(working_lines, merged_stage_b, include_review=False)
        composed_ab = _compose_pua_and_stage_b(lines, pua_res.edits, merged_stage_b, all_review)
        replayed_after_b, _ = apply_and_verify(lines, composed_ab, include_review=False)
        if replayed_after_b != expected_after_b:
            raise AssertionError("Cross-layer PUA + Stage B replay parity mismatch")
        apply_and_verify(lines, composed_ab, include_review=True)

        # Layer 3b: Censored `**` / `〇〇` slot restoration (only on original non-PUA lines to keep offsets exact)
        harmony_model = (
            str(router.config.get("default_routing", {}).get("harmony_filler") or "").strip()
            if router is not None
            else ""
        )
        harmony_primary = (
            harmony_model
            if (router is not None and harmony_model and router.supports_model(harmony_model, group_override))
            else eff_primary
        )

        def _harmony_llm(sys_p: str, usr_p: str) -> str:
            assert router is not None
            if use_pools and router.config.get("pools"):
                try:
                    txt, _ = router.chat_pool("primary", sys_p, usr_p)
                    return txt
                except RouterContentPolicyError:
                    return "{}"
            txt, _ = router.chat(harmony_primary, sys_p, usr_p, group=group_override)
            return txt

        harmony_edits = restore_known_slots(
            lines,
            skip_lines=harmony_skip,
            llm_fill=_harmony_llm if (use_model and restore_harmony_with_llm and router is not None) else None,
        )
        merged_pre_cleanup = _merge_non_overlapping(composed_ab + harmony_edits)
        post_cleanup_edits = cleanup_post_edit_artifacts(lines, merged_pre_cleanup)
        merged_edits = _merge_non_overlapping(merged_pre_cleanup + post_cleanup_edits)

        # Validate replay & render both editions from the single edit-log
        ai_text = render_ai_edition(lines, merged_edits, include_review=False, line_ending="\n")
        reader_text = render_reader_edition(lines, merged_edits, include_review=False)

        ai_dir = out_dir / "ai"
        reader_dir = out_dir / "reader"
        epub_dir = out_dir / "epub"
        log_dir = out_dir / "edit_logs"
        ai_dir.mkdir(parents=True, exist_ok=True)
        reader_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        ai_path = ai_dir / f"{slug}.ai.txt"
        reader_path = reader_dir / f"{slug}.reader.txt"
        log_path = log_dir / f"{slug}.edit_log.json"
        epub_candidate = epub_dir / f"{slug}.epub"
        src_resolved = src_path.resolve()
        for target_p in (ai_path, reader_path, log_path, epub_candidate):
            if target_p.resolve() == src_resolved:
                raise ValueError(f"Refusing to overwrite source path: {src_path}")

        ai_path.write_text(ai_text, encoding="utf-8")
        reader_path.write_text(reader_text, encoding="utf-8")

        epub_path_str: str | None = None
        chapters_count = 0
        if render_epub and reader_text.strip():
            title, author = infer_title_and_author(src_path)
            chapters_count = build_epub(reader_text, epub_candidate, title=title, author=author)
            epub_path_str = str(epub_candidate)

        cost_after = router.ledger.total_cost_cny() if router else 0.0
        book_cost = round(cost_after - cost_before, 6)
        policy_res = scan_book_policy(reader_text.splitlines())

        stats = {
            "encoding_detected": decoded.encoding,
            "is_mixed_encoding": decoded.is_mixed,
            "lines_total": len(lines),
            "pua_total": pua_res.pua_count,
            "pua_unique": pua_res.unique_codepoints,
            "pua_jieba_before": pua_res.jieba_ratio_before,
            "pua_jieba_after": pua_res.jieba_ratio_after,
            "soup_cleared_lines": len(cleared_lines),
            "soup_clusters": clusters[:10],
            "rule_pre_edits": rule_pre_count,
            "harmony_edits": len(harmony_edits),
            "llm_tasks": llm_tasks_count,
            "llm_stats": llm_stats,
            "total_approved_edits": sum(1 for e in merged_edits if e.status == "approved"),
            "total_review_edits": sum(1 for e in merged_edits if e.status == "review"),
            "review_queue_count": len(all_review),
            "epub_chapters": chapters_count,
            "cost_cny": book_cost,
            "policy_scan": policy_res,
        }

        edit_log = BookEditLog(
            source_file=str(src_path),
            source_sha256=decoded.sha256,
            encoding=decoded.encoding,
            line_count=len(lines),
            edits=merged_edits,
            review_queue=all_review,
            proper_nouns=proper_nouns_out,
            pua_map_info={
                "unique": pua_res.unique_codepoints,
                "cache_hit": pua_res.cache_hit,
                "uncertain": pua_res.uncertain_codepoints,
            },
            stats=stats,
            cost_summary={"book_cost_cny": book_cost},
        )
        edit_log.save(log_path)

        return {
            "source_file": str(src_path),
            "ai_path": str(ai_path),
            "reader_path": str(reader_path),
            "epub_path": epub_path_str,
            "log_path": str(log_path),
            "stats": stats,
        }
    finally:
        if own_router and router is not None:
            router.close()


def _run_doctor(
    *,
    project_root: Path,
    config_path: Path | None,
    base_url: str | None,
    api_key: str | None,
    api_format: str | None,
    primary: str | None,
    secondary: str | None,
    provider: str | None,
    dry_run: bool,
) -> int:
    router = LLMRouter(
        project_root,
        config_path=config_path,
        base_url=base_url,
        api_key=api_key,
        api_format=api_format,
    )
    try:
        p_mod = primary or router.resolve_default_primary(provider)
        if secondary is None or str(secondary) == "USE_DEFAULT":
            s_mod: str | None = None if provider is not None else router.default_secondary
        elif secondary.strip().lower() in ("", "none", "off", "null"):
            s_mod = None
        else:
            s_mod = secondary

        targets = [("primary", p_mod)]
        if s_mod and s_mod != p_mod:
            targets.append(("secondary", s_mod))

        checks: list[dict[str, Any]] = []
        all_ok = True
        for role, mod_name in targets:
            info = router.describe_route(mod_name, provider)
            entry: dict[str, Any] = {"role": role, **info}
            if not info.get("eligible"):
                entry["ok"] = False
                entry["error"] = info.get("error") or "No eligible route."
                all_ok = False
            elif dry_run:
                entry["ok"] = True
            else:
                try:
                    txt, metrics = router.chat(
                        mod_name,
                        "Return strict JSON.",
                        'Reply with {"ok":true}',
                        group=provider,
                        max_tokens=64,
                        max_retries=2,
                    )
                    if not txt.strip():
                        entry["ok"] = False
                        entry["error"] = "Probe returned empty response."
                        all_ok = False
                    elif metrics.finish_reason == "length":
                        entry["ok"] = False
                        entry["error"] = "Probe response truncated (finish_reason=length)."
                        all_ok = False
                    else:
                        entry["ok"] = True
                    entry["latency_s"] = metrics.latency_s
                    entry["finish_reason"] = metrics.finish_reason
                    entry["usage"] = {
                        "prompt_tokens": metrics.prompt_tokens,
                        "cached_tokens": metrics.cached_tokens,
                        "completion_tokens": metrics.completion_tokens,
                        "reasoning_tokens": metrics.reasoning_tokens,
                    }
                    entry["sample_reply"] = txt[:80]
                except Exception as exc:
                    entry["ok"] = False
                    entry["error"] = str(exc)
                    all_ok = False
            checks.append(entry)
        print(json.dumps({"ok": all_ok, "checks": checks}, ensure_ascii=False, indent=2))
        return 0 if all_ok else 1
    finally:
        router.close()


def _add_llm_connection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--config", type=Path, default=None, help="Optional models JSON path (defaults to config/models.json if present)")
    p.add_argument("--base-url", default=None, help="Override API Base URL")
    p.add_argument("--api-key", default=None, help="Override API Key")
    p.add_argument("--api-format", choices=["openai", "anthropic", "gemini"], default=None, help="Override API format")
    p.add_argument("--primary", default=None, help="Primary model ID (defaults to env or config default_routing)")
    p.add_argument("--secondary", default="USE_DEFAULT", help="Secondary Tier-C model ID (pass 'none' to disable)")
    p.add_argument("--provider", "--group", dest="group", default=None, help="Override configured provider name")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="novel-wash")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tri = sub.add_parser("triage", help="Scan TXT files for encoding & pollution triage")
    p_tri.add_argument("paths", nargs="+", type=Path)

    p_wash = sub.add_parser("wash", help="Wash one or more TXT novels")
    p_wash.add_argument("paths", nargs="+", type=Path)
    p_wash.add_argument("--out", type=Path, default=Path("output"))
    p_wash.add_argument("--no-model", action="store_true")
    _add_llm_connection_args(p_wash)
    p_wash.add_argument("--force-full-rules", action="store_true")
    p_wash.add_argument("--no-epub", action="store_true")
    p_wash.add_argument("--harmony-llm", action="store_true")
    p_wash.add_argument("--use-pools", action="store_true")

    p_doc = sub.add_parser("doctor", help="Check API connectivity, format normalization, and routing")
    _add_llm_connection_args(p_doc)
    p_doc.set_defaults(secondary=None)
    p_doc.add_argument("--dry-run", action="store_true", help="Inspect resolved URL/format/key presence without network call")

    p_ep = sub.add_parser("epub-check", help="Validate and repair malformed EPUB archives")
    p_ep.add_argument("paths", nargs="+", type=Path)
    p_ep.add_argument("--out", type=Path, default=Path("output/epub_fixed"))
    p_ep.add_argument("--strip-telegram", action="store_true")

    p_pol = sub.add_parser("scan-policy", help="Read-only political & fadeout scan across TXT/EPUB files")
    p_pol.add_argument("paths", nargs="+", type=Path)
    p_pol.add_argument("--terms", type=Path, default=None)
    p_pol.add_argument("--no-fadeout", action="store_true")
    p_pol.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "triage":
        res = [triage_file(p).to_dict() for p in args.paths]
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "doctor":
        return _run_doctor(
            project_root=args.project_root,
            config_path=args.config,
            base_url=args.base_url,
            api_key=args.api_key,
            api_format=args.api_format,
            primary=args.primary,
            secondary=args.secondary,
            provider=args.group,
            dry_run=args.dry_run,
        )
    if args.cmd == "scan-policy":
        if args.out is not None:
            out_resolved = args.out.resolve()
            input_resolved = {p.resolve() for p in args.paths}
            same_inode = args.out.exists() and any(
                p.exists() and os.path.samefile(args.out, p) for p in args.paths
            )
            if out_resolved in input_resolved or same_inode:
                print(
                    f"Error: --out path '{args.out}' cannot match an input path.",
                    file=sys.stderr,
                )
                return 2
        res = [
            scan_path_policy(p, terms_path=args.terms, include_fadeout_hints=not args.no_fadeout)
            for p in args.paths
        ]
        payload = json.dumps(res, ensure_ascii=False, indent=2)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0
    if args.cmd == "epub-check":
        res = [
            check_and_repair_epub(p, args.out, strip_telegram_banner=args.strip_telegram).to_dict()
            for p in args.paths
        ]
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "wash":
        router = (
            None
            if args.no_model
            else LLMRouter(
                args.project_root,
                config_path=args.config,
                base_url=args.base_url,
                api_key=args.api_key,
                api_format=args.api_format,
            )
        )
        try:
            out = []
            for p in args.paths:
                out.append(
                    wash_book(
                        p,
                        project_root=args.project_root,
                        out_dir=args.out,
                        use_model=not args.no_model,
                        primary_model=args.primary,
                        secondary_model=args.secondary,
                        group_override=args.group,
                        force_full_rules=args.force_full_rules,
                        render_epub=not args.no_epub,
                        restore_harmony_with_llm=args.harmony_llm,
                        use_pools=args.use_pools,
                        router=router,
                    )
                )
            print(json.dumps(out, ensure_ascii=False, indent=2))
        finally:
            if router is not None:
                router.close()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
