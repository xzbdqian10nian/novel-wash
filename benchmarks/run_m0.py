"""M0 4-model live benchmark on real local gold samples (`gold_set.jsonl`)."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import time
from pathlib import Path
from typing import Any

from novel_wash.edit_log import EditOp, apply_edits_to_line
from novel_wash.encoding import read_novel_file
from novel_wash.group_soup import detect_soup_and_ads
from novel_wash.inline_rules import (
    CJK_RE,
    DirtyLineTask,
    _line_corruption_evidence,
    _render_peeled_with_map,
    _soft_protected_indices,
    _window_cjk_indices,
    analyze_inline_pollution,
)
from novel_wash.llm_cleaner import build_system_prompt, format_batch_user_prompt, parse_batch_decisions
from novel_wash.llm_client import LLMRouter
from novel_wash.shadow import build_shadow
from novel_wash.verifier import verify_line_decisions


def _deleted_orig_indices(orig: str, gold_clean: str) -> set[int]:
    """Compute character indices in `orig` that were removed to produce `gold_clean`.

    Uses contextual DP subsequence alignment when `gold_clean` is an exact subsequence of `orig`
    so repeated punctuation (`!`, `.`) never misaligns (as happens with difflib on `!⑤...` vs `...`).
    """
    m, n = len(orig), len(gold_clean)
    if n == 0:
        return set(range(m))
    # Fast subsequence check
    pos_map: list[int] | None = None
    # Maximize contiguous matched runs (reward adjacent matches + late alignment for repeated punct)
    # Only runs on short benchmark lines (< 500 chars), so O(m*n) DP is <1ms.
    it = iter(enumerate(orig))
    is_subseq = True
    j = 0
    for ch in gold_clean:
        for _, och in it:
            if och == ch:
                break
        else:
            is_subseq = False
            break
    if is_subseq and m * n <= 200_000:
        # DP backwards: suffix feasibility + score contiguous matches higher
        # dp_next matches gold_clean[j] at earliest index? Better: standard LCS where length == n, tie-break by contig score
        neg = -10**9
        # match_score[i][j] = best score matching gold_clean[j:] inside orig[i:] IF suffix matches
        # First compute can_match[i][j]
        can = [[False] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            can[i][n] = True
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                can[i][j] = can[i + 1][j] or (orig[i] == gold_clean[j] and can[i + 1][j + 1])
        if can[0][0]:
            kept: set[int] = set()
            i = 0
            for j in range(n):
                best_i = -1
                best_score = (-1, -1)
                for k in range(i, m):
                    if orig[k] == gold_clean[j] and can[k + 1][j + 1]:
                        # Prefer keeping adjacent to previous kept char, or adjacent to next char
                        contig_prev = 1 if (kept and k - 1 in kept) else 0
                        contig_next = 1 if (j + 1 < n and k + 1 < m and orig[k + 1] == gold_clean[j + 1]) else 0
                        score = (contig_prev + contig_next, k if gold_clean[j] in "!！" else -k)
                        if score > best_score:
                            best_score = score
                            best_i = k
                assert best_i >= 0
                kept.add(best_i)
                i = best_i + 1
            return set(range(m)) - kept

    sm = difflib.SequenceMatcher(None, orig, gold_clean, autojunk=False)
    deleted: set[int] = set()
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            deleted.update(range(i1, i2))
    return deleted


def _prepare_gold_tasks(
    records: list[dict[str, Any]],
    book_files: dict[str, Path],
) -> tuple[dict[str, dict[str, Any]], list[DirtyLineTask], dict[str, list[str]], dict[str, set[int]], dict[str, set[int]]]:
    """Run deterministic layers on manifest-configured books to produce exact candidate tasks for gold records."""

    by_book: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_book.setdefault(rec["book"], []).append(rec)

    book_contexts: dict[str, dict[str, Any]] = {}
    all_tasks: list[DirtyLineTask] = []
    proper_by_book: dict[str, list[str]] = {}
    gold_del_idx: dict[str, set[int]] = {}
    soup_cleared_by_book: dict[str, set[int]] = {}

    for book, recs in by_book.items():
        if book not in book_files:
            raise SystemExit(f"Gold dataset references book key {book!r} missing from manifest.")
        book_path = book_files[book]
        if not book_path.is_file():
            raise SystemExit(f"Manifest source file for {book!r} does not exist: {book_path}")
        decoded = read_novel_file(book_path)
        n_lines = len(decoded.lines)
        for rec in recs:
            ln = int(rec["line_num"])
            if not (1 <= ln <= n_lines):
                raise SystemExit(
                    f"Record {rec.get('id', '?')}: line_num {ln} out of range for {book_path.name} (1..{n_lines})."
                )
            if decoded.lines[ln - 1] != rec["orig"]:
                raise SystemExit(
                    f"Record {rec.get('id', '?')}: source line mismatch at {book}:L{ln}; wrong edition/file."
                )

        soup_edits, _, _, soup_nums = detect_soup_and_ads(decoded.lines)
        soup_cleared_by_book[book] = {e.line for e in soup_edits if e.op == "clear_line"}
        inline_res = analyze_inline_pollution(decoded.lines, soup_edits, soup_nums)
        proper_by_book[book] = inline_res.proper_nouns
        task_map = {t.line_num: t for t in inline_res.tasks}
        pre_by_line: dict[int, list[EditOp]] = {}
        for e in soup_edits + inline_res.pre_edits:
            pre_by_line.setdefault(e.line, []).append(e)

        book_contexts[book] = {
            "lines": decoded.lines,
            "pre_by_line": pre_by_line,
            "task_map": task_map,
        }

        for rec in recs:
            ln = int(rec["line_num"])
            gold_del_idx[rec["id"]] = _deleted_orig_indices(rec["orig"], rec["gold_clean"])
            t = task_map.get(ln)
            if t is not None and rec["kind"] in ("inline", "yueyi_block"):
                all_tasks.append(t)

    return book_contexts, all_tasks, proper_by_book, gold_del_idx, soup_cleared_by_book


def _load_book_map(manifest_path: Path | None) -> dict[str, Path]:
    path = manifest_path or (
        Path(os.environ["NOVEL_WASH_CORPUS_MANIFEST"]) if "NOVEL_WASH_CORPUS_MANIFEST" in os.environ else None
    )
    if path is None or not path.is_file():
        raise SystemExit(
            "Provide --manifest <corpus.json> (see config/corpus.example.json) or set NOVEL_WASH_CORPUS_MANIFEST."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["key"]): Path(item["path"]) for item in raw.get("books", []) if "key" in item and "path" in item}


def main(argv: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run M0 4-model gold benchmark using a JSON corpus manifest.")
    parser.add_argument("--manifest", type=Path, default=None, help="JSON corpus manifest (see config/corpus.example.json)")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path(__file__).resolve().parent / "gold_set.jsonl",
        help="Gold benchmark JSONL path (default: benchmarks/gold_set.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "output" / "benchmarks",
        help="Directory for generated m0_results.json and m0_report.md (default: output/benchmarks/)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model IDs to benchmark (defaults to configured models)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional provider override for all benchmarked models",
    )
    args = parser.parse_args(argv)
    book_files = _load_book_map(args.manifest)

    gold_path = args.gold
    if not gold_path.is_file():
        raise SystemExit(f"Gold JSONL not found: {gold_path}. Build it first via benchmarks/build_gold.py.")
    records = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    book_contexts, _, proper_by_book, gold_del_idx, soup_cleared_by_book = _prepare_gold_tasks(records, book_files)

    # Check deterministic code-layer performance first
    code_soup_hits = 0
    code_soup_total = 0
    hard_neg_pass = 0
    hard_neg_total = 0
    eval_items: list[tuple[dict[str, Any], DirtyLineTask]] = []

    for rec in records:
        book = rec["book"]
        ln = rec["line_num"]
        ctx = book_contexts[book]
        pre_ops = ctx["pre_by_line"].get(ln, [])
        is_cleared = ln in soup_cleared_by_book[book]

        if rec["kind"] == "soup_clear":
            code_soup_total += 1
            if is_cleared:
                code_soup_hits += 1
        elif rec["kind"] == "hard_neg":
            hard_neg_total += 1
            if not is_cleared and not pre_ops and ln not in ctx["task_map"]:
                hard_neg_pass += 1
            elif not is_cleared and not pre_ops:
                # Even if tagged (unlikely), will check after LLM too
                eval_items.append((rec, ctx["task_map"][ln]))
        else:
            t = ctx["task_map"].get(ln)
            if t is not None:
                eval_items.append((rec, t))

    print(
        f"Deterministic Pre-Check: soup_clear={code_soup_hits}/{code_soup_total}, "
        f"hard_neg_untouched={hard_neg_pass}/{hard_neg_total}, llm_eval_lines={len(eval_items)}"
    )

    # Remap line_num -> unique synthetic benchmark line_num (1..N) so distinct books never collide in JSON keys
    synth_tasks: list[DirtyLineTask] = []
    synth_to_rec: dict[int, tuple[dict[str, Any], DirtyLineTask]] = {}
    synth_peeled_lines: list[str] = []
    noun_pool: list[str] = []
    for book_key in sorted(proper_by_book):
        noun_pool.extend(proper_by_book[book_key][:25])
    combined_nouns = sorted(set(noun_pool))[:60]
    sys_prompt = build_system_prompt(combined_nouns)

    for idx, (rec, orig_t) in enumerate(eval_items, start=1):
        synth_t = DirtyLineTask(
            line_num=idx,
            orig_line=orig_t.orig_line,
            pre_peeled_line=orig_t.pre_peeled_line,
            peeled_to_orig_offsets=orig_t.peeled_to_orig_offsets,
            tagged_text=orig_t.tagged_text,
            candidates=orig_t.candidates,
        )
        synth_tasks.append(synth_t)
        synth_to_rec[idx] = (rec, orig_t)
        synth_peeled_lines.append(orig_t.pre_peeled_line)

    router = LLMRouter(project_root)
    if args.models:
        chosen_models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        enabled_models = [
            m
            for m, mc in router.config.get("models", {}).items()
            if isinstance(mc, dict) and mc.get("enabled", True) and router.supports_model(m, args.provider)
        ]
        chosen_models = enabled_models[:4] or [router.resolve_default_primary(args.provider)]

    # Preflight every chosen model + provider before running any benchmark batches
    for m_chk in chosen_models:
        r_info = router.describe_route(m_chk, args.provider)
        if not r_info.get("eligible"):
            raise SystemExit(
                f"Invalid benchmark route {m_chk!r} (provider={args.provider or r_info.get('provider')}): "
                f"{r_info.get('error')}"
            )

    models_to_test = [(m, args.provider) for m in chosen_models]
    model_results: dict[str, Any] = {}
    raw_del_by_model: dict[str, dict[int, set[int]]] = {}

    try:
        for model_name, pref_group in models_to_test:
            print(f"\n=== Benchmarking {model_name} (provider={pref_group or 'default'}) ===")
            c_before = router.ledger.total_cost_cny()
            t0 = time.perf_counter()
            del_decisions: dict[int, set[int]] = {}
            json_ok_batches = 0
            total_batches = 0
            failed_batches = 0
            unparsed_batches = 0
            batch_size = 12

            for b0 in range(0, len(synth_tasks), batch_size):
                batch = synth_tasks[b0 : b0 + batch_size]
                total_batches += 1
                uprompt = format_batch_user_prompt(batch, synth_peeled_lines)
                try:
                    raw, m = router.chat(model_name, sys_prompt, uprompt, group=pref_group)
                    parsed, missing = parse_batch_decisions(raw, batch)
                    if not missing:
                        json_ok_batches += 1
                    elif missing:
                        sub_sz = max(4, (len(missing) + 1) // 2) if len(missing) > 4 else len(missing)
                        for s0 in range(0, len(missing), sub_sz):
                            sub_b = missing[s0 : s0 + sub_sz]
                            raw2, _ = router.chat(
                                model_name, sys_prompt, format_batch_user_prompt(sub_b, synth_peeled_lines), group=pref_group
                            )
                            p2, _ = parse_batch_decisions(raw2, sub_b)
                            parsed.update(p2)
                    if not parsed and batch:
                        unparsed_batches += 1
                        print(f"  Batch {total_batches} produced 0 usable decisions on {model_name}")
                    else:
                        del_decisions.update(parsed)
                except Exception as exc:
                    failed_batches += 1
                    print(f"  Batch {total_batches} err on {model_name}: {exc}")

            if total_batches > 0 and (failed_batches + unparsed_batches) == total_batches:
                raise SystemExit(
                    f"Benchmark aborted: all {total_batches} batches failed or returned unusable responses for {model_name}."
                )

            elapsed = round(time.perf_counter() - t0, 2)
            cost = round(router.ledger.total_cost_cny() - c_before, 6)
            raw_del_by_model[model_name] = del_decisions

            tp = fp = fn = tn = 0
            fp_examples: list[str] = []
            fn_examples: list[str] = []
            tier_c_fp = tier_c_tp = 0

            for s_idx, (rec, orig_t) in synth_to_rec.items():
                pred_del = del_decisions.get(s_idx, set())
                g_del = gold_del_idx[rec["id"]]
                for c in orig_t.candidates:
                    should_del = c.orig_start in g_del
                    did_del = c.cid in pred_del
                    if should_del and did_del:
                        tp += 1
                        if c.tier == "C":
                            tier_c_tp += 1
                    elif (not should_del) and did_del:
                        fp += 1
                        if c.tier == "C":
                            tier_c_fp += 1
                        fp_examples.append(f"{rec['id']} cid={c.cid}('{c.char}', tier={c.tier})")
                    elif should_del and (not did_del):
                        fn += 1
                        fn_examples.append(f"{rec['id']} cid={c.cid}('{c.char}', tier={c.tier})")
                    else:
                        tn += 1

            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec_rate = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec_rate / (prec + rec_rate) if (prec + rec_rate) else 0.0
            fp_rate = fp / (fp + tn) if (fp + tn) else 0.0

            model_results[model_name] = {
                "preferred_group": pref_group,
                "elapsed_s": elapsed,
                "cost_cny": cost,
                "json_first_pass_batches": f"{json_ok_batches}/{total_batches}",
                "failed_batches": failed_batches,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tier_c_tp": tier_c_tp,
                "tier_c_fp": tier_c_fp,
                "precision": round(prec, 4),
                "recall": round(rec_rate, 4),
                "f1": round(f1, 4),
                "false_del_rate": round(fp_rate, 4),
                "fp_examples": fp_examples[:15],
                "fn_examples": fn_examples[:15],
            }
            print(
                f"  Result: P={prec:.3f} R={rec_rate:.3f} F1={f1:.3f} | "
                f"FP={fp} (Tier-C FP={tier_c_fp}) FN={fn} | cost=¥{cost:.6f} time={elapsed}s"
            )
            if fp_examples:
                print("  FP samples:", fp_examples[:8])

        # Evaluate cross-family 2-model Tier-C consensus combinations present in the current run
        canonical_pairs = [
            ("deepseek-v4.1-flash-1017", "glm-5.3-flash"),
            ("deepseek-v4.1-flash-1017", "minimax-m3.1-flash"),
            ("deepseek-v4.1-flash-1017", "hy4-preview"),
            ("hy4-preview", "glm-5.3-flash"),
            ("hy4-preview", "minimax-m3.1-flash"),
            ("hy4-preview", "deepseek-v4.1-flash-1017"),
            ("minimax-m3.1-flash", "glm-5.3-flash"),
            ("minimax-m3.1-flash", "deepseek-v4.1-flash-1017"),
        ]
        pairs_to_test = [
            (p_name, s_name)
            for p_name, s_name in canonical_pairs
            if p_name in raw_del_by_model and s_name in raw_del_by_model
        ]
        pair_results: dict[str, Any] = {}
        for primary_name, sec_name in pairs_to_test:
            tp = fp = fn = tn = review_cnt = 0
            fp_samples: list[str] = []
            for s_idx, (rec, orig_t) in synth_to_rec.items():
                p_del = raw_del_by_model[primary_name].get(s_idx, set())
                s_del = raw_del_by_model[sec_name].get(s_idx, set())
                outcome = verify_line_decisions(
                    orig_t,
                    p_del,
                    primary_name,
                    secondary_del_cids=s_del,
                    secondary_model=sec_name,
                    proper_nouns=set(combined_nouns),
                )
                approved_starts = {e.start for e in outcome.edits if e.status == "approved"}
                review_starts = {e.start for e in outcome.edits if e.status == "review"}
                review_cnt += len(review_starts)
                g_del = gold_del_idx[rec["id"]]
                for c in orig_t.candidates:
                    should_del = c.orig_start in g_del
                    did_del = c.orig_start in approved_starts
                    if should_del and did_del:
                        tp += 1
                    elif (not should_del) and did_del:
                        fp += 1
                        fp_samples.append(f"{rec['id']}('{c.char}', tier={c.tier})")
                    elif should_del and (not did_del):
                        fn += 1
                    else:
                        tn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec_rate = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec_rate / (prec + rec_rate) if (prec + rec_rate) else 0.0
            pair_results[f"{primary_name}+{sec_name}"] = {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn, "tier_c_review": review_cnt,
                "precision": round(prec, 4), "recall": round(rec_rate, 4), "f1": round(f1, 4),
                "fp_samples": fp_samples[:10],
            }

        summary = {
            "code_layer": {
                "soup_clear": f"{code_soup_hits}/{code_soup_total}",
                "hard_neg_untouched": f"{hard_neg_pass}/{hard_neg_total}",
            },
            "single_models": model_results,
            "consensus_pairs": pair_results,
            "ledger": router.ledger.summary(),
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_json = args.out_dir / "m0_results.json"
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Write readable Markdown report
        md_lines = [
            "# M0 4 模型金标盲测对比报告\n",
            f"- 纯代码预检：独立群号汤清零 `{code_soup_hits}/{code_soup_total}`，干净难负例零误伤 `{hard_neg_pass}/{hard_neg_total}`，进入模型评测共 `{len(synth_tasks)}` 行。\n",
            "## 单模型表现（带 ⟦编号+字⟧ 受限二分类协议）\n",
            "| 模型 | 首选分组 | 精确率 (P) | 召回率 (R) | F1 | 正常字误删数 (FP / Tier-C FP) | 漏删 (FN) | JSON 首轮全过 | 耗时 | 实测费用 (CNY) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mname, r in model_results.items():
            md_lines.append(
                f"| `{mname}` | `{r['preferred_group']}` | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
                f"{r['fp']} ({r['tier_c_fp']}) | {r['fn']} | {r['json_first_pass_batches']} | {r['elapsed_s']}s | ¥{r['cost_cny']:.6f} |"
            )
        md_lines.extend([
            "\n## 跨家族 Tier-C 双审组合效果（经 `verify()` 逐点门禁）\n",
            "| 主力 + 辅助模型 | 精确率 (P) | 召回率 (R) | F1 | 最终误删 (FP) | 拦截进 review 队列 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for pname, pr in pair_results.items():
            md_lines.append(
                f"| `{pname}` | {pr['precision']:.4f} | {pr['recall']:.4f} | {pr['f1']:.4f} | {pr['fp']} | {pr['tier_c_review']} |"
            )
        md_lines.append("\n## 详细费用账本\n```json\n" + json.dumps(router.ledger.summary(), ensure_ascii=False, indent=2) + "\n```\n")

        out_md = args.out_dir / "m0_report.md"
        out_md.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"\nSaved M0 report to {out_md}")
    finally:
        router.close()


if __name__ == "__main__":
    main()
