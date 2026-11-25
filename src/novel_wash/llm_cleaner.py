"""Layer 4: Batched numbered-candidate LLM cleaner with prefix caching, checkpoints, and Tier-C cross-check."""

from __future__ import annotations

try:
    import fcntl
except ImportError:  # Windows fallback: thread lock + tempfile atomic replace remain active
    fcntl = None
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from .edit_log import EditOp
from .inline_rules import DirtyLineTask, InlineAnalysisResult
from .llm_client import LLMRouter, RouterConfigError, RouterContentPolicyError, resolve_model_family
from .verifier import verify_line_decisions

SYSTEM_PROMPT_BASE = (
    "你是中文网文反爬投毒清洗判定器。\n"
    "输入文本曾被防爬虫程序随机插入干扰字。代码已预先剥离符号和生僻字，并用 ⟦编号+字⟧ 标出了候选汉字（其中既有插入的数字谐音/群号水印干扰字，也可能混有误标的正常正文用字）。\n"
    "任务：逐一独立判断每个 [L行号] 里的编号字，应该删除(del)还是保留(keep)。\n"
    "判定原则：\n"
    "1. 只有当该字在当前句子语义中明显是强行插入、去掉后句子才恢复通顺的干扰字时 -> 放入 del；\n"
    "2. 若该字本身是正常句子的组成部分（如代词、动词、名词、正常修饰语、“三条败犬”的“三”、“的时候”、人名/地名或合理数字），必须放入 keep；\n"
    "3. 拿不准时宁可放入 keep（漏删可接受，误删不可接受）；\n"
    "4. 只能对编号做二分类，绝不要输出正文文本或修改任何标点。\n"
    "严格只输出 JSON 对象，格式：\n"
    '{"L行号": {"del": [编号], "keep": [编号]}}'
)


def _no_duplicate_keys_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for k, v in pairs:
        if k in obj:
            raise json.JSONDecodeError(f"Duplicate JSON key: {k!r}", "", 0)
        obj[k] = v
    return obj


_PROSE_BRACKET_TAG_RE = re.compile(
    r"\[(?=[^\[\]{}\"\\,]*[A-Za-z\u4e00-\u9fff])"
    r"(?:(?:\d+\.[ \t]+)?[A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff _\-.:：]*)\]"
)
_PLAIN_PROSE_HEADER_RE = re.compile(
    r"^(?:(?:\d+\.[ \t]+)?[A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff _\-.:：!！?？]*)$"
)
_JSON_LITERAL_WORDS = {"null", "true", "false"}


def _reject_json_constant(val: str) -> Any:
    raise json.JSONDecodeError(f"Non-standard JSON constant not allowed: {val!r}", val, 0)


def _is_valid_prose_prefix(prefix: str) -> bool:
    """Allow only clean non-JSON prose headers/labels (e.g. `Result:`, `[Result]`, `[1. Result]`, `[Step 1] [Result]`)."""
    p = prefix.strip()
    if not p:
        return True
    if any(ch in '{}"\\,' for ch in p):
        return False
    for m in _PROSE_BRACKET_TAG_RE.finditer(p):
        inner = m.group(0)[1:-1].strip()
        tokens = [tok.lower() for tok in re.split(r"[\s:：\-.,]+", inner) if tok]
        if not tokens or any(tok in _JSON_LITERAL_WORDS for tok in tokens):
            return False
    outside = _PROSE_BRACKET_TAG_RE.sub(" ", p)
    if "[" in outside or "]" in outside:
        return False
    for raw_seg in outside.splitlines():
        seg = raw_seg.strip()
        if not seg:
            continue
        if not _PLAIN_PROSE_HEADER_RE.fullmatch(seg):
            return False
        words = [w.strip(":：!！?？.-_").lower() for w in re.split(r"\s+", seg) if w.strip(":：!！?？.-_")]
        if not words or any(w in _JSON_LITERAL_WORDS for w in words):
            return False
    return True


def _extract_json(raw: str) -> dict[str, Any]:
    s = raw.strip()
    if s.startswith("```"):
        if not re.search(r"\n?```\s*$", s) or s.count("```") < 2:
            raise json.JSONDecodeError("Unclosed markdown code fence", s, 0)
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()
    try:
        obj = json.loads(s, object_pairs_hook=_no_duplicate_keys_hook, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        if "Duplicate JSON key" in str(exc) or "Non-standard JSON constant" in str(exc) or "Unclosed markdown" in str(exc):
            raise
        obj_start = s.find("{")
        if obj_start < 0:
            raise json.JSONDecodeError("No top-level JSON object found", s, 0) from exc
        prefix = s[:obj_start]
        decoder = json.JSONDecoder(object_pairs_hook=_no_duplicate_keys_hook, parse_constant=_reject_json_constant)
        cand, end = decoder.raw_decode(s, obj_start)
        suffix = s[end:].strip()
        fence_open = re.search(r"```(?:json)?\s*$", prefix)
        if fence_open and suffix == "```":
            prefix = prefix[: fence_open.start()]
            suffix = ""
        elif fence_open:
            raise json.JSONDecodeError("Unclosed markdown code fence before JSON object", s, 0) from exc
        if suffix:
            raise json.JSONDecodeError("Trailing content after top-level JSON object", s, end)
        if not _is_valid_prose_prefix(prefix):
            raise json.JSONDecodeError("Invalid non-prose prefix before JSON object", s, 0) from exc
        obj = cand
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Top-level JSON value must be an object", s, 0)
    return obj


def _task_context_pair(line_num: int, peeled_lines: list[str]) -> tuple[str, str]:
    """Return immediate previous and next nonblank peeled context lines (matches prompt builder)."""
    prev_ln = line_num - 1
    while prev_ln >= 1 and not peeled_lines[prev_ln - 1].strip():
        prev_ln -= 1
    prev_ctx = peeled_lines[prev_ln - 1].strip()[:120] if prev_ln >= 1 else ""

    next_ln = line_num + 1
    while next_ln <= len(peeled_lines) and not peeled_lines[next_ln - 1].strip():
        next_ln += 1
    next_ctx = peeled_lines[next_ln - 1].strip()[:120] if next_ln <= len(peeled_lines) else ""
    return prev_ctx, next_ctx


def task_fingerprint(task: DirtyLineTask, *, prev_ctx: str = "", next_ctx: str = "") -> str:
    """Stable SHA-256[:16] fingerprint covering original text, pre-peeled line, neighbor [ctx], and exact CIDs."""
    payload = {
        "orig": task.orig_line,
        "peeled": task.pre_peeled_line,
        "prev_ctx": prev_ctx,
        "next_ctx": next_ctx,
        "cands": [[c.cid, c.orig_start, c.orig_end, c.char, c.tier] for c in task.candidates],
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.Lock] = {}


def _get_path_lock(path: Path) -> threading.Lock:
    key = path.resolve()
    with _PATH_LOCKS_GUARD:
        lk = _PATH_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _PATH_LOCKS[key] = lk
        return lk


def _save_atomic_json_locked(path: Path, data: dict[str, Any], merge_key: str | None = None, merge_meta: dict[str, Any] | None = None) -> None:
    """Persist checkpoint atomically under per-path thread + flock locks, merging concurrent writes if valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _get_path_lock(path), lock_path.open("a+", encoding="utf-8") as lk:
        if fcntl is not None:
            fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            if merge_key and path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict) and (merge_meta is None or existing.get("metadata") == merge_meta):
                        for k in ("primary", "secondary"):
                            ex_map = existing.get(k)
                            cur_map = data.setdefault(k, {})
                            if isinstance(ex_map, dict):
                                for line_k, v in ex_map.items():
                                    cur_map.setdefault(line_k, v)
                except Exception:
                    pass
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tf:
                    tf.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
                    tf.flush()
                    os.fsync(tf.fileno())
                os.replace(tmp_name, path)
            except BaseException:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
                raise
        finally:
            if fcntl is not None:
                fcntl.flock(lk.fileno(), fcntl.LOCK_UN)


def build_system_prompt(proper_nouns: list[str]) -> str:
    if not proper_nouns:
        return SYSTEM_PROMPT_BASE
    nouns_str = "、".join(proper_nouns[:60])
    return f"{SYSTEM_PROMPT_BASE}\n本书高频专名/词条参考（切勿误删）：{nouns_str}"


def format_batch_user_prompt(
    batch: list[DirtyLineTask],
    peeled_lines: list[str],
) -> str:
    """Format a batch of dirty tagged lines with 1 read-only context line around non-adjacent entries."""
    lines_out: list[str] = []
    batch_line_nums = {t.line_num for t in batch}
    for idx, t in enumerate(batch):
        prev_ln = t.line_num - 1
        while prev_ln >= 1 and not peeled_lines[prev_ln - 1].strip():
            prev_ln -= 1
        if prev_ln >= 1 and prev_ln not in batch_line_nums and (idx == 0 or batch[idx - 1].line_num < prev_ln):
            lines_out.append(f"[ctx] {peeled_lines[prev_ln - 1].strip()[:120]}")

        lines_out.append(f"[L{t.line_num}] {t.tagged_text}")

        next_ln = t.line_num + 1
        while next_ln <= len(peeled_lines) and not peeled_lines[next_ln - 1].strip():
            next_ln += 1
        if (
            next_ln <= len(peeled_lines)
            and next_ln not in batch_line_nums
            and (idx + 1 == len(batch) or batch[idx + 1].line_num > next_ln)
        ):
            lines_out.append(f"[ctx] {peeled_lines[next_ln - 1].strip()[:120]}")
    return "\n".join(lines_out)


def parse_batch_decisions(
    raw_content: str,
    batch: list[DirtyLineTask],
) -> tuple[dict[int, set[int]], list[DirtyLineTask]]:
    """Strictly parse model JSON decisions, requiring complete disjoint int CID sets per line."""
    parsed = _extract_json(raw_content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM decision payload is not a JSON object")
    del_by_line: dict[int, set[int]] = {}
    missing: list[DirtyLineTask] = []

    for t in batch:
        has_l = f"L{t.line_num}" in parsed
        has_num = str(t.line_num) in parsed
        if has_l and has_num:
            missing.append(t)
            continue
        entry = parsed.get(f"L{t.line_num}") if has_l else parsed.get(str(t.line_num))
        if not isinstance(entry, dict) or "del" not in entry or "keep" not in entry:
            missing.append(t)
            continue
        raw_del = entry["del"]
        raw_keep = entry["keep"]
        if not isinstance(raw_del, list) or not isinstance(raw_keep, list):
            missing.append(t)
            continue
        if any(type(x) is not int for x in raw_del) or any(type(x) is not int for x in raw_keep):
            missing.append(t)
            continue
        valid_ids = {c.cid for c in t.candidates}
        del_ids = set(raw_del)
        keep_ids = set(raw_keep)
        if (
            not del_ids.issubset(valid_ids)
            or not keep_ids.issubset(valid_ids)
            or not del_ids.isdisjoint(keep_ids)
            or (del_ids | keep_ids) != valid_ids
        ):
            missing.append(t)
            continue
        del_by_line[t.line_num] = del_ids

    return del_by_line, missing


def _call_and_parse_once(
    batch: list[DirtyLineTask],
    prompt: str,
    sys_prompt: str,
    router: LLMRouter,
    model: str,
    group_override: str | None,
    *,
    pool_name: str | None = None,
    exclude_families: frozenset[str] = frozenset(),
) -> tuple[str, dict[int, set[int]], list[DirtyLineTask], str, str]:
    """Single attempt returning (status: ok|policy|err, del_map, missing, used_family, used_model)."""
    if pool_name is not None:
        mtok = min(6144, max(2048, 800 + 300 * len(batch)))
        bad_routes: set[str] = set()
        for _ in range(3):
            try:
                raw, m = router.chat_pool(
                    pool_name,
                    sys_prompt,
                    prompt,
                    exclude_families=exclude_families,
                    exclude_routes=frozenset(bad_routes),
                    max_tokens=mtok,
                    defer_success=True,
                )
            except RouterConfigError:
                raise
            except RouterContentPolicyError:
                return "policy", {}, list(batch), "", ""
            except (httpx.HTTPError, RuntimeError):
                return "err", {}, list(batch), "", ""

            if m.finish_reason == "length":
                router.report_bad_output(m.group, m.model, probe_token=m.probe_token)
                bad_routes.add(f"{m.group}/{m.model}")
                continue
            try:
                dm, ms = parse_batch_decisions(raw, batch)
                if batch and not dm:
                    # Non-empty batch yielding zero valid line decisions is treated as bad output, not healthy recovery.
                    router.report_bad_output(m.group, m.model, probe_token=m.probe_token)
                    bad_routes.add(f"{m.group}/{m.model}")
                    continue
                if hasattr(router, "report_good_output"):
                    router.report_good_output(m.group, m.model, probe_token=m.probe_token)
                fam = m.family or resolve_model_family(m.model, getattr(router, "config", None))
                return "ok", dm, ms, fam, m.model
            except (ValueError, KeyError) as exc:
                router.report_bad_output(m.group, m.model, probe_token=m.probe_token)
                if isinstance(exc, json.JSONDecodeError) or str(exc) == "LLM decision payload is not a JSON object":
                    bad_routes.add(f"{m.group}/{m.model}")
                    continue
                raise
            except Exception:
                router.report_bad_output(m.group, m.model, probe_token=m.probe_token)
                raise
        return "err", {}, list(batch), "", ""

    try:
        raw, m = router.chat(model, sys_prompt, prompt, group=group_override)
        dm, ms = parse_batch_decisions(raw, batch)
        m_fam = getattr(m, "family", "") if m is not None else ""
        m_name = getattr(m, "model", model) if m is not None else model
        fam = m_fam or resolve_model_family(m_name, getattr(router, "config", None))
        return "ok", dm, ms, fam, m_name
    except RouterConfigError:
        raise
    except RouterContentPolicyError:
        return "policy", {}, list(batch), "", ""
    except (ValueError, KeyError) as exc:
        if isinstance(exc, json.JSONDecodeError) or str(exc) == "LLM decision payload is not a JSON object":
            return "err", {}, list(batch), "", ""
        raise
    except (httpx.HTTPError, RuntimeError):
        return "err", {}, list(batch), "", ""


def _run_batch_recursive(
    batch: list[DirtyLineTask],
    peeled_lines: list[str],
    sys_prompt: str,
    router: LLMRouter,
    model: str,
    group_override: str | None,
    *,
    singleton_retries: int = 1,
    pool_name: str | None = None,
    exclude_families: frozenset[str] = frozenset(),
    depth: int = 0,
    max_depth: int = 2,
    line_families: dict[int, str] | None = None,
    line_models: dict[int, str] | None = None,
) -> tuple[dict[int, set[int]], list[DirtyLineTask]]:
    if not batch:
        return {}, []
    user_prompt = format_batch_user_prompt(batch, peeled_lines)
    status, del_map, missing, used_fam, used_mod = _call_and_parse_once(
        batch, user_prompt, sys_prompt, router, model, group_override, pool_name=pool_name, exclude_families=exclude_families
    )
    if line_families is not None and used_fam:
        for ln in del_map:
            line_families[ln] = used_fam
    if line_models is not None and used_mod:
        for ln in del_map:
            line_models[ln] = used_mod

    if status == "policy":
        if len(batch) == 1:
            bare_prompt = f"[L{batch[0].line_num}] {batch[0].tagged_text}"
            if bare_prompt != user_prompt:
                st_b, dm_b, ms_b, fam_b, mod_b = _call_and_parse_once(
                    batch, bare_prompt, sys_prompt, router, model, group_override, pool_name=pool_name, exclude_families=exclude_families
                )
                if line_families is not None and fam_b:
                    for ln in dm_b:
                        line_families[ln] = fam_b
                if line_models is not None and mod_b:
                    for ln in dm_b:
                        line_models[ln] = mod_b
                if st_b == "ok":
                    return dm_b, ms_b
            return {}, list(batch)
        singleton_retries = 0

    if not missing:
        return del_map, []

    if len(batch) == 1:
        if singleton_retries > 0:
            sub_dm, sub_ms = _run_batch_recursive(
                batch, peeled_lines, sys_prompt, router, model, group_override,
                singleton_retries=singleton_retries - 1, pool_name=pool_name,
                exclude_families=exclude_families, depth=depth, max_depth=max_depth,
                line_families=line_families, line_models=line_models,
            )
            del_map.update(sub_dm)
            return del_map, sub_ms
        return del_map, missing

    if len(missing) == 1:
        sub_dm, sub_ms = _run_batch_recursive(
            missing, peeled_lines, sys_prompt, router, model, group_override,
            singleton_retries=1, pool_name=pool_name, exclude_families=exclude_families,
            depth=depth, max_depth=max_depth, line_families=line_families, line_models=line_models,
        )
        del_map.update(sub_dm)
        return del_map, sub_ms

    if depth >= max_depth:
        return del_map, missing

    mid = len(missing) // 2
    left_dm, left_ms = _run_batch_recursive(
        missing[:mid], peeled_lines, sys_prompt, router, model, group_override,
        singleton_retries=singleton_retries, pool_name=pool_name, exclude_families=exclude_families,
        depth=depth + 1, max_depth=max_depth, line_families=line_families, line_models=line_models,
    )
    right_dm, right_ms = _run_batch_recursive(
        missing[mid:], peeled_lines, sys_prompt, router, model, group_override,
        singleton_retries=singleton_retries, pool_name=pool_name, exclude_families=exclude_families,
        depth=depth + 1, max_depth=max_depth, line_families=line_families, line_models=line_models,
    )
    del_map.update(left_dm)
    del_map.update(right_dm)
    return del_map, left_ms + right_ms


def _run_model_over_tasks(
    tasks: list[DirtyLineTask],
    peeled_lines: list[str],
    sys_prompt: str,
    router: LLMRouter,
    model: str,
    *,
    batch_size: int,
    checkpoint_data: dict[str, Any],
    checkpoint_key: str,
    checkpoint_path: Path | None,
    group_override: str | None = None,
    progress_label: str = "",
    pool_name: str | None = None,
    exclude_families: frozenset[str] = frozenset(),
    default_family: str = "",
    out_families: dict[int, str] | None = None,
    out_models: dict[int, str] | None = None,
) -> tuple[dict[int, set[int]], set[int]]:
    saved_map: dict[str, Any] = checkpoint_data.setdefault(checkpoint_key, {})
    unresolved_root: dict[str, Any] = checkpoint_data.setdefault("unresolved", {})
    unresolved_map: dict[str, str] = unresolved_root.setdefault(checkpoint_key, {})
    require_provenance = pool_name is not None

    task_fp = {
        t.line_num: task_fingerprint(t, prev_ctx=p_ctx, next_ctx=n_ctx)
        for t in tasks
        for p_ctx, n_ctx in (_task_context_pair(t.line_num, peeled_lines),)
    }
    results: dict[int, set[int]] = {}
    line_families: dict[int, str] = out_families if out_families is not None else {}
    line_models: dict[int, str] = out_models if out_models is not None else {}
    fallback_fam = default_family or resolve_model_family(model, getattr(router, "config", None))
    for t in tasks:
        raw_entry = saved_map.get(str(t.line_num))
        if isinstance(raw_entry, dict) and raw_entry.get("fp") == task_fp[t.line_num]:
            raw_del = raw_entry.get("del")
            if isinstance(raw_del, list) and all(type(x) is int for x in raw_del):
                valid_ids = {c.cid for c in t.candidates}
                raw_set = set(raw_del)
                if raw_set.issubset(valid_ids):
                    entry_fam = str(raw_entry.get("fam") or "").strip().lower()
                    entry_mod = str(raw_entry.get("mod") or "").strip()
                    router_cfg = getattr(router, "config", None) or {}
                    models_cfg = router_cfg.get("models") if isinstance(router_cfg, dict) else None
                    expected_fam = resolve_model_family(entry_mod, router_cfg) if entry_mod else ""
                    if require_provenance:
                        mod_cfg = models_cfg.get(entry_mod) if isinstance(models_cfg, dict) else None
                        mod_enabled = isinstance(mod_cfg, dict) and bool(mod_cfg.get("enabled", True))
                        if (
                            not entry_fam
                            or not entry_mod
                            or not mod_enabled
                            or not expected_fam
                            or entry_fam != expected_fam
                            or entry_fam in exclude_families
                        ):
                            continue
                    else:
                        if entry_mod and expected_fam and entry_fam and entry_fam != expected_fam:
                            continue
                        entry_fam = expected_fam or entry_fam or fallback_fam
                        entry_mod = entry_mod or model
                    results[t.line_num] = raw_set
                    line_families[t.line_num] = entry_fam
                    line_models[t.line_num] = entry_mod

    pending = [t for t in tasks if t.line_num not in results]
    unresolved_lines: set[int] = set()
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    total_batches = len(batches)
    if not batches:
        return results, unresolved_lines

    cfg = getattr(router, "config", None) or {}
    if pool_name is not None and hasattr(router, "pool_concurrency"):
        max_workers = router.pool_concurrency(pool_name, exclude_families=exclude_families)
    else:
        mcfg = cfg.get("models", {}).get(model, {})
        eff_group = group_override or mcfg.get("preferred_group") or ""
        max_workers = max(1, int(cfg.get("groups", {}).get(eff_group, {}).get("max_concurrency", 1)))
    ckpt_lock = threading.Lock()
    meta_ref = checkpoint_data.get("metadata")
    done_b = 0
    dirty_since_flush = 0
    last_flush = time.monotonic()

    def _flush_ckpt_locked() -> None:
        nonlocal dirty_since_flush, last_flush
        if checkpoint_path is not None and dirty_since_flush > 0:
            _save_atomic_json_locked(
                checkpoint_path,
                checkpoint_data,
                merge_key=checkpoint_key,
                merge_meta=meta_ref if isinstance(meta_ref, dict) else None,
            )
            dirty_since_flush = 0
            last_flush = time.monotonic()

    def _commit_batch(
        del_map: dict[int, set[int]],
        missing: list[DirtyLineTask],
        batch_fams: dict[int, str],
        batch_mods: dict[int, str],
    ) -> None:
        nonlocal done_b, dirty_since_flush
        with ckpt_lock:
            for ln, cids in del_map.items():
                results[ln] = cids
                fam = batch_fams.get(ln) or ("" if require_provenance else fallback_fam)
                mod = batch_mods.get(ln) or ("" if require_provenance else model)
                line_families[ln] = fam
                line_models[ln] = mod
                saved_map[str(ln)] = {"fp": task_fp[ln], "del": sorted(cids), "fam": fam, "mod": mod}
                unresolved_map.pop(str(ln), None)
                unresolved_lines.discard(ln)
            for m_task in missing:
                unresolved_lines.add(m_task.line_num)
                unresolved_map[str(m_task.line_num)] = task_fp[m_task.line_num]
            dirty_since_flush += 1
            done_b += 1
            if dirty_since_flush >= 5 or (time.monotonic() - last_flush) >= 2.0 or done_b == total_batches:
                _flush_ckpt_locked()
            if progress_label and (done_b == 1 or done_b % 5 == 0 or done_b == total_batches):
                print(
                    f"  [{progress_label}] batch {done_b}/{total_batches} "
                    f"(cost={router.ledger.total_cost_cny():.4f} CNY)",
                    flush=True,
                )

    def _run_batch(batch: list[DirtyLineTask]) -> tuple[dict[int, set[int]], list[DirtyLineTask], dict[int, str], dict[int, str]]:
        b_fams: dict[int, str] = {}
        b_mods: dict[int, str] = {}
        dm, ms = _run_batch_recursive(
            batch, peeled_lines, sys_prompt, router, model, group_override,
            pool_name=pool_name, exclude_families=exclude_families,
            line_families=b_fams, line_models=b_mods,
        )
        return dm, ms, b_fams, b_mods

    try:
        if max_workers == 1 or total_batches == 1:
            for batch in batches:
                dm, ms, bf, bm = _run_batch(batch)
                _commit_batch(dm, ms, bf, bm)
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, total_batches)) as pool:
                futs = [pool.submit(_run_batch, batch) for batch in batches]
                for fut in as_completed(futs):
                    dm, ms, bf, bm = fut.result()
                    _commit_batch(dm, ms, bf, bm)
    finally:
        with ckpt_lock:
            _flush_ckpt_locked()

    return results, unresolved_lines


def run_llm_inline_cleaning(
    inline_res: InlineAnalysisResult,
    peeled_lines: list[str],
    router: LLMRouter,
    *,
    primary_model: str,
    secondary_model: str | None,
    source_sha256: str,
    checkpoint_path: Path | None = None,
    batch_size: int = 16,
    secondary_batch_size: int = 12,
    group_override: str | None = None,
    use_pools: bool = False,
) -> tuple[list[EditOp], list[dict[str, Any]], dict[str, Any]]:
    """Run primary batched classification + targeted Tier-C cross-family verification."""
    tasks = inline_res.tasks
    if not tasks:
        return [], [], {
            "primary_tasks": 0,
            "tier_c_crosscheck_tasks": 0,
            "rejected_ops": 0,
            "primary_unresolved": 0,
            "secondary_unresolved": 0,
            "incomplete": False,
        }

    cfg = getattr(router, "config", None) or {}
    pools_enabled = bool(use_pools and cfg.get("pools"))
    primary_family = resolve_model_family(primary_model, cfg)
    secondary_family = resolve_model_family(secondary_model, cfg) if secondary_model else ""
    valid_cross_family = bool(
        secondary_model
        and secondary_model != primary_model
        and primary_family
        and secondary_family
        and primary_family != secondary_family
    )

    sys_prompt = build_system_prompt(inline_res.proper_nouns)
    base_meta = {
        "schema_version": 3,
        "source_sha256": source_sha256,
        "primary_model": primary_model,
        "secondary_model": secondary_model,
        "primary_family": primary_family,
        "secondary_family": secondary_family,
        "group_override": group_override,
        "prompt_sha256": hashlib.sha256(sys_prompt.encode("utf-8")).hexdigest()[:16],
        "task_count": len(tasks),
    }
    metadata = {**base_meta, "mode": "pooled" if pools_enabled else "single"}

    checkpoint_data: dict[str, Any] = {
        "metadata": metadata,
        "primary": {},
        "secondary": {},
        "unresolved": {"primary": {}, "secondary": {}},
    }
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                l_meta = loaded.get("metadata")
                if l_meta == metadata or (not pools_enabled and l_meta == base_meta):
                    checkpoint_data = loaded
                    checkpoint_data["metadata"] = metadata
                    checkpoint_data.setdefault("primary", {})
                    checkpoint_data.setdefault("secondary", {})
                    checkpoint_data.setdefault("unresolved", {"primary": {}, "secondary": {}})
                elif pools_enabled and isinstance(l_meta, dict) and l_meta.get("mode") == "single":
                    # Explicit migration of a known single-model checkpoint into pooled mode
                    l_core = {k: v for k, v in l_meta.items() if k != "mode"}
                    if l_core == base_meta:
                        checkpoint_data = loaded
                        checkpoint_data["metadata"] = metadata
                        for k_sec, fam_s, mod_s in (
                            ("primary", primary_family, primary_model),
                            ("secondary", secondary_family, secondary_model or ""),
                        ):
                            sec_map = checkpoint_data.setdefault(k_sec, {})
                            if isinstance(sec_map, dict) and fam_s and mod_s:
                                for v in sec_map.values():
                                    if isinstance(v, dict):
                                        v.setdefault("fam", fam_s)
                                        v.setdefault("mod", mod_s)
                        checkpoint_data.setdefault("unresolved", {"primary": {}, "secondary": {}})
        except Exception:
            pass

    line_primary_fams: dict[int, str] = {}
    line_primary_mods: dict[int, str] = {}
    primary_decisions, primary_unresolved = _run_model_over_tasks(
        tasks,
        peeled_lines,
        sys_prompt,
        router,
        primary_model,
        batch_size=batch_size,
        checkpoint_data=checkpoint_data,
        checkpoint_key="primary",
        checkpoint_path=checkpoint_path,
        group_override=group_override,
        progress_label="primary:pool" if pools_enabled else f"primary:{primary_model}",
        pool_name="primary" if pools_enabled else None,
        default_family=primary_family,
        out_families=line_primary_fams,
        out_models=line_primary_mods,
    )

    # Targeted cross-check ONLY on resolved lines where primary proposes deleting at least one Tier-C site
    tier_c_tasks: list[DirtyLineTask] = []
    for t in tasks:
        if t.line_num in primary_unresolved:
            continue
        p_del = primary_decisions.get(t.line_num, set())
        if any(c.tier == "C" and c.cid in p_del for c in t.candidates):
            tier_c_tasks.append(t)

    secondary_decisions: dict[int, set[int]] = {}
    secondary_unresolved: set[int] = set()
    line_secondary_fams: dict[int, str] = {}
    line_secondary_mods: dict[int, str] = {}
    if tier_c_tasks and pools_enabled:
        by_fam: dict[str, list[DirtyLineTask]] = {}
        for t in tier_c_tasks:
            pfam = line_primary_fams.get(t.line_num, "")
            by_fam.setdefault(pfam, []).append(t)
        for pfam, fam_tasks in sorted(by_fam.items()):
            has_cross = bool(pfam) and hasattr(router, "pool_has_eligible_route") and router.pool_has_eligible_route(
                "secondary", exclude_families=frozenset({pfam})
            )
            if not has_cross:
                secondary_unresolved.update(t.line_num for t in fam_tasks)
                continue
            s_dec, s_unres = _run_model_over_tasks(
                fam_tasks,
                peeled_lines,
                sys_prompt,
                router,
                secondary_model or primary_model,
                batch_size=secondary_batch_size,
                checkpoint_data=checkpoint_data,
                checkpoint_key="secondary",
                checkpoint_path=checkpoint_path,
                group_override=group_override,
                progress_label=f"tier-c:pool(!={pfam})",
                pool_name="secondary",
                exclude_families=frozenset({pfam}),
                default_family=secondary_family,
                out_families=line_secondary_fams,
                out_models=line_secondary_mods,
            )
            secondary_decisions.update(s_dec)
            secondary_unresolved.update(s_unres)
    elif valid_cross_family and secondary_model and tier_c_tasks:
        secondary_decisions, secondary_unresolved = _run_model_over_tasks(
            tier_c_tasks,
            peeled_lines,
            sys_prompt,
            router,
            secondary_model,
            batch_size=secondary_batch_size,
            checkpoint_data=checkpoint_data,
            checkpoint_key="secondary",
            checkpoint_path=checkpoint_path,
            group_override=group_override,
            progress_label=f"tier-c:{secondary_model}",
            default_family=secondary_family,
            out_families=line_secondary_fams,
            out_models=line_secondary_mods,
        )

    all_edits: list[EditOp] = []
    all_review: list[dict[str, Any]] = []
    rejected_total = 0
    proper_set = set(inline_res.proper_nouns)

    for t in tasks:
        if t.line_num in primary_unresolved:
            all_review.append(
                {
                    "line": t.line_num,
                    "reason": "llm_primary_unresolved",
                    "original": t.orig_line,
                }
            )
            continue
        p_del = primary_decisions.get(t.line_num, set())
        p_fam_line = line_primary_fams.get(t.line_num, "" if pools_enabled else primary_family)
        p_mod_line = line_primary_mods.get(t.line_num, "" if pools_enabled else primary_model)
        s_fam_line = line_secondary_fams.get(t.line_num, "" if pools_enabled else secondary_family)
        s_mod_line = line_secondary_mods.get(t.line_num, "" if pools_enabled else (secondary_model or ""))

        if t.line_num in secondary_unresolved:
            s_del = None
            all_review.append(
                {
                    "line": t.line_num,
                    "reason": "llm_secondary_unresolved",
                    "original": t.orig_line,
                }
            )
        else:
            s_del = secondary_decisions.get(t.line_num) if (secondary_model or pools_enabled) else None
            if pools_enabled and s_del is not None:
                if not p_fam_line or not s_fam_line or not p_mod_line or not s_mod_line or p_fam_line == s_fam_line:
                    s_del = None
        outcome = verify_line_decisions(
            t,
            p_del,
            p_mod_line or primary_model,
            secondary_del_cids=s_del,
            secondary_model=(s_mod_line or None) if (s_del is not None or not pools_enabled) else None,
            primary_family=p_fam_line,
            secondary_family=s_fam_line,
            proper_nouns=proper_set,
        )
        all_edits.extend(outcome.edits)
        all_review.extend(outcome.review_rows)
        rejected_total += outcome.rejected_count

    stats = {
        "primary_tasks": len(tasks),
        "tier_c_crosscheck_tasks": len(tier_c_tasks),
        "llm_approved_edits": sum(1 for e in all_edits if e.status == "approved"),
        "llm_review_edits": sum(1 for e in all_edits if e.status == "review"),
        "rejected_ops": rejected_total,
        "primary_unresolved": len(primary_unresolved),
        "secondary_unresolved": len(secondary_unresolved),
        "incomplete": bool(primary_unresolved or secondary_unresolved),
    }
    return all_edits, all_review, stats
