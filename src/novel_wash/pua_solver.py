"""Layer 1: Fanqie PUA substitution cipher solver & global versioned cache.

Solves 1-to-1 PUA codepoint mappings (e.g. U+E3E8..U+E55B) using interleaved plaintext
contexts + batched LLM calls + cross-check & jieba coherence validation, caching maps
by codepoint-set fingerprint for reuse across books.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import jieba

from .edit_log import EditOp

jieba.setLogLevel(60)

PUA_RE = re.compile(r"[\ue000-\uf8ff]")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class PuaSolveResult:
    pua_count: int
    unique_codepoints: int
    char_map: dict[str, str]
    edits: list[EditOp]
    cache_hit: bool
    jieba_ratio_before: float
    jieba_ratio_after: float
    uncertain_codepoints: list[str]


def _lexicon_coherence(lines: list[str]) -> float:
    sample = "\n".join(lines[:1500])
    runs = CJK_RUN_RE.findall(sample)
    total = sum(len(r) for r in runs)
    if total == 0:
        return 0.0
    multi = 0
    for r in runs:
        if len(r) < 2:
            continue
        for w in jieba.cut(r):
            if len(w) >= 2 and jieba.get_FREQ(w, 0) > 0:
                multi += len(w)
    return round(multi / total, 4)


def _fingerprint_codepoints(codepoints: list[str]) -> str:
    joined = ",".join(f"U+{ord(c):04X}" for c in sorted(codepoints))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _source_pua_fingerprint(
    lines: list[str],
    ordered_cps: list[str] | None = None,
    counts: Counter[str] | None = None,
) -> str:
    """Full-content SHA-256 digest binding cached PUA mappings to the complete source input."""
    del ordered_cps, counts
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _is_valid_bijection(cmap: dict[str, str]) -> bool:
    vals = list(cmap.values())
    if any(len(v) != 1 or not ("\u4e00" <= v <= "\u9fff") for v in vals):
        return False
    return len(set(vals)) == len(vals)


def _top_cp_coherence_ratio(
    lines_by_cp: dict[str, list[str]],
    ordered_cps: list[str],
    cmap: dict[str, str],
    counts: Counter[str],
) -> float:
    sample_cps = [cp for cp in ordered_cps[:10] if cp in cmap]
    if not sample_cps:
        return 0.0
    hits = sum(1 for cp in sample_cps if _local_jieba_score(lines_by_cp, cp, cmap[cp], cmap, counts[cp]) > 0.0)
    return hits / len(sample_cps)


def _apply_map_to_lines(lines: list[str], cmap: dict[str, str]) -> list[str]:
    if not cmap:
        return lines
    trans = str.maketrans(cmap)
    return [line.translate(trans) for line in lines]


def _collect_contexts(
    lines: list[str],
    codepoints: list[str],
    current_map: dict[str, str],
    *,
    max_contexts_per_cp: int = 4,
) -> dict[str, list[str]]:
    """Collect clean context snippets where `cp` appears as `[P<hex>]` and other solved PUA are substituted."""
    need = set(codepoints)
    contexts: dict[str, list[str]] = {cp: [] for cp in codepoints}
    for line in lines:
        if not line.strip() or not PUA_RE.search(line):
            continue
        cps_in_line = set(PUA_RE.findall(line)) & need
        if not cps_in_line:
            continue
        for cp in cps_in_line:
            if len(contexts[cp]) >= max_contexts_per_cp:
                continue
            # Prefer lines where most other PUA chars are already resolved
            other_unresolved = sum(1 for c in PUA_RE.findall(line) if c != cp and c not in current_map)
            # Build readable snippet with target cp marked as ⟦?⟧
            snippet_chars: list[str] = []
            for ch in line:
                if ch == cp:
                    snippet_chars.append("⟦?⟧")
                elif ch in current_map:
                    snippet_chars.append(current_map[ch])
                elif "\ue000" <= ch <= "\uf8ff":
                    snippet_chars.append("□")
                else:
                    snippet_chars.append(ch)
            snippet = "".join(snippet_chars)
            if len(snippet) > 90:
                pos = snippet.find("⟦?⟧")
                lo = max(0, pos - 36)
                hi = min(len(snippet), pos + 40)
                snippet = snippet[lo:hi]
            # Sort/insert favoring fewer unresolved blanks
            contexts[cp].append((other_unresolved, snippet))

    out: dict[str, list[str]] = {}
    for cp, items in contexts.items():
        items.sort(key=lambda x: x[0])
        out[cp] = [s for _, s in items[:max_contexts_per_cp]]
    return out


def _extract_json_dict(raw: str) -> dict[str, Any]:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return json.loads(s[start : end + 1])
    return json.loads(s)


_TOP_ZH_FREQ_BONUS = set(
    "的一是不了在人我有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从"
)


def _render_tag_line(line: str, char_map: dict[str, str], *, highlight_cp: str | None = None) -> str:
    parts: list[str] = []
    for ch in line:
        if highlight_cp is not None and ch == highlight_cp:
            parts.append("⟦?⟧")
        elif ch in char_map:
            parts.append(char_map[ch])
        elif "\ue000" <= ch <= "\uf8ff":
            parts.append(f"[{ord(ch):04X}]")
        else:
            parts.append(ch)
    return "".join(parts)


def _local_jieba_score(
    lines_by_cp: dict[str, list[str]],
    cp: str,
    candidate_char: str,
    char_map: dict[str, str],
    total_count: int,
) -> float:
    """Estimate bigram/word coherence gain when assigning `cp -> candidate_char`."""
    test_map = dict(char_map)
    test_map[cp] = candidate_char
    sample_lines = lines_by_cp.get(cp, [])[:10]
    if not sample_lines:
        return 0.0
    hits = 0
    for line in sample_lines:
        rendered = _render_tag_line(line, test_map)
        idx = 0
        while True:
            pos = rendered.find(candidate_char, idx)
            if pos < 0:
                break
            lo = max(0, pos - 4)
            hi = min(len(rendered), pos + 5)
            frag = "".join(c for c in rendered[lo:hi] if "\u4e00" <= c <= "\u9fff")
            if frag:
                for w in jieba.cut(frag):
                    if len(w) >= 2 and candidate_char in w and jieba.get_FREQ(w, 0) > 0:
                        hits += len(w)
                        break
            idx = pos + 1
    freq_bonus = 0.25 if (total_count >= 50 and candidate_char in _TOP_ZH_FREQ_BONUS) else 0.0
    return (hits / len(sample_lines)) + freq_bonus


def _select_paragraph_blocks(
    lines: list[str],
    char_map: dict[str, str],
    *,
    max_blocks: int = 10,
    max_chars_per_block: int = 1500,
) -> list[list[str]]:
    """Build contiguous non-empty paragraph blocks covering diverse unresolved PUA codepoints."""
    windows: list[tuple[set[str], list[str]]] = []
    cur: list[str] = []
    cur_len = 0
    cur_tags: set[str] = set()

    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        pua_here = {c for c in PUA_RE.findall(s) if c not in char_map}
        cur.append(s)
        cur_len += len(s) + 1
        cur_tags.update(pua_here)
        if cur_len >= max_chars_per_block or len(cur) >= 35:
            if cur_tags:
                windows.append((cur_tags, cur))
            cur = []
            cur_len = 0
            cur_tags = set()
    if cur and cur_tags:
        windows.append((cur_tags, cur))

    if not windows:
        return []

    selected: list[list[str]] = []
    covered: set[str] = set()
    # Always start with opening chapter block
    first_tags, first_lines = windows.pop(0)
    selected.append(first_lines)
    covered.update(first_tags)

    while windows and len(selected) < max_blocks:
        best_idx = max(range(len(windows)), key=lambda i: len(windows[i][0] - covered))
        gain = len(windows[best_idx][0] - covered)
        if gain == 0:
            break
        tags, blk = windows.pop(best_idx)
        selected.append(blk)
        covered.update(tags)
    return selected


_COMMON_ZH_POOL = (
    "的一是不了在人我有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从别手口打走特和生"
)


@lru_cache(maxsize=1)
def _get_jieba_bigram_lm() -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    """Build cached Simplified Chinese character bigram & unigram counts from jieba's lexicon."""
    jieba.initialize()
    bigrams: Counter[tuple[str, str]] = Counter()
    unigrams: Counter[str] = Counter()
    for word, freq in jieba.dt.FREQ.items():
        if not freq or len(word) < 2:
            continue
        if not all("\u4e00" <= ch <= "\u9fff" for ch in word):
            continue
        for k, ch in enumerate(word):
            unigrams[ch] += freq
            if k + 1 < len(word):
                bigrams[(ch, word[k + 1])] += freq
    return dict(bigrams), dict(unigrams)


def _book_plain_bigrams(lines: list[str]) -> Counter[tuple[str, str]]:
    bg: Counter[tuple[str, str]] = Counter()
    for line in lines:
        for run in CJK_RUN_RE.findall(line):
            for k in range(len(run) - 1):
                bg[(run[k], run[k + 1])] += 200
    return bg


def _max_weight_bipartite_matching(weights: list[list[float]]) -> list[int]:
    """Solve rectangular N x M (M >= N) maximum-weight assignment via Kuhn-Munkres in O(N^2 * M)."""
    n = len(weights)
    if n == 0:
        return []
    m = len(weights[0])
    if m < n:
        raise ValueError(f"Matching matrix requires cols >= rows ({m} < {n})")

    # Convert max-weight to non-negative min-cost per row
    cost = [[0.0] * m for _ in range(n)]
    for i in range(n):
        row_max = max(weights[i])
        for j in range(m):
            cost[i][j] = row_max - weights[i][j]

    inf = 1e18
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        row_i = cost[i - 1]
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            row_i0 = cost[i0 - 1]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row_i0[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    ans = [-1] * n
    for j in range(1, m + 1):
        if p[j] > 0:
            ans[p[j] - 1] = j - 1
    return ans


def _lm_context_score(
    lines_by_cp: dict[str, list[str]],
    cp: str,
    cand_ch: str,
    trial_map: dict[str, str],
    jb_bigrams: dict[tuple[str, str], int],
    book_bigrams: Counter[tuple[str, str]],
) -> float:
    """Compute average character bigram log-score of assigning `cp -> cand_ch` in `trial_map` (collisions allowed)."""
    sample_lines = lines_by_cp.get(cp, [])[:24]
    if not sample_lines:
        return 0.0
    tmap = dict(trial_map)
    tmap[cp] = cand_ch
    total = 0.0
    occ = 0
    for line in sample_lines:
        n = len(line)
        for idx, ch in enumerate(line):
            if ch != cp:
                continue
            occ += 1
            if idx > 0:
                raw_l = line[idx - 1]
                lch = tmap.get(raw_l, raw_l) if ("\ue000" <= raw_l <= "\uf8ff") else raw_l
                if "\u4e00" <= lch <= "\u9fff":
                    f = jb_bigrams.get((lch, cand_ch), 0) + book_bigrams.get((lch, cand_ch), 0)
                    total += math.log1p(f)
            if idx + 1 < n:
                raw_r = line[idx + 1]
                rch = tmap.get(raw_r, raw_r) if ("\ue000" <= raw_r <= "\uf8ff") else raw_r
                if "\u4e00" <= rch <= "\u9fff":
                    f = jb_bigrams.get((cand_ch, rch), 0) + book_bigrams.get((cand_ch, rch), 0)
                    total += math.log1p(f)
    return (total / max(1, occ)) / 10.0  # Normalize roughly into [0.0, 2.2]


def _render_kwic_contexts(
    lines_by_cp: dict[str, list[str]],
    cp: str,
    char_map: dict[str, str],
    *,
    max_contexts: int = 5,
) -> list[str]:
    """Render concise decoded snippets around `cp` marking target as `[cur]` or `[?]`."""
    cur_label = char_map.get(cp, "?")
    cands: list[tuple[int, str]] = []
    for line in lines_by_cp.get(cp, [])[:25]:
        pos = line.find(cp)
        if pos < 0:
            continue
        lo = max(0, pos - 9)
        hi = min(len(line), pos + 10)
        seg = line[lo:hi]
        unresolved = 0
        out: list[str] = []
        target_marked = False
        for idx_s, ch in enumerate(seg):
            if lo + idx_s == pos and not target_marked:
                out.append(f"[{cur_label}]")
                target_marked = True
            elif ch in char_map:
                out.append(char_map[ch])
            elif "\ue000" <= ch <= "\uf8ff":
                out.append("□")
                unresolved += 1
            else:
                out.append(ch)
        cands.append((unresolved, "".join(out)))
    cands.sort(key=lambda x: x[0])
    seen: set[str] = set()
    uniq: list[str] = []
    for _, sn in cands:
        if sn not in seen:
            seen.add(sn)
            uniq.append(sn)
        if len(uniq) >= max_contexts:
            break
    return uniq


def _parse_kwic_proposals(raw: str, valid_hex: set[str]) -> dict[str, list[tuple[str, float]]]:
    """Parse flexible `{"HEX": ["字", conf, "备选"]}` or `{"HEX": "字"}` proposals."""
    try:
        parsed = _extract_json_dict(raw)
    except Exception:
        return {}
    out: dict[str, list[tuple[str, float]]] = {}
    for k, val in parsed.items():
        hx = str(k).strip().strip("[]").upper()
        if hx not in valid_hex:
            continue
        props: list[tuple[str, float]] = []
        if isinstance(val, str):
            ch = val.strip()
            if len(ch) == 1 and "\u4e00" <= ch <= "\u9fff":
                props.append((ch, 0.9))
        elif isinstance(val, list) and val:
            ch1 = str(val[0]).strip()
            conf = 0.85
            if len(val) >= 2 and isinstance(val[1], (int, float)):
                conf = max(0.1, min(1.0, float(val[1])))
            if len(ch1) == 1 and "\u4e00" <= ch1 <= "\u9fff":
                props.append((ch1, conf))
            if len(val) >= 3:
                ch2 = str(val[2]).strip()
                if len(ch2) == 1 and "\u4e00" <= ch2 <= "\u9fff" and ch2 != ch1:
                    props.append((ch2, max(0.2, conf * 0.55)))
        if props:
            out[hx] = props
    return out


def _solve_pua_matching_round(
    ordered_cps: list[str],
    cur_map: dict[str, str],
    lines_by_cp: dict[str, list[str]],
    counts: Counter[str],
    votes_by_cp: dict[str, dict[str, float]],
    consensus_bonus: dict[tuple[str, str], float],
    jb_bigrams: dict[tuple[str, str], int],
    jb_unigrams: dict[str, int],
    book_bigrams: Counter[tuple[str, str]],
    implicit_keep: dict[str, float] | None = None,
) -> dict[str, str]:
    """Build proposal context M*, score (cp, char) jointly, and solve exact Hungarian matching twice."""
    # Proposal context M* allows collisions intentionally so paired swaps (门口->自己) reinforce each other!
    m_star: dict[str, str] = dict(cur_map)
    for cp in ordered_cps:
        vmap = votes_by_cp.get(cp)
        if vmap:
            best_c, best_w = max(vmap.items(), key=lambda kv: kv[1])
            if best_w >= 1.5:
                m_star[cp] = best_c

    cur_vals = set(cur_map.values())
    cand_pool: set[str] = set(cur_vals)
    for cp in ordered_cps:
        for ch in votes_by_cp.get(cp, {}):
            cand_pool.add(ch)
    for ch in _COMMON_ZH_POOL:
        if len(cand_pool) < len(ordered_cps) + 120:
            cand_pool.add(ch)
    cols = sorted(cand_pool)
    col_idx = {ch: j for j, ch in enumerate(cols)}
    n_rows = len(ordered_cps)
    n_cols = len(cols)
    unmapped_weight = 0.25
    keep_bonus = implicit_keep or {}

    def _run_once(ctx_map: dict[str, str]) -> dict[str, str]:
        weights: list[list[float]] = [[-1e6] * (n_cols + n_rows) for _ in range(n_rows)]
        for i, cp in enumerate(ordered_cps):
            row_cands: set[str] = set(votes_by_cp.get(cp, {}).keys())
            if cp in cur_map:
                row_cands.add(cur_map[cp])
            # Expand common pool ONLY when LLM actually proposed candidates for this codepoint
            if votes_by_cp.get(cp):
                row_cands.update(_COMMON_ZH_POOL[:60])
            elif cp in cur_map:
                # Allow swaps/cycles with other already-mapped characters if beneficial
                row_cands.update(cur_map.values())
            cur_ch = cur_map.get(cp)
            vmap = votes_by_cp.get(cp, {})
            for ch in row_cands:
                j = col_idx.get(ch)
                if j is None:
                    continue
                lm_sc = _lm_context_score(lines_by_cp, cp, ch, ctx_map, jb_bigrams, book_bigrams)
                llm_sc = vmap.get(ch, 0.0)
                # Reject zero-vote mappings for unmapped chars or cold-pool substitutions without votes/swap
                if cp not in cur_map and llm_sc <= 0.0:
                    continue
                if ch != cur_ch and llm_sc <= 0.0 and not vmap and ch not in cur_map.values():
                    continue
                if ch != cur_ch and llm_sc <= 0.0 and lm_sc < 0.35:
                    continue
                if ch == cur_ch:
                    stay_sc = keep_bonus.get(cp, 1.15)
                else:
                    stay_sc = 0.0
                cons_sc = consensus_bonus.get((cp, ch), 0.0)
                freq_sc = 0.04 * math.log1p(jb_unigrams.get(ch, 0)) if counts[cp] >= 15 else 0.0
                weights[i][j] = llm_sc + 1.6 * lm_sc + stay_sc + cons_sc + freq_sc
            weights[i][n_cols + i] = unmapped_weight

        assigned = _max_weight_bipartite_matching(weights)
        matched: dict[str, str] = {}
        for i, cp in enumerate(ordered_cps):
            col = assigned[i]
            if 0 <= col < n_cols:
                matched[cp] = cols[col]
        return matched

    m1 = _run_once(m_star)
    m2 = _run_once(m1)
    return m2


def _polish_pua_map_globally(
    lines: list[str],
    ordered_cps: list[str],
    char_map: dict[str, str],
    lines_by_cp: dict[str, list[str]],
    counts: Counter[str],
    llm_call: Callable[[str, str, str | None], str],
    *,
    primary_model: str | None,
    secondary_model: str | None,
    prior_uncertain: list[str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Break bijective deadlocks and fill holes via unconstrained KWIC + Hungarian matching."""
    jb_bigrams, jb_unigrams = _get_jieba_bigram_lm()
    book_bigrams = _book_plain_bigrams(lines)
    hex_to_cp = {f"{ord(cp):04X}": cp for cp in ordered_cps}
    valid_hex = set(hex_to_cp.keys())

    seed_map = dict(char_map)
    cur_map = dict(char_map)
    cur_coh = _lexicon_coherence(_apply_map_to_lines(lines, cur_map))
    votes_by_cp: dict[str, dict[str, float]] = {}
    primary_top: dict[str, str] = {}
    consensus_bonus: dict[tuple[str, str], float] = {}
    implicit_keep: dict[str, float] = {}

    kwic_sys = (
        "你是顶尖中文小说替代密码审校专家。小说中部分汉字被一对一替换，当前已有初步译文，但其中约 10%-20% 的高频或低频字"
        "可能因为局部词汇恰好通顺而被错误锁定（如把“自己”错译成“门口”、把“一个”错译成“一样”、把“手里”错译成“段里”、把“知道”错译成“站辙”、把“得到”错译成“得底”），或尚未译出（显示为 [?]）。\n"
        "任务：逐行审阅每个编码的全部上下文（待审字用方括号 [当前字] 标出，其余邻近字为当前译文，也可能含错），结合汉语语法与剧情语义：\n"
        "1. 当前映射完全正确的条目直接跳过，不需要输出；\n"
        "2. 凡是当前映射明显不通顺、可疑或为 [?] 的条目，请给出真实简体汉字。\n"
        "重要：不需要避开任何“已占用汉字”！允许不同编码输出同一个字，程序会自动用全局二分匹配解决对换与冲突。\n"
        "只输出严格 JSON 对象，格式：{\"E4E7\": [\"自\", 0.95, \"门\"], \"E3F5\": [\"和\", 0.9]}（分别为首选字、0~1置信度、可选次优字）。"
    )

    # Round 1: Focused KWIC audit in batches of ~50 codepoints so zero entries get overlooked
    batch_sz = 50
    for b_idx in range(0, len(ordered_cps), batch_sz):
        batch = ordered_cps[b_idx : b_idx + batch_sz]
        rows = []
        for cp in batch:
            hx = f"{ord(cp):04X}"
            cur_ch = cur_map.get(cp, "?")
            snips = " ‖ ".join(_render_kwic_contexts(lines_by_cp, cp, cur_map, max_contexts=5))
            rows.append(f"{hx} cur={cur_ch} n={counts[cp]} | {snips}")
        try:
            raw = llm_call(kwic_sys, "\n".join(rows), primary_model)
            props = _parse_kwic_proposals(raw, valid_hex)
        except Exception:
            props = {}
        flagged_cps: set[str] = set()
        for hx, cands in props.items():
            cp = hex_to_cp[hx]
            flagged_cps.add(cp)
            vmap = votes_by_cp.setdefault(cp, {})
            for idx_c, (ch, conf) in enumerate(cands):
                w = (3.6 * conf) if idx_c == 0 else (1.8 * conf)
                vmap[ch] = max(vmap.get(ch, 0.0), w)
                if idx_c == 0:
                    primary_top[cp] = ch
        for cp in batch:
            if cp in cur_map and cp not in flagged_cps:
                # Audited as fluent by LLM: mild guard against low-freq stealing while allowing explicit fixes
                implicit_keep[cp] = 1.3 + 0.12 * math.log1p(counts[cp])

    cand_map_r1 = _solve_pua_matching_round(
        ordered_cps, cur_map, lines_by_cp, counts, votes_by_cp, consensus_bonus,
        jb_bigrams, jb_unigrams, book_bigrams, implicit_keep=implicit_keep,
    )
    if _is_valid_bijection(cand_map_r1):
        coh_r1 = _lexicon_coherence(_apply_map_to_lines(lines, cand_map_r1))
        if coh_r1 >= cur_coh - 0.005 and len(cand_map_r1) >= len(cur_map):
            print(f"  [pua] pass4 r1 polish: mapped={len(cand_map_r1)}/{len(ordered_cps)} coh={cur_coh:.4f}->{coh_r1:.4f}", flush=True)
            cur_map = cand_map_r1
            cur_coh = coh_r1

    # Round 2: Cross-family hole-fill & verification of changed/unmapped/low-LM codepoints in batches of 50
    verify_model = secondary_model or primary_model
    scored_cps = [
        (
            cp,
            _lm_context_score(lines_by_cp, cp, cur_map[cp], cur_map, jb_bigrams, book_bigrams)
            if cp in cur_map
            else -1.0,
        )
        for cp in ordered_cps
    ]
    scored_cps.sort(key=lambda x: x[1])
    suspects = [
        cp
        for cp, sc in scored_cps
        if cp not in cur_map or cur_map.get(cp) != char_map.get(cp) or sc < 0.88
    ][:120]
    if suspects:
        for sb_idx in range(0, len(suspects), 50):
            s_batch = suspects[sb_idx : sb_idx + 50]
            rows2 = []
            for cp in s_batch:
                hx = f"{ord(cp):04X}"
                cur_ch = cur_map.get(cp, "?")
                snips = " ‖ ".join(_render_kwic_contexts(lines_by_cp, cp, cur_map, max_contexts=5))
                rows2.append(f"{hx} cur={cur_ch} n={counts[cp]} | {snips}")
            try:
                raw2 = llm_call(kwic_sys, "\n".join(rows2), verify_model)
                props2 = _parse_kwic_proposals(raw2, valid_hex)
            except Exception:
                props2 = {}
            flagged2: set[str] = set()
            for hx, cands in props2.items():
                cp = hex_to_cp[hx]
                flagged2.add(cp)
                vmap = votes_by_cp.setdefault(cp, {})
                for idx_c, (ch, conf) in enumerate(cands):
                    w = (3.4 * conf) if idx_c == 0 else (1.7 * conf)
                    vmap[ch] = vmap.get(ch, 0.0) + w
                    if idx_c == 0 and primary_top.get(cp) == ch and verify_model != primary_model:
                        consensus_bonus[(cp, ch)] = 1.8
            for cp in s_batch:
                if cp in cur_map and cp not in flagged2:
                    implicit_keep[cp] = max(implicit_keep.get(cp, 1.0), 1.2 + 0.10 * math.log1p(counts[cp]))

        cand_map_r2 = _solve_pua_matching_round(
            ordered_cps, cur_map, lines_by_cp, counts, votes_by_cp, consensus_bonus,
            jb_bigrams, jb_unigrams, book_bigrams, implicit_keep=implicit_keep,
        )
        if _is_valid_bijection(cand_map_r2):
            coh_r2 = _lexicon_coherence(_apply_map_to_lines(lines, cand_map_r2))
            if coh_r2 >= cur_coh - 0.005 and len(cand_map_r2) >= len(cur_map):
                print(f"  [pua] pass4 r2 ({verify_model}) polish: mapped={len(cand_map_r2)}/{len(ordered_cps)} coh={cur_coh:.4f}->{coh_r2:.4f}", flush=True)
                cur_map = cand_map_r2
                cur_coh = coh_r2

    # Round 3: Continuous narrative polish over 2 consecutive ~2,200-char slices covering opening chapters
    nonempty_src = [ln.strip() for ln in lines if ln.strip()]
    slices: list[list[str]] = []
    cur_slice: list[str] = []
    cur_chars = 0
    for s in nonempty_src:
        cur_slice.append(s)
        cur_chars += len(s)
        if cur_chars >= 2200:
            slices.append(cur_slice)
            cur_slice = []
            cur_chars = 0
            if len(slices) >= 2:
                break
    if cur_slice and len(slices) < 2:
        slices.append(cur_slice)

    cont_sys = (
        "下面是同一部小说的连贯正文。其中部分字来自一对一替换密码，格式为 `当前译字[HEX]`（例如 `门[E4E7]口[E534]`）。\n"
        "注意：由于双射错误，可能出现整词错位但单看偏旁像词的情况（例如“一样”应是“一个”、“站辙”应是“知道”、“得底”应是“得到”、“在铮说略”应是“在小说里”、“肉机连断”应是“当机立断”、“蒸样段后”应是“几个月后”、“有人”应是“家人”、“样针”应是“打疫苗/打吊针”）。\n"
        "请逐句通读剧情，找出所有不合语境或仍为 `?[HEX]` 的加密字，输出它们的真实简体汉字。\n"
        "不要避开已占用字！只需输出有把握的纠错 JSON：{\"E4E7\": [\"自\", 0.95], \"E534\": [\"己\", 0.95]}。完全正确的不输出。"
    )
    any_cont_props = False
    for sl in slices:
        rendered_slice = []
        for s in sl:
            if not PUA_RE.search(s):
                rendered_slice.append(s)
            else:
                rendered_slice.append(
                    "".join(
                        f"{cur_map.get(ch, '?')}[{ord(ch):04X}]" if ("\ue000" <= ch <= "\uf8ff") else ch
                        for ch in s
                    )
                )
        try:
            raw3 = llm_call(cont_sys, "\n".join(rendered_slice), verify_model)
            props3 = _parse_kwic_proposals(raw3, valid_hex)
            print(f"[PUA Polish R3] model={verify_model}, proposals={len(props3)}")
        except Exception:
            props3 = {}
        if props3:
            any_cont_props = True
            for hx, cands in props3.items():
                cp = hex_to_cp[hx]
                vmap = votes_by_cp.setdefault(cp, {})
                for idx_c, (ch, conf) in enumerate(cands):
                    w = (4.2 * conf) if idx_c == 0 else (2.1 * conf)
                    vmap[ch] = vmap.get(ch, 0.0) + w
                    # Lower keep inertia for items explicitly flagged wrong in continuous reading
                    if cp in implicit_keep:
                        implicit_keep[cp] = 1.0
    if any_cont_props:
        cand_map_r3 = _solve_pua_matching_round(
            ordered_cps, cur_map, lines_by_cp, counts, votes_by_cp, consensus_bonus,
            jb_bigrams, jb_unigrams, book_bigrams, implicit_keep=implicit_keep,
        )
        if _is_valid_bijection(cand_map_r3):
            coh_r3 = _lexicon_coherence(_apply_map_to_lines(lines, cand_map_r3))
            if coh_r3 >= cur_coh - 0.005 and len(cand_map_r3) >= len(cur_map):
                print(f"  [pua] pass4 r3 continuous polish: mapped={len(cand_map_r3)}/{len(ordered_cps)} coh={cur_coh:.4f}->{coh_r3:.4f}", flush=True)
                cur_map = cand_map_r3
                cur_coh = coh_r3

    new_uncertain: list[str] = []
    prior_by_cp: dict[str, list[str]] = {}
    for item in prior_uncertain or ():
        s_item = str(item).strip()
        m_u = re.match(r"U\+([0-9A-Fa-f]{4,6}):(.+)$", s_item)
        if m_u and m_u.group(2) != "unmapped":
            prior_by_cp.setdefault(chr(int(m_u.group(1), 16)), []).append(s_item)

    for cp in ordered_cps:
        if cp not in cur_map:
            new_uncertain.append(f"U+{ord(cp):04X}:unmapped")
            continue
        has_supporting_vote = votes_by_cp.get(cp, {}).get(cur_map[cp], 0.0) > 0.0
        if cp in prior_by_cp and not has_supporting_vote:
            # Preserve earlier Pass-3 disagreements unless Pass 4 explicitly voted for the retained mapping
            for tag in prior_by_cp[cp]:
                if tag not in new_uncertain:
                    new_uncertain.append(tag)
        if cp not in seed_map and not votes_by_cp.get(cp):
            lm_val = _lm_context_score(lines_by_cp, cp, cur_map[cp], cur_map, jb_bigrams, book_bigrams)
            if lm_val < 0.35:
                tag_low = f"U+{ord(cp):04X}:low_evidence"
                if tag_low not in new_uncertain:
                    new_uncertain.append(tag_low)
    return cur_map, new_uncertain


def _solve_tagged_context_batch(
    lines: list[str],
    batch: list[str],
    counts: Counter[str],
    char_map: dict[str, str],
    sys_prompt: str,
    llm_call: Callable[[str, str, str | None], str],
    model_name: str | None,
    *,
    max_contexts_per_cp: int,
    used_vals_note: str = "",
    include_guess: bool = False,
) -> dict[str, str]:
    """Solve remaining unmapped or uncertain codepoints using mostly-decoded sentence snippets."""
    need = set(batch)
    samples_map: dict[str, list[str]] = {cp: [] for cp in batch}
    for line in lines:
        if not line.strip():
            continue
        present = set(PUA_RE.findall(line)) & need
        if not present:
            continue
        for cp in present:
            if len(samples_map[cp]) >= max_contexts_per_cp:
                continue
            rendered = _render_tag_line(line.strip(), char_map, highlight_cp=cp)
            if len(rendered) > 120:
                pos = rendered.find("⟦?⟧")
                lo = max(0, pos - 48)
                hi = min(len(rendered), pos + 52)
                rendered = rendered[lo:hi]
            samples_map[cp].append(rendered)

    id_to_cp = {f"P{j:03d}": cp for j, cp in enumerate(batch, start=1)}
    prompt_lines: list[str] = []
    if used_vals_note:
        prompt_lines.append(used_vals_note)
    for pid, cp in id_to_cp.items():
        ctxs = samples_map[cp] or ["(无上下文)"]
        if include_guess:
            prompt_lines.append(f"{pid} (当前猜测={char_map.get(cp, '?')}, 出现{counts[cp]}次): {' | '.join(ctxs)}")
        else:
            prompt_lines.append(f"{pid} (出现{counts[cp]}次): {' | '.join(ctxs)}")
    user_prompt = "\n".join(prompt_lines)

    try:
        raw = llm_call(sys_prompt, user_prompt, model_name)
        parsed = _extract_json_dict(raw)
        out: dict[str, str] = {}
        for pid, cp in id_to_cp.items():
            val = str(parsed.get(pid, "")).strip()
            if len(val) == 1 and "\u4e00" <= val <= "\u9fff":
                out[cp] = val
        return out
    except Exception:
        if len(batch) <= 1:
            return {}
        mid = len(batch) // 2
        merged = _solve_tagged_context_batch(
            lines, batch[:mid], counts, char_map, sys_prompt, llm_call, model_name,
            max_contexts_per_cp=max_contexts_per_cp, used_vals_note=used_vals_note, include_guess=include_guess,
        )
        merged.update(
            _solve_tagged_context_batch(
                lines, batch[mid:], counts, char_map, sys_prompt, llm_call, model_name,
                max_contexts_per_cp=max_contexts_per_cp, used_vals_note=used_vals_note, include_guess=include_guess,
            )
        )
        return merged


def _solve_pua_batch(
    lines: list[str],
    batch: list[str],
    counts: Counter[str],
    char_map: dict[str, str],
    sys_prompt: str,
    llm_call: Callable[[str, str, str | None], str],
    model_name: str | None,
    *,
    max_contexts_per_cp: int,
    include_guess: bool = False,
) -> dict[str, str]:
    """Legacy compact snippet solver (used for tiny <=15 PUA inputs)."""
    ctxs = _collect_contexts(lines, batch, char_map, max_contexts_per_cp=max_contexts_per_cp)
    id_to_cp = {f"P{j:03d}": cp for j, cp in enumerate(batch, start=1)}
    prompt_lines = []
    for pid, cp in id_to_cp.items():
        samples = " | ".join(ctxs[cp]) if ctxs[cp] else "(无上下文)"
        if include_guess:
            prompt_lines.append(f"{pid} (当前猜测={char_map.get(cp, '?')}, 出现{counts[cp]}次): {samples}")
        else:
            prompt_lines.append(f"{pid} (出现{counts[cp]}次): {samples}")
    user_prompt = "\n".join(prompt_lines)

    try:
        raw = llm_call(sys_prompt, user_prompt, model_name)
        parsed = _extract_json_dict(raw)
        out: dict[str, str] = {}
        for pid, cp in id_to_cp.items():
            val = str(parsed.get(pid, "")).strip()
            if len(val) == 1 and "\u4e00" <= val <= "\u9fff":
                out[cp] = val
        return out
    except Exception:
        if len(batch) <= 1:
            return {}
        mid = len(batch) // 2
        merged = _solve_pua_batch(
            lines, batch[:mid], counts, char_map, sys_prompt, llm_call, model_name,
            max_contexts_per_cp=max_contexts_per_cp, include_guess=include_guess,
        )
        merged.update(
            _solve_pua_batch(
                lines, batch[mid:], counts, char_map, sys_prompt, llm_call, model_name,
                max_contexts_per_cp=max_contexts_per_cp, include_guess=include_guess,
            )
        )
        return merged


def solve_pua_cipher(
    lines: list[str],
    cache_dir: Path,
    llm_call: Callable[[str, str, str | None], str] | None = None,
    *,
    primary_model: str | None = None,
    secondary_model: str | None = None,
    batch_size: int = 20,
) -> PuaSolveResult:
    """Detect and decode PUA substitution cipher using paragraph-context bijective iterative solving."""
    counts: Counter[str] = Counter()
    lines_by_cp: dict[str, list[str]] = {}
    for line in lines:
        cps = PUA_RE.findall(line)
        if not cps:
            continue
        for ch in cps:
            counts[ch] += 1
        s = line.strip()
        if s:
            for ch in set(cps):
                lines_by_cp.setdefault(ch, []).append(s)

    if not counts:
        coh = _lexicon_coherence(lines)
        return PuaSolveResult(0, 0, {}, [], True, coh, coh, [])

    total_pua = sum(counts.values())
    ordered_cps = [cp for cp, _ in counts.most_common()]
    fp = _fingerprint_codepoints(ordered_cps)
    src_fp = _source_pua_fingerprint(lines, ordered_cps, counts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    exact_cache_path = cache_dir / f"pua_map_{fp}_{src_fp[:8]}.json"
    legacy_cache_path = cache_dir / f"pua_map_{fp}.json"

    coh_before = _lexicon_coherence(lines)
    cand_cache_paths = [exact_cache_path, legacy_cache_path]
    for extra in sorted(cache_dir.glob(f"pua_map_{fp}_*.json")):
        if extra not in cand_cache_paths:
            cand_cache_paths.append(extra)

    legacy_seed_map: dict[str, str] | None = None
    for cache_path in cand_cache_paths:
        if not cache_path.exists():
            continue
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cmap = cached.get("char_map", {})
        if not isinstance(cmap, dict) or not _is_valid_bijection(cmap):
            continue
        covered = sum(1 for cp in ordered_cps if cp in cmap)
        if covered < len(ordered_cps) * 0.95:
            continue
        cached_src_fp = cached.get("source_fingerprint")
        decoded = _apply_map_to_lines(lines, cmap)
        coh_after = _lexicon_coherence(decoded)
        if coh_after <= coh_before + 0.15:
            continue
        if cached_src_fp != src_fp:
            continue
        # Upgrade legacy v1/v2/v3 real-book caches (>= 50 codepoints) through Pass 4 v4 global Hungarian polish
        if len(ordered_cps) > 50 and cached.get("cache_schema_version", 1) < 4 and llm_call is not None:
            legacy_seed_map = dict(cmap)
            continue
        cached_uncertain = [str(x) for x in cached.get("uncertain", [])]
        for cp in ordered_cps:
            if cp not in cmap:
                tag = f"U+{ord(cp):04X}:unmapped"
                if tag not in cached_uncertain:
                    cached_uncertain.append(tag)
        edits = _build_pua_edits(
            lines,
            cmap,
            source="cache:pua_map",
            exclude_cps=_uncertain_codepoints(cached_uncertain),
        )
        return PuaSolveResult(
            total_pua,
            len(ordered_cps),
            cmap,
            edits,
            True,
            coh_before,
            coh_after,
            cached_uncertain,
        )

    if llm_call is None:
        return PuaSolveResult(
            total_pua,
            len(ordered_cps),
            {},
            [],
            False,
            coh_before,
            coh_before,
            [f"U+{ord(c):04X}:unmapped" for c in ordered_cps],
        )

    char_map: dict[str, str] = {}
    used_chars: dict[str, str] = {}  # decoded_char -> cp owner

    def _assign_bijective(proposals: dict[str, str]) -> None:
        """Merge candidate `cp -> ch` bijectively using local jieba word gain & count tie-break."""
        by_char: dict[str, list[str]] = {}
        for cp, ch in proposals.items():
            if len(ch) != 1 or not ("\u4e00" <= ch <= "\u9fff"):
                continue
            by_char.setdefault(ch, []).append(cp)
        for ch, cands in by_char.items():
            existing = used_chars.get(ch)
            pool = list(cands)
            if existing and existing not in pool:
                pool.append(existing)
            if len(pool) == 1:
                winner = pool[0]
            else:
                winner = max(
                    pool,
                    key=lambda c: (_local_jieba_score(lines_by_cp, c, ch, char_map, counts[c]), counts[c]),
                )
            for loser in pool:
                if loser != winner and char_map.get(loser) == ch:
                    char_map.pop(loser, None)
            old_ch = char_map.get(winner)
            if old_ch and old_ch != ch and used_chars.get(old_ch) == winner:
                used_chars.pop(old_ch, None)
            char_map[winner] = ch
            used_chars[ch] = winner

    uncertain: list[str] = []

    # Small synthetic cipher fast path (keeps compatibility with unit tests while strictly enforcing bijection & gate)
    if len(ordered_cps) <= 15:
        legacy_sys = (
            "你是中文小说乱码密码破译专家。小说中有部分汉字被加密字体替换成了私有区占位符。\n"
            "每个编码（如 P001）唯一对应一个现代简体中文汉字。我将给出每个编码在原文中的多条上下文片段（用 ⟦?⟧ 标出待破译字，□ 表示其他尚未破译字）。\n"
            "请根据汉语语法、词组搭配和语境，推断每个编码对应的唯一简体汉字（必须恰好 1 个汉字）。\n"
            "只输出严格 JSON 对象，键为编码名，值为单个汉字，例如：{\"P001\":\"要\",\"P002\":\"我\"}"
        )
        for i in range(0, len(ordered_cps), batch_size):
            batch = ordered_cps[i : i + batch_size]
            proposals = _solve_pua_batch(
                lines, batch, counts, char_map, legacy_sys, llm_call, primary_model, max_contexts_per_cp=4
            )
            _assign_bijective(proposals)
        decoded_lines = _apply_map_to_lines(lines, char_map)
        coh_after = _lexicon_coherence(decoded_lines)
        for cp in ordered_cps:
            if cp not in char_map:
                uncertain.append(f"U+{ord(cp):04X}:unmapped")
        bijective_ok = _is_valid_bijection(char_map)
        gate_passed = bool(char_map) and bijective_ok and (coh_after > coh_before + 0.15)
        if not gate_passed:
            if not bijective_ok:
                uncertain.append("gate_failed:bijection")
            if coh_after <= coh_before + 0.15:
                uncertain.append("gate_failed:coherence")
            return PuaSolveResult(
                total_pua, len(ordered_cps), char_map, [], False, coh_before, coh_after, uncertain
            )
        if len(char_map) >= len(ordered_cps) * 0.95:
            exact_cache_path.write_text(
                json.dumps(
                    {
                        "fingerprint": fp,
                        "source_fingerprint": src_fp,
                        "unique_codepoints": len(ordered_cps),
                        "total_pua": total_pua,
                        "jieba_ratio_before": coh_before,
                        "jieba_ratio_after": coh_after,
                        "char_map": char_map,
                        "codepoint_hex_map": {f"U+{ord(k):04X}": v for k, v in char_map.items()},
                        "uncertain": uncertain,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        edits = _build_pua_edits(lines, char_map, source=f"pua_solver:{primary_model or 'llm'}")
        return PuaSolveResult(total_pua, len(ordered_cps), char_map, edits, False, coh_before, coh_after, uncertain)

    # Full paragraph-context iterative bijective solver
    para_sys = (
        "你是顶尖中文小说替代密码破译专家。下面这段连贯的小说正文中，有部分常用汉字被替换成了 4 位十六进制编码标签（例如 [E4F3]、[E3F2]）。\n"
        "关键规则：同一个方括号标签代表同一个汉字；不同的标签【严格 1 对 1 双射】，绝不可能对应同一个汉字！\n"
        "任务：通读全文语境，结合汉语语法、常见搭配与前文人设剧情，推断当前文本中出现的所有 [XXXX] 标签分别对应的真实简体汉字。\n"
        "严格要求：\n"
        "1. 键是 4 位大写十六进制标签名（不带括号，如 \"E4F3\"），值必须恰好是 1 个现代简体汉字；\n"
        "2. 绝不可把已经在《已占用汉字》列表里的字再分配给新标签；同一批输出里也绝不要重复同一个汉字；\n"
        "3. 只输出你有把握的映射，极个别拿不准的低频生僻字可暂不输出；\n"
        "4. 只输出严格 JSON 对象，不要解释，格式：{\"E4F3\":\"的\",\"E3F2\":\"要\"}"
    )

    hex_to_cp = {f"{ord(cp):04X}": cp for cp in ordered_cps}

    if legacy_seed_map is not None:
        _assign_bijective(legacy_seed_map)
        blocks = []
    else:
        blocks = _select_paragraph_blocks(lines, char_map, max_blocks=16, max_chars_per_block=550)
    votes: dict[str, Counter[str]] = {}
    for b_idx, blk in enumerate(blocks, start=1):
        rendered_blk = "\n".join(_render_tag_line(l, char_map) for l in blk)
        unmapped_in_blk = {m.group(1) for m in re.finditer(r"\[([0-9A-F]{4})\]", rendered_blk)}
        if not unmapped_in_blk:
            continue
        locked_str = "".join(sorted(used_chars))
        usr_p = f"已占用汉字（请勿重复分配给任何新标签）：{locked_str}\n\n正文段落：\n{rendered_blk}"
        try:
            raw = llm_call(para_sys, usr_p, primary_model)
            parsed = _extract_json_dict(raw)
        except Exception:
            parsed = {}
        raw_props: dict[str, str] = {}
        val_owners: dict[str, list[str]] = {}
        for k, v in parsed.items():
            hx = str(k).strip().strip("[]").upper()
            cp = hex_to_cp.get(hx)
            val = str(v).strip()
            if cp and len(val) == 1 and "\u4e00" <= val <= "\u9fff":
                votes.setdefault(cp, Counter())[val] += 1
                if cp not in char_map and val not in used_chars:
                    raw_props[cp] = val
                    val_owners.setdefault(val, []).append(cp)

        # Build coherent non-colliding trial_map for joint word scoring across neighbor tags
        non_colliding = {
            cp: val for cp, val in raw_props.items() if len(val_owners[val]) == 1
        }
        trial_map = {**char_map, **non_colliding}
        for val, cands in val_owners.items():
            if len(cands) > 1:
                best_cp = max(
                    cands,
                    key=lambda c: (_local_jieba_score(lines_by_cp, c, val, trial_map, counts[c]), counts[c]),
                )
                trial_map[best_cp] = val

        high_conf_props: dict[str, str] = {}
        for cp, val in list(trial_map.items()):
            if cp in char_map:
                continue
            if _local_jieba_score(lines_by_cp, cp, val, trial_map, counts[cp]) >= 1.2:
                high_conf_props[cp] = val
        _assign_bijective(high_conf_props)
        if b_idx == 1 or b_idx % 4 == 0 or b_idx == len(blocks):
            print(f"  [pua] pass1 block {b_idx}/{len(blocks)} locked={len(char_map)}/{len(ordered_cps)}", flush=True)

    # Consolidate remaining Pass 1 votes globally (evaluated jointly over top-voted trial map)
    top_votes: dict[str, str] = {}
    for cp in ordered_cps:
        if cp not in char_map and cp in votes:
            val, _ = votes[cp].most_common(1)[0]
            if val not in used_chars:
                top_votes[cp] = val
    trial_final = {**char_map, **top_votes}
    best_remaining: dict[str, str] = {}
    for cp in ordered_cps:
        if cp in char_map or cp not in votes:
            continue
        ranked = sorted(
            votes[cp].items(),
            key=lambda kv: (kv[1], _local_jieba_score(lines_by_cp, cp, kv[0], trial_final, counts[cp])),
            reverse=True,
        )
        for val, vcnt in ranked:
            if val not in used_chars:
                best_remaining[cp] = val
                break
    _assign_bijective(best_remaining)

    # Pass 2: Hole-filling for remaining codepoints in near-plaintext sentence contexts
    hole_sys = (
        "你是中文小说乱码破译专家。文章中 90% 以上的汉字已经还原，仅剩少量汉字仍标注为 ⟦?⟧。\n"
        "请根据上下文词组搭配，推断每个 P 编号处的唯一简体汉字（绝不可重复已占用的汉字！）。\n"
        "只输出严格 JSON 对象，格式：{\"P001\":\"字\"}"
    )
    hole_batch = min(batch_size, 15)
    for _round in range(2 if legacy_seed_map is None else 0):
        missing_cps = [cp for cp in ordered_cps if cp not in char_map]
        if not missing_cps:
            break
        for i in range(0, len(missing_cps), hole_batch):
            batch = missing_cps[i : i + hole_batch]
            locked_note = f"已占用汉字（严禁重复）：{''.join(sorted(used_chars))}"
            solved = _solve_tagged_context_batch(
                lines,
                batch,
                counts,
                char_map,
                hole_sys,
                llm_call,
                primary_model,
                max_contexts_per_cp=5,
                used_vals_note=locked_note,
            )
            filtered = {cp: ch for cp, ch in solved.items() if ch not in used_chars}
            if not filtered:
                # Allow collision competition if strict unused returned empty
                filtered = solved
            _assign_bijective(filtered)

    # Pass 3: Secondary cross-check on any remaining unmapped or low-coherence mappings
    if legacy_seed_map is None and secondary_model:
        suspects = [
            cp
            for cp in ordered_cps
            if cp not in char_map
            or _local_jieba_score(lines_by_cp, cp, char_map[cp], char_map, counts[cp]) < 0.8
            or counts[cp] <= 2
        ]
        print(f"  [pua] pass3 secondary ({secondary_model}) checking {len(suspects)} suspects; mapped={len(char_map)}/{len(ordered_cps)}", flush=True)
        if suspects:
            for i in range(0, len(suspects), hole_batch):
                batch = suspects[i : i + hole_batch]
                locked_note = f"其他已锁定汉字（非必要勿碰撞）：{''.join(sorted(used_chars))}"
                solved2 = _solve_tagged_context_batch(
                    lines,
                    batch,
                    counts,
                    char_map,
                    hole_sys,
                    llm_call,
                    secondary_model,
                    max_contexts_per_cp=5,
                    used_vals_note=locked_note,
                    include_guess=True,
                )
                for cp, val2 in solved2.items():
                    old_val = char_map.get(cp)
                    if not old_val and val2 not in used_chars:
                        _assign_bijective({cp: val2})
                    elif val2 and val2 != old_val:
                        s_old = _local_jieba_score(lines_by_cp, cp, old_val or "", char_map, counts[cp]) if old_val else 0.0
                        s_new = _local_jieba_score(lines_by_cp, cp, val2, char_map, counts[cp])
                        if val2 not in used_chars and s_new > s_old + 0.15:
                            if old_val and used_chars.get(old_val) == cp:
                                used_chars.pop(old_val, None)
                            char_map[cp] = val2
                            used_chars[val2] = cp
                        else:
                            uncertain.append(f"U+{ord(cp):04X}:{old_val or '?'}/{val2}")

    for cp in ordered_cps:
        if cp not in char_map:
            tag = f"U+{ord(cp):04X}:unmapped"
            if tag not in uncertain:
                uncertain.append(tag)

    # Pass 4: Global unconstrained KWIC + Hungarian bipartite polish to break bijective traps (real ciphers)
    if len(ordered_cps) > 50 and char_map:
        char_map, uncertain = _polish_pua_map_globally(
            lines,
            ordered_cps,
            char_map,
            lines_by_cp,
            counts,
            llm_call,
            primary_model=primary_model,
            secondary_model=secondary_model,
            prior_uncertain=uncertain,
        )

    decoded_lines = _apply_map_to_lines(lines, char_map)
    coh_after = _lexicon_coherence(decoded_lines)
    bijective_ok = _is_valid_bijection(char_map)
    gate_passed = bool(char_map) and bijective_ok and (coh_after > coh_before + 0.15)

    if not gate_passed:
        if not bijective_ok:
            uncertain.append("gate_failed:bijection")
        if coh_after <= coh_before + 0.15:
            uncertain.append("gate_failed:coherence")
        return PuaSolveResult(
            total_pua,
            len(ordered_cps),
            char_map,
            [],
            False,
            coh_before,
            coh_after,
            uncertain,
        )

    if len(char_map) >= len(ordered_cps) * 0.95:
        exact_cache_path.write_text(
            json.dumps(
                {
                    "cache_schema_version": 4,
                    "fingerprint": fp,
                    "source_fingerprint": src_fp,
                    "unique_codepoints": len(ordered_cps),
                    "total_pua": total_pua,
                    "jieba_ratio_before": coh_before,
                    "jieba_ratio_after": coh_after,
                    "char_map": char_map,
                    "codepoint_hex_map": {f"U+{ord(k):04X}": v for k, v in char_map.items()},
                    "uncertain": uncertain,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    edits = _build_pua_edits(
        lines,
        char_map,
        source=f"pua_solver:{primary_model or 'llm'}",
        exclude_cps=_uncertain_codepoints(uncertain),
    )
    return PuaSolveResult(
        total_pua,
        len(ordered_cps),
        char_map,
        edits,
        False,
        coh_before,
        coh_after,
        uncertain,
    )


def _uncertain_codepoints(uncertain: list[str]) -> set[str]:
    out: set[str] = set()
    for item in uncertain:
        m = re.match(r"U\+([0-9A-Fa-f]{4,6})\b", str(item).strip())
        if m:
            out.add(chr(int(m.group(1), 16)))
    return out


def _build_pua_edits(
    lines: list[str],
    char_map: dict[str, str],
    *,
    source: str,
    exclude_cps: set[str] | None = None,
) -> list[EditOp]:
    excluded = exclude_cps or set()
    edits: list[EditOp] = []
    for line_num, line in enumerate(lines, start=1):
        if not PUA_RE.search(line):
            continue
        for i, ch in enumerate(line):
            if ch in char_map and ch not in excluded:
                edits.append(
                    EditOp(
                        line=line_num,
                        start=i,
                        end=i + 1,
                        op="sub",
                        before=ch,
                        after=char_map[ch],
                        tier="pua",
                        reason=f"pua_decode:U+{ord(ch):04X}",
                        source=source,
                        status="approved",
                    )
                )
    return edits
