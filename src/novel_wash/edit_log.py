"""Immutable character-level edit log, replay verifier, and dual-edition renderers."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CHAPTER_HEADING_RE = re.compile(
    r"^[ \t\u3000]*(?:"
    r"(?:\d{1,5}[.．、]\s*)?第\s*[零〇一二三四五六七八九十百千万两\d]+\s*章[^\r\n]{0,46}|"
    r"第\s*[零〇一二三四五六七八九十百千万两\d]+\s*[节回卷集部篇](?:$|[\s：:·\-—][^\r\n]{1,36})|"
    r"(?:序章|楔子|尾声|番外|后记|完结感言|上架感言)(?:$|[：:\s·\-—][^\r\n]{0,40})"
    r")$"
)

_UNIT_HEADING_RE = re.compile(r"^第\s*[零〇一二三四五六七八九十百千万两\d]+\s*[节回卷集部篇](.*)$")
_CHAPTER_NUM_HEADING_RE = re.compile(r"^(?:\d{1,5}[.．、]\s*)?第\s*[零〇一二三四五六七八九十百千万两\d]+\s*章(.*)$")
_CHAPTER_PREFIX_ONLY_RE = re.compile(
    r"^[ \t\u3000]*(?:(?:\d{1,5}[.．、]\s*)?第\s*[零〇一二三四五六七八九十百千万两\d]+\s*章|第\s*[零〇一二三四五六七八九十百千万两\d]+\s*[节回卷集部篇]|(?:序章|楔子|尾声|番外|后记|完结感言|上架感言))"
)
_NUMBERED_CAND_RE = re.compile(r"^\s*(\d{1,4})\s*([.．、])\s*(?![\d％%])(\S[^\r\n]{0,48})\s*$")
_VOLUME_MARKER_RE = re.compile(r"^(?:第\s*[零〇一二三四五六七八九十百千万两\d]+\s*卷|卷\s*[零〇一二三四五六七八九十\d]+)(?:$|[\s：:·\-—])")
_SPECIAL_HEADING_RE = re.compile(r"^(?:序章|楔子|尾声|番外|后记|完结感言|上架感言)(.*)$")
_VALID_EDIT_OPS = {"del", "sub", "ins", "clear_line"}


def chapter_heading_prefix_end(line: str, *, allow_long_dirty: bool = False) -> int:
    """Return the end offset of a chapter heading prefix (e.g. `第216章`) in `line`, or 0 if not a heading."""
    s = line.strip()
    if not s:
        return 0
    if not allow_long_dirty:
        if not is_chapter_heading(s):
            return 0
    elif len(s) > 120:
        return 0
    m_ch = _CHAPTER_NUM_HEADING_RE.match(s)
    if m_ch:
        tail = m_ch.group(1)
        if tail:
            if s.endswith("。") and not allow_long_dirty:
                return 0
            if re.match(r"^(?:也|已经|都|就|是|的|里|里面|之后|过后|才|只|还|并|终于|说完|写完|结束)", tail) and s.endswith(("。", "！", "？")):
                return 0
            if re.match(r"^[ \t\u3000]*[：:][ \t\u3000]*(?![ \t\u3000]*第\s*[零〇一二三四五六七八九十百千万两\d]+\s*章).*。$", tail):
                return 0
            if re.match(r"^[\s：:]+(?:他|她|它|我|你|这|那|也|已经|都).*?[，,].*。$", tail):
                return 0
            if len(s) > 28 and "，" in s and s.endswith("。") and not allow_long_dirty:
                return 0
    elif not is_chapter_heading(s):
        return 0
    mp = _CHAPTER_PREFIX_ONLY_RE.match(line)
    return mp.end() if mp else 0


def detect_numbered_scheme(lines: list[str]) -> set[int] | None:
    """Detect book-adaptive `1. 标题` / `1、标题` chapter headings via longest monotonic chain DP.

    Returns a set of 0-based line indices into `lines` belonging to the validated heading chain,
    or `None` when the book does not use numbered chapter headings.
    """
    std_count = 0
    total_nonblank = 0
    # Compute cumulative body chars and nonblank lines per line index
    cum_chars: list[int] = [0] * (len(lines) + 1)
    cum_nb: list[int] = [0] * (len(lines) + 1)
    by_sep: dict[str, list[tuple[int, int, str]]] = {}
    vol_indices: list[int] = []

    running = 0
    for idx, raw in enumerate(lines):
        s = raw.strip()
        cum_chars[idx] = running
        cum_nb[idx] = total_nonblank
        if s:
            total_nonblank += 1
            if is_chapter_heading(s):
                std_count += 1
            if _VOLUME_MARKER_RE.match(s):
                vol_indices.append(idx)
            m = _NUMBERED_CAND_RE.match(s)
            if m and len(s) <= 48:
                num = int(m.group(1))
                sep = m.group(2)
                title = m.group(3).strip()
                # Reject countdowns/decimals/list-like sentences (e.g. `3、2、1开战！`, `1. 他说道……。`)
                if (
                    1 <= num <= 5000
                    and not re.search(r"^\d+\s*[、.．]", title)
                    and not (len(s) > 24 and "，" in title and title.endswith("。"))
                    and not (sep == "." and re.match(r"^(?:首先|其次|最后|第一|第二|第三)[：:，,、\s]", title))
                ):
                    by_sep.setdefault(sep, []).append((idx, num, title))
            running += len(s)
    cum_chars[len(lines)] = running
    cum_nb[len(lines)] = total_nonblank

    if not by_sep:
        return None

    vol_set_prefix = vol_indices  # sorted list of 0-based volume line indices
    best_overall: list[int] = []
    best_sep_total = 0
    best_first_num = 9999

    for sep, cands in by_sep.items():
        if len(cands) < 5:
            continue
        m_len = len(cands)
        dp = [1] * m_len
        prev = [-1] * m_len
        start_bonus = [1 if num <= 5 else 0 for _, num, _ in cands]

        vol_ptr = 0
        # For each candidate pair within lookback window, link monotonic chapters separated by body prose
        for j in range(m_len):
            idx_j, num_j, _ = cands[j]
            # Look back at most 20 candidates (more than enough even with noisy lists)
            for i in range(j - 1, max(-1, j - 25), -1):
                idx_i, num_i, _ = cands[i]
                body_chars = cum_chars[idx_j] - cum_chars[idx_i] - len(lines[idx_i].strip())
                if body_chars < 150:
                    continue
                is_next = (num_j == num_i + 1) or (1 <= num_j - num_i <= 4)
                is_vol_reset = False
                if not is_next and num_j == 1 and num_i >= 3:
                    # Allow reset to 1 when a volume marker line appears between idx_i and idx_j
                    is_vol_reset = any(idx_i < v < idx_j for v in vol_set_prefix)
                if not (is_next or is_vol_reset):
                    continue
                step_gain = 2 if num_j == num_i + 1 else 1
                cand_score = dp[i] + step_gain + (start_bonus[i] if prev[i] == -1 else 0)
                if cand_score > dp[j]:
                    dp[j] = cand_score
                    prev[j] = i

        end_j = max(range(m_len), key=lambda k: dp[k])
        chain_pos: list[int] = []
        cur = end_j
        while cur >= 0:
            chain_pos.append(cur)
            cur = prev[cur]
        chain_pos.reverse()

        # Trim head if DP didn't include start bonus? Prefer actual chain length >= 5
        chain_line_indices = [cands[p][0] for p in chain_pos]
        if len(chain_line_indices) > len(best_overall):
            best_overall = chain_line_indices
            best_sep_total = len(cands)
            best_first_num = cands[chain_pos[0]][1]

    min_len = 5 if total_nonblank < 2000 else 10
    if len(best_overall) < min_len:
        return None
    if std_count >= 2:
        if len(best_overall) < max(20, 4 * std_count):
            return None
        line_gaps = [cum_nb[best_overall[k + 1]] - cum_nb[best_overall[k]] for k in range(len(best_overall) - 1)]
        char_gaps = [
            cum_chars[best_overall[k + 1]] - cum_chars[best_overall[k]] - len(lines[best_overall[k]].strip())
            for k in range(len(best_overall) - 1)
        ]
        if not line_gaps or statistics.median(line_gaps) < 10 or statistics.median(char_gaps) < 500:
            return None
    if best_first_num > 5:
        return None
    ratio = len(best_overall) / max(1, best_sep_total)
    if ratio < 0.65 and len(best_overall) < 40:
        return None
    return set(best_overall)


def is_chapter_heading(line: str) -> bool:
    """Return True if a standalone line is a genuine volume/chapter heading rather than prose."""
    s = line.strip()
    if not s or len(s) > 50:
        return False
    m_ch = _CHAPTER_NUM_HEADING_RE.match(s)
    if m_ch:
        tail = m_ch.group(1)
        if not tail:
            return True
        has_sep = bool(re.match(r"^[\s：:·\-—（(【\[“\"『「0-9]", tail))
        if not has_sep:
            if s.endswith("。"):
                return False
            if re.match(
                r"^(?:也|已经|都|就|是|的|里|里面|之后|过后|才|只|还|并|终于|说完|写完|结束)",
                tail,
            ) and s.endswith(("。", "！", "？")):
                return False
        if re.match(r"^[ \t\u3000]*[：:][ \t\u3000]*(?![ \t\u3000]*第[零〇一二三四五六七八九十百千万两\d]+章).*。$", tail):
            return False
        if re.match(r"^[\s：:]+(?:他|她|它|我|你|这|那|也|已经|都).*?[，,].*。$", tail):
            return False
        if len(s) > 28 and "，" in s and s.endswith("。"):
            return False
        return True
    m_unit = _UNIT_HEADING_RE.match(s)
    if m_unit:
        tail = m_unit.group(1)
        if not tail:
            return True
        if not re.match(r"^[\s：:·\-—]+\S", tail):
            return False
        if len(s) > 36 or s.endswith("。"):
            return False
        return True
    m_sp = _SPECIAL_HEADING_RE.match(s)
    if m_sp:
        tail = m_sp.group(1)
        if not tail:
            return True
        if not re.match(r"^[：:\s·\-—]", tail):
            return False
        if len(s) > 45:
            return False
        if s.endswith("。") and ("，" in tail or len(s) > 20 or ("：" not in tail and ":" not in tail)):
            return False
        return True
    return False


@dataclass(frozen=True)
class EditOp:
    """Single character-span operation anchored to the original line."""

    line: int  # 1-based physical line index
    start: int  # 0-based inclusive start offset in original line
    end: int  # 0-based exclusive end offset in original line
    op: str  # "del" | "sub" | "ins" | "clear_line"
    before: str
    after: str
    tier: str  # "encoding" | "pua" | "soup" | "ad" | "rule_sym" | "rule_tier_r" | "llm_tier_n" | "llm_tier_c" | "harmony"
    reason: str
    source: str
    status: str = "approved"  # "approved" | "review" | "rejected"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditOp":
        return cls(**data)


@dataclass
class BookEditLog:
    """Complete auditable patch log for a single book."""

    source_file: str
    source_sha256: str
    encoding: str
    line_count: int
    edits: list[EditOp] = field(default_factory=list)
    review_queue: list[dict[str, Any]] = field(default_factory=list)
    proper_nouns: list[str] = field(default_factory=list)
    pua_map_info: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    cost_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "encoding": self.encoding,
            "line_count": self.line_count,
            "proper_nouns": self.proper_nouns,
            "pua_map_info": self.pua_map_info,
            "stats": self.stats,
            "cost_summary": self.cost_summary,
            "edits": [e.to_dict() for e in self.edits],
            "review_queue": self.review_queue,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "BookEditLog":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            source_file=data["source_file"],
            source_sha256=data["source_sha256"],
            encoding=data["encoding"],
            line_count=data["line_count"],
            proper_nouns=data.get("proper_nouns", []),
            pua_map_info=data.get("pua_map_info", {}),
            stats=data.get("stats", {}),
            cost_summary=data.get("cost_summary", {}),
            edits=[EditOp.from_dict(e) for e in data.get("edits", [])],
            review_queue=data.get("review_queue", []),
        )


def _normalize_line_edits(orig_line: str, line_edits: list[EditOp], *, include_review: bool) -> list[EditOp]:
    allowed = {"approved", "review"} if include_review else {"approved"}
    active = [e for e in line_edits if e.status in allowed]
    if not active:
        return []

    for e in active:
        if e.op not in _VALID_EDIT_OPS:
            raise ValueError(f"Unsupported edit op {e.op!r} at line {e.line}")
        if "\n" in e.after or "\r" in e.after:
            raise ValueError(f"Line break not allowed in edit replacement at line {e.line}")
        if e.start < 0 or e.end > len(orig_line) or e.start > e.end:
            raise ValueError(f"Out-of-bounds edit at line {e.line}: [{e.start}, {e.end}) vs len {len(orig_line)}")

    approved_clears = [e for e in active if e.op == "clear_line" and e.status == "approved"]
    if approved_clears:
        clear = approved_clears[0]
        if clear.start != 0 or clear.end != len(orig_line) or clear.before != orig_line or clear.after != "":
            raise ValueError(f"Invalid clear_line op at line {clear.line}: bounds or text mismatch")
        return [clear]

    non_clear_approved = [e for e in active if e.op != "clear_line" and e.status == "approved"]
    review_clears = [e for e in active if e.op == "clear_line" and e.status == "review"]
    if review_clears and not non_clear_approved:
        clear = review_clears[0]
        if clear.start != 0 or clear.end != len(orig_line) or clear.before != orig_line or clear.after != "":
            raise ValueError(f"Invalid clear_line op at line {clear.line}: bounds or text mismatch")
        return [clear]

    ordered = sorted((e for e in active if e.op != "clear_line"), key=lambda e: (e.start, e.end))
    prev_end = 0
    for e in ordered:
        if e.start < prev_end:
            raise ValueError(f"Overlapping edits at line {e.line}: {prev_end} > {e.start}")
        actual = orig_line[e.start : e.end]
        if actual != e.before:
            raise ValueError(
                f"Source mismatch at line {e.line} [{e.start}:{e.end}]: expected {e.before!r}, got {actual!r}"
            )
        prev_end = e.end
    return ordered


def apply_edits_to_line(
    orig_line: str,
    line_edits: list[EditOp],
    *,
    include_review: bool = False,
    _normalized: list[EditOp] | None = None,
) -> str:
    """Apply non-overlapping edits to a single original line and verify untouched spans (G4)."""
    ordered = (
        _normalized
        if _normalized is not None
        else _normalize_line_edits(orig_line, line_edits, include_review=include_review)
    )
    if not ordered:
        return orig_line
    if len(ordered) == 1 and ordered[0].op == "clear_line":
        return ""

    parts: list[str] = []
    untouched_slices: list[str] = []
    cursor = 0
    for e in ordered:
        untouched = orig_line[cursor : e.start]
        untouched_slices.append(untouched)
        parts.append(untouched)
        parts.append(e.after)
        cursor = e.end
    tail = orig_line[cursor:]
    untouched_slices.append(tail)
    parts.append(tail)

    result = "".join(parts)

    # G4 Replay invariant: re-read untouched intervals from result and assert exact equality.
    r_cursor = 0
    for idx, e in enumerate(ordered):
        expected_untouched = untouched_slices[idx]
        got_untouched = result[r_cursor : r_cursor + len(expected_untouched)]
        if got_untouched != expected_untouched:
            raise AssertionError(
                f"G4 replay invariant violated at line {e.line}: untouched prefix mismatch"
            )
        r_cursor += len(expected_untouched)
        got_after = result[r_cursor : r_cursor + len(e.after)]
        if got_after != e.after:
            raise AssertionError(f"G4 replay invariant violated at line {e.line}: edited span mismatch")
        r_cursor += len(e.after)
    if result[r_cursor:] != untouched_slices[-1]:
        raise AssertionError(f"G4 replay invariant violated at line {ordered[-1].line}: untouched tail mismatch")
    return result


def apply_and_verify(
    original_lines: list[str],
    edits: list[EditOp],
    *,
    include_review: bool = False,
) -> tuple[list[str], set[int]]:
    """Apply edits across all lines and return (rendered_lines, cleared_line_numbers)."""
    n_lines = len(original_lines)
    by_line: dict[int, list[EditOp]] = {}
    for e in edits:
        if e.line < 1 or e.line > n_lines:
            raise ValueError(f"Edit line {e.line} out of range [1, {n_lines}]")
        by_line.setdefault(e.line, []).append(e)

    out_lines: list[str] = []
    cleared_lines: set[int] = set()

    for idx, orig in enumerate(original_lines, start=1):
        line_ops = by_line.get(idx)
        if not line_ops:
            out_lines.append(orig)
            continue
        norm = _normalize_line_edits(orig, line_ops, include_review=include_review)
        has_clear = any(e.op == "clear_line" for e in norm)
        rendered = apply_edits_to_line(orig, line_ops, include_review=include_review, _normalized=norm)
        if has_clear:
            cleared_lines.add(idx)
            out_lines.append("")
        else:
            if orig.strip() != "" and rendered.strip() == "":
                cleared_lines.add(idx)
            out_lines.append(rendered)

    if len(out_lines) != len(original_lines):
        raise AssertionError("Physical line count changed during apply_and_verify")
    return out_lines, cleared_lines


def render_ai_edition(
    original_lines: list[str],
    edits: list[EditOp],
    *,
    include_review: bool = False,
    line_ending: str = "\n",
) -> str:
    """Render AI scan edition: strictly preserves 1-to-1 physical line numbers."""
    out_lines, _ = apply_and_verify(original_lines, edits, include_review=include_review)
    return line_ending.join(out_lines) + (line_ending if original_lines else "")


def render_reader_edition(
    original_lines: list[str],
    edits: list[EditOp],
    *,
    include_review: bool = False,
) -> str:
    """Render clean reader edition: collapses emptied junk lines & normalizes spacing."""
    out_lines, cleared_lines = apply_and_verify(original_lines, edits, include_review=include_review)
    numbered_indices = detect_numbered_scheme(out_lines) or set()
    cleaned: list[str] = []
    blank_run = 0
    for idx, line in enumerate(out_lines, start=1):
        if idx in cleared_lines:
            continue
        stripped = line.rstrip()
        if not stripped:
            blank_run += 1
            if blank_run <= 1 and cleaned:
                cleaned.append("")
            continue
        blank_run = 0
        # Normalize leading ASCII spaces on headings cleanly; preserve fullwidth paragraph indents
        if is_chapter_heading(stripped) or (idx - 1) in numbered_indices:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            cleaned.append(stripped.lstrip(" \t\u3000"))
            cleaned.append("")
            blank_run = 1
        else:
            cleaned.append(stripped)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return "\n".join(cleaned) + ("\n" if cleaned else "")
