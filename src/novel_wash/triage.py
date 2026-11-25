"""Fast novel corpus health triage (encoding, PUA, group soup, inline noise, yueyi blocks, harmony slots)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .encoding import DecodedFile, read_novel_file

CJK_CLASS = r"\u4e00-\u9fff\u3400-\u4dbf"
CIRCLED_RE = re.compile(r"[①-⓿㈠-㈩㊀-㊰⑴-⒇⒜-ⓩ❶-❿⒈-⒛➀-➓]")
HARD_BETWEEN_CJK_RE = re.compile(rf"(?<=[{CJK_CLASS}])[\^\]\}}\{{\[|\\<>`$#]+(?=[{CJK_CLASS}])")
LATIN_BETWEEN_CJK_RE = re.compile(rf"(?<=[{CJK_CLASS}])[a-z](?=[{CJK_CLASS}])")
PUA_RE = re.compile(r"[\ue000-\uf8ff]")
HARMONY_RE = re.compile(r"\*{2,}|[〇○]{2}")
AD_RE = re.compile(r"笔趣阁|搜书吧|顶点小说|飘天文学|尚香书院|记住网址|发布页[:：]|m\.lingdian", re.I)
YUEYI_ANCHOR_RE = re.compile(
    r"(?:[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad\u0337\u0370-\u03ff\u0488\u0489"
    r"\u0500-\u05ff\u10a0-\u10ff\u1200-\u16ff\u19d0-\u19df\u1d00-\u1fff\u2113\u212f\u2160-\u2188"
    r"\u2205\u2206\u2373\u2474-\u24ff\u2776-\u2793\u3021-\u302f\u3036\u303b\ua720-\uabbf"
    r"\U00010280-\U0001034f\U0001d400-\U0001d7ff]|"
    r"(?<![\u0400-\u04ff])[\u0400-\u04ff]{1,2}(?![\u0400-\u04ff]))"
)
TAG_LIST_LINE_RE = re.compile(rf"^(?:[{CJK_CLASS}\w？?…]+(?:\|[{CJK_CLASS}\w？?…]+){{2,}})$")
SAFE_LATIN_PATTERNS = re.compile(r"(?:T恤|S形|X光|san值|SAN值|维他命[A-Za-z]|[A-Z]级|[A-Z]超|[A-Z]型)")


@dataclass(frozen=True)
class TriageResult:
    file: str
    mb: float
    encoding: str
    is_mixed_encoding: bool
    lines: int
    pua_count: int
    pua_unique: int
    repl_count: int
    circled_lines: int
    hard_cjk_lines: int
    latin_cjk_lines: int
    yueyi_block_lines: int
    harmony_slots: int
    ad_lines: int
    grade: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def triage_decoded(decoded: DecodedFile, *, file_label: str = "", raw_size: int = 0) -> TriageResult:
    text = decoded.text
    pua_matches = PUA_RE.findall(text)
    pua_count = len(pua_matches)
    pua_unique = len(set(pua_matches))
    repl_count = text.count("\ufffd")
    harmony_slots = len(HARMONY_RE.findall(text))

    circled_lines = 0
    hard_cjk_lines = 0
    latin_cjk_lines = 0
    yueyi_block_lines = 0
    ad_lines = 0

    for line in decoded.lines:
        s = line.strip()
        if not s:
            continue
        if CIRCLED_RE.search(s):
            circled_lines += 1
        if YUEYI_ANCHOR_RE.search(s):
            yueyi_block_lines += 1
        if HARD_BETWEEN_CJK_RE.search(s) and not TAG_LIST_LINE_RE.match(s):
            # Ignore innocent [正常标签/弹幕/打赏] bracket spans when no other noise exists
            cleaned_brackets = re.sub(r"\[[^\[\]\r\n]{1,20}\]", "", s)
            if HARD_BETWEEN_CJK_RE.search(cleaned_brackets):
                hard_cjk_lines += 1
        s_safe_lat = SAFE_LATIN_PATTERNS.sub("", s)
        if LATIN_BETWEEN_CJK_RE.search(s_safe_lat):
            latin_cjk_lines += 1
        if AD_RE.search(s):
            ad_lines += 1

    if pua_count > 50:
        grade = "pua_cipher"
    elif yueyi_block_lines >= 5 or circled_lines >= 30 and hard_cjk_lines >= 15:
        grade = "heavy_soup_block"
    elif hard_cjk_lines >= 50 or latin_cjk_lines >= 100:
        grade = "heavy_inline"
    elif circled_lines >= 30:
        grade = "heavy_soup"
    elif hard_cjk_lines >= 5 or circled_lines >= 5 or ad_lines >= 5 or harmony_slots >= 20:
        grade = "moderate"
    elif decoded.encoding not in ("utf-8", "utf-8-sig"):
        grade = "encoding_only"
    else:
        grade = "clean"

    return TriageResult(
        file=file_label,
        mb=round(raw_size / (1024 * 1024), 2),
        encoding=decoded.encoding,
        is_mixed_encoding=decoded.is_mixed,
        lines=len(decoded.lines),
        pua_count=pua_count,
        pua_unique=pua_unique,
        repl_count=repl_count,
        circled_lines=circled_lines,
        hard_cjk_lines=hard_cjk_lines,
        latin_cjk_lines=latin_cjk_lines,
        yueyi_block_lines=yueyi_block_lines,
        harmony_slots=harmony_slots,
        ad_lines=ad_lines,
        grade=grade,
    )


def triage_file(path: Path) -> TriageResult:
    raw_size = path.stat().st_size
    decoded = read_novel_file(path)
    return triage_decoded(decoded, file_label=str(path), raw_size=raw_size)


def triage_paths(paths: list[Path]) -> list[TriageResult]:
    return [triage_file(p) for p in paths]
