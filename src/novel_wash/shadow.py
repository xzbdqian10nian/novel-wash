"""Shadow text normalization with exact character-span mapping back to original text.

Normalization (OpenCC Trad->Simp and fullwidth ASCII digits/letters) is ONLY used on
the shadow string for detection/matching. All edits are projected back to exact offsets
in the original line so fullwidth Chinese punctuation and original formatting are never
accidentally corrupted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from opencc import OpenCC

_OPENCC_T2S = OpenCC("t2s")


@lru_cache(maxsize=16384)
def _norm_char(ch: str) -> str:
    code = ord(ch)
    # Fullwidth 0-9, A-Z, a-z -> ASCII (keep fullwidth Chinese punctuation untouched)
    if 0xFF10 <= code <= 0xFF19 or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
        return chr(code - 0xFEE0)
    if ch == "\u3000":
        return " "
    if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
        converted = _OPENCC_T2S.convert(ch)
        return converted if converted else ch
    return ch


@dataclass(frozen=True)
class ShadowLine:
    orig: str
    text: str
    span_map: tuple[tuple[int, int], ...]

    def to_orig_span(self, shadow_start: int, shadow_end: int) -> tuple[int, int]:
        if shadow_start < 0 or shadow_end > len(self.text) or shadow_start >= shadow_end:
            raise ValueError(f"Invalid shadow span [{shadow_start}, {shadow_end}) for len {len(self.text)}")
        orig_start = self.span_map[shadow_start][0]
        orig_end = self.span_map[shadow_end - 1][1]
        return orig_start, orig_end


def build_shadow(orig: str) -> ShadowLine:
    """Build a detection shadow string preserving an exact char->orig span map."""
    out_chars: list[str] = []
    spans: list[tuple[int, int]] = []
    for idx, ch in enumerate(orig):
        mapped = _norm_char(ch)
        for mch in mapped:
            out_chars.append(mch)
            spans.append((idx, idx + 1))
    return ShadowLine(orig=orig, text="".join(out_chars), span_map=tuple(spans))
