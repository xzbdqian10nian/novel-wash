"""Read-only political sensitivity & adult fadeout/omission scanner (zero LLM cost)."""

from __future__ import annotations

import html
import importlib.resources as pkg_resources
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from .encoding import read_novel_file
from .epub_fix import _decode_xml_bytes

_MODERN_YEAR_RE = re.compile(r"(?<!\d)(?:19\d{2}|20[0-2]\d)(?!\d)")
_ELLIPSIS_LINE_RE = re.compile(r"(?:……|\.{6,}|——)")
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_XHTML_BLOCK_TAG_RE = re.compile(
    r'''</?(?:[^\s:/>=<]+:)?(?:p|div|br|li|ul|ol|h[1-6]|tr|td|th|blockquote|section|article|header|footer|pre|hr)(?=[\s/>])(?:[^>"']|"[^"]*"|'[^']*')*>''',
    re.I,
)
_XHTML_STRIP_CONTAINER_RE = re.compile(
    r'''<((?:[^\s:/>=<]+:)?(?:head|script|style))(?=[\s>])(?:[^>"']|"[^"]*"|'[^']*')*(?<!/)>.*?</\1\s*>''',
    re.I | re.S,
)
_XHTML_ANY_TAG_RE = re.compile(r"<[^>]+>")


def _compile_alt(terms: list[str]) -> re.Pattern[str] | None:
    cleaned = sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)
    if not cleaned:
        return None
    return re.compile("|".join(re.escape(t) for t in cleaned))


def load_policy_terms(terms_path: Path | None = None) -> dict[str, list[str]]:
    """Load local `policy_terms.json`, else `policy_terms.example.json`, else bundled package data."""
    if terms_path is not None:
        if not terms_path.exists():
            raise FileNotFoundError(f"Policy terms file not found: {terms_path}")
        data = json.loads(terms_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Policy terms file must contain a JSON object: {terms_path}")
        validated: dict[str, list[str]] = {}
        for k, v in data.items():
            if not isinstance(v, list) or not all(isinstance(item, str) for item in v):
                raise ValueError(f"Policy terms field '{k}' must be a list of strings: {terms_path}")
            validated[k] = list(v)
        return validated

    candidates = [_DEFAULT_CONFIG_DIR / "policy_terms.json", _DEFAULT_CONFIG_DIR / "policy_terms.example.json"]
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: list(v) for k, v in data.items() if isinstance(v, list)}
    raw = pkg_resources.files("novel_wash.data").joinpath("policy_terms.default.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, dict):
        return {k: list(v) for k, v in data.items() if isinstance(v, list)}
    return {}


def _scan_sections_policy(
    sections: list[list[str]],
    *,
    terms_path: Path | None = None,
    include_fadeout_hints: bool = True,
    max_samples: int = 5,
) -> dict[str, Any]:
    """Scan line sections read-only without bleeding context windows across separate files/EPUB members."""
    terms = load_policy_terms(terms_path)
    tier_a_re = _compile_alt(terms.get("tier_a_exact", []))
    tier_b_re = _compile_alt(terms.get("tier_b_contextual", []))
    modern_re = _compile_alt(terms.get("modern_context_markers", []))
    fantasy_re = _compile_alt(terms.get("fantasy_negative_markers", []))

    tier_a_hits = 0
    tier_b_hits = 0
    samples: list[dict[str, Any]] = []

    line_offset = 0
    for sec_lines in sections:
        n = len(sec_lines)
        for i, raw in enumerate(sec_lines):
            if not raw.strip():
                continue
            if tier_a_re is not None:
                m_a = tier_a_re.search(raw)
                if m_a:
                    tier_a_hits += 1
                    if len(samples) < max_samples:
                        st = max(0, m_a.start() - 35)
                        en = min(len(raw), m_a.end() + 35)
                        samples.append(
                            {"line": line_offset + i + 1, "tier": "A", "term": m_a.group(), "snippet": raw[st:en].strip()}
                        )
                    continue
            if tier_b_re is not None:
                m_b = tier_b_re.search(raw)
                if m_b:
                    lo = max(0, i - 2)
                    hi = min(n, i + 3)
                    win = "\n".join(sec_lines[lo:hi])
                    # Strip all Tier-B terms before checking external modern markers so e.g. `公安部` cannot trigger `公安`
                    win_clean = tier_b_re.sub(" ", win)
                    has_modern = bool(_MODERN_YEAR_RE.search(win_clean) or (modern_re and modern_re.search(win_clean)))
                    has_fantasy = bool(fantasy_re and fantasy_re.search(win_clean))
                    if has_modern and not has_fantasy:
                        tier_b_hits += 1
                        if len(samples) < max_samples:
                            st = max(0, m_b.start() - 35)
                            en = min(len(raw), m_b.end() + 35)
                            samples.append(
                                {"line": line_offset + i + 1, "tier": "B", "term": m_b.group(), "snippet": raw[st:en].strip()}
                            )
        line_offset += n

    level = "high" if (tier_a_hits >= 1 or tier_b_hits >= 3) else ("medium" if tier_b_hits >= 1 else "none")
    result: dict[str, Any] = {
        "level": level,
        "tier_a_hits": tier_a_hits,
        "tier_b_hits": tier_b_hits,
        "hit_count": tier_a_hits + tier_b_hits,
        "samples": samples,
    }

    if include_fadeout_hints:
        exp_re = _compile_alt(terms.get("fadeout_explicit", []))
        cue_re = _compile_alt(terms.get("fadeout_intimate_cues", []))
        exp_cnt = 0
        ell_cnt = 0
        f_samples: list[dict[str, Any]] = []
        line_offset = 0
        for sec_lines in sections:
            n = len(sec_lines)
            for i, raw in enumerate(sec_lines):
                if not raw.strip():
                    continue
                if exp_re is not None:
                    me = exp_re.search(raw)
                    if me:
                        exp_cnt += 1
                        if len(f_samples) < max_samples:
                            f_samples.append({"line": line_offset + i + 1, "kind": "explicit", "snippet": raw.strip()[:80]})
                        continue
                if cue_re is not None and _ELLIPSIS_LINE_RE.search(raw):
                    win1 = "\n".join(sec_lines[max(0, i - 1) : min(n, i + 2)])
                    if cue_re.search(win1):
                        ell_cnt += 1
                        if len(f_samples) < max_samples:
                            f_samples.append(
                                {"line": line_offset + i + 1, "kind": "intimate_ellipsis", "snippet": raw.strip()[:80]}
                            )
            line_offset += n
        result["fadeout_hints"] = {
            "explicit_count": exp_cnt,
            "ellipsis_count": ell_cnt,
            "samples": f_samples,
        }

    return result


def scan_book_policy(
    lines: list[str],
    *,
    terms_path: Path | None = None,
    include_fadeout_hints: bool = True,
    max_samples: int = 5,
) -> dict[str, Any]:
    """Scan lines read-only for political sensitivity (never mutates text) and optional fadeout hints."""
    return _scan_sections_policy(
        [lines],
        terms_path=terms_path,
        include_fadeout_hints=include_fadeout_hints,
        max_samples=max_samples,
    )


def _extract_xhtml_visible_lines(raw_bytes: bytes) -> list[str]:
    """Decode XHTML/HTML bytes with declared encoding and preserve inline continuity across formatting newlines."""
    text = _decode_xml_bytes(raw_bytes)
    text = _XHTML_STRIP_CONTAINER_RE.sub(" ", text)
    # Collapse pretty-print newlines inside source markup while preserving block breaks
    text = re.sub(r"\s*[\r\n]+\s*", " ", text)
    text = _XHTML_BLOCK_TAG_RE.sub("\n", text)
    text = _XHTML_ANY_TAG_RE.sub("", text)
    text = html.unescape(text)
    # Remove horizontal spaces inserted only when source XML broke CJK inline tags across lines (`文化\n<em>大革命</em>`)
    text = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", text)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def scan_path_policy(
    path: Path,
    *,
    terms_path: Path | None = None,
    include_fadeout_hints: bool = True,
    max_samples: int = 5,
) -> dict[str, Any]:
    """Read-only political & fadeout scan on a TXT or EPUB path (never modifies source files)."""
    if path.suffix.lower() == ".epub":
        sections: list[list[str]] = []
        with zipfile.ZipFile(path, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith((".xhtml", ".html", ".htm")):
                    sec = _extract_xhtml_visible_lines(zf.read(name))
                    if sec:
                        sections.append(sec)
        report = _scan_sections_policy(
            sections,
            terms_path=terms_path,
            include_fadeout_hints=include_fadeout_hints,
            max_samples=max_samples,
        )
    else:
        lines = read_novel_file(path).lines
        report = _scan_sections_policy(
            [lines],
            terms_path=terms_path,
            include_fadeout_hints=include_fadeout_hints,
            max_samples=max_samples,
        )
    report["file"] = str(path)
    return report
