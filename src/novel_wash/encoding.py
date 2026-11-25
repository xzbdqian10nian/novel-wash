"""Strict encoding detection and chunk-level fallback for Chinese web novels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from charset_normalizer import from_bytes

# Top ~350 high-frequency simplified Chinese characters + common Chinese punctuation
COMMON_ZH_SAMPLE = set(
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严"
    "，。！？：“”‘’、《》；（）"
)


@dataclass(frozen=True)
class DecodedFile:
    path: Path
    sha256: str
    encoding: str
    text: str
    lines: list[str]
    line_ending: str
    is_mixed: bool = False


def _score_decoded(text: str) -> float:
    if not text:
        return 0.0
    sample = text[:200_000]
    n = len(sample)
    repl = sample.count("\ufffd")
    ctrl = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\r\n\t")
    common = sum(1 for ch in sample if ch in COMMON_ZH_SAMPLE)
    cjk_main = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    cjk_ext = sum(1 for ch in sample if "\u3400" <= ch <= "\u4dbf")
    pua = sum(1 for ch in sample if "\ue000" <= ch <= "\uf8ff")
    # Common-char density is highest for true encoding; PUA allowed if Chinese punctuation exists.
    score = (common * 3.0 + cjk_main * 1.0 - cjk_ext * 1.5 - repl * 50.0 - ctrl * 30.0) / max(n, 1)
    if pua > 0 and repl == 0 and common > 0:
        score += 0.2
    return score


def _detect_bom(raw: bytes) -> tuple[str | None, int]:
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", 3
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le", 2
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be", 2
    return None, 0


def _decode_line_best(b_line: bytes) -> tuple[str, str | None]:
    """Decode one line strictly picking the highest-scoring encoding; ASCII lines return enc=None."""
    if not any(b >= 0x80 for b in b_line):
        return b_line.decode("ascii"), None
    best_text: str | None = None
    best_enc: str | None = None
    best_key: tuple[float, int, int] = (-1e9, -1, 0)
    for enc in ("utf-8", "gb18030", "big5"):
        try:
            t = b_line.decode(enc, errors="strict")
            sc = _score_decoded(t)
            common = sum(1 for ch in t if ch in COMMON_ZH_SAMPLE)
            # Prefer UTF-8 on exact ties (or equal common density)
            pref = 1 if enc == "utf-8" else 0
            key = (round(sc, 6), common, pref)
            if key > best_key:
                best_key = key
                best_text = t
                best_enc = enc
        except UnicodeDecodeError:
            continue
    if best_text is not None:
        return best_text, best_enc
    best = from_bytes(b_line).best()
    if best is not None:
        return str(best), best.encoding
    return b_line.decode("utf-8", errors="replace"), "utf-8-replace"


def _is_strong_utf8_win(b_line: bytes, whole_enc: str) -> tuple[bool, int, int]:
    """Return (is_win, u8_common, alt_common) for a non-ASCII line when strict UTF-8 cleanly beats `whole_enc`."""
    if not any(b >= 0x80 for b in b_line):
        return False, 0, 0
    try:
        u8_t = b_line.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False, 0, 0
    if "\ufffd" in u8_t:
        return False, 0, 0
    u8_common = sum(1 for ch in u8_t if ch in COMMON_ZH_SAMPLE)
    try:
        alt_t = b_line.decode(whole_enc, errors="strict")
    except UnicodeDecodeError:
        return True, u8_common, 0
    alt_common = sum(1 for ch in alt_t if ch in COMMON_ZH_SAMPLE)
    if _score_decoded(u8_t) > _score_decoded(alt_t) + 0.50 and u8_common >= max(1, alt_common + 1):
        return True, u8_common, alt_common
    return False, u8_common, alt_common


def _changed_lines_improved(alt_text: str, new_text: str) -> bool:
    alt_lines = alt_text.splitlines()
    new_lines = new_text.splitlines()
    if len(alt_lines) != len(new_lines):
        return _score_decoded(new_text) > _score_decoded(alt_text)
    diffs_alt = [a for a, b in zip(alt_lines, new_lines) if a != b]
    diffs_new = [b for a, b in zip(alt_lines, new_lines) if a != b]
    if not diffs_new:
        return False
    return _score_decoded("\n".join(diffs_new)) > _score_decoded("\n".join(diffs_alt))


def _has_mixed_utf8_line_evidence(raw: bytes, whole_enc: str) -> bool:
    """Return True if at least one non-ASCII line decodes strictly as UTF-8 cleanly better than `whole_enc`."""
    for b_line in raw.splitlines():
        if _is_strong_utf8_win(b_line, whole_enc)[0]:
            return True
    return False


def _decode_chunk_fallback(raw: bytes) -> tuple[str, str]:
    """Line-aligned ~4KB chunk decoding with mixed-encoding intra-chunk line resolution."""
    parts: list[str] = []
    used_encs: set[str] = set()
    lines_bytes = raw.splitlines(keepends=True)
    buf: list[bytes] = []
    buf_len = 0

    def flush(chunk: bytes) -> None:
        nonlocal parts, used_encs
        # Fast path: chunk decodes cleanly as UTF-8
        try:
            u8_chunk = chunk.decode("utf-8", errors="strict")
            parts.append(u8_chunk)
            if any(b >= 0x80 for b in chunk):
                used_encs.add("utf-8")
            return
        except UnicodeDecodeError:
            pass

        best_text: str | None = None
        best_enc = "utf-8"
        best_score = -1e9
        for enc in ("gb18030", "big5"):
            try:
                t = chunk.decode(enc, errors="strict")
                sc = _score_decoded(t)
                if sc > best_score:
                    best_score = sc
                    best_text = t
                    best_enc = enc
            except UnicodeDecodeError:
                continue

        if (b"\n" in chunk or b"\r" in chunk) and (
            best_text is None or _has_mixed_utf8_line_evidence(chunk, best_enc)
        ):
            sub_parts: list[str] = []
            sub_encs: set[str] = set()
            changed_u8_texts: list[str] = []
            changed_alt_texts: list[str] = []
            for sub_line in chunk.splitlines(keepends=True):
                lt, lenc = _decode_line_best(sub_line)
                sub_parts.append(lt)
                if lenc:
                    sub_encs.add(lenc)
                    if best_text is not None and lenc != best_enc:
                        try:
                            changed_alt_texts.append(sub_line.decode(best_enc, errors="strict"))
                            changed_u8_texts.append(lt)
                        except UnicodeDecodeError:
                            changed_u8_texts.append(lt)
            joined = "".join(sub_parts)
            switched_win = bool(changed_u8_texts) and (
                not changed_alt_texts
                or _score_decoded("".join(changed_u8_texts)) > _score_decoded("".join(changed_alt_texts))
            )
            if best_text is None or (joined.count("\ufffd") == 0 and switched_win):
                parts.append(joined)
                used_encs.update(sub_encs or {best_enc})
                return

        if best_text is not None:
            parts.append(best_text)
            if any(b >= 0x80 for b in chunk):
                used_encs.add(best_enc)
            return

        if b"\n" in chunk or b"\r" in chunk:
            for sub_line in chunk.splitlines(keepends=True):
                lt, lenc = _decode_line_best(sub_line)
                parts.append(lt)
                if lenc:
                    used_encs.add(lenc)
            return

        lt, lenc = _decode_line_best(chunk)
        parts.append(lt)
        if lenc:
            used_encs.add(lenc)

    for b_line in lines_bytes:
        buf.append(b_line)
        buf_len += len(b_line)
        if buf_len >= 4096:
            flush(b"".join(buf))
            buf.clear()
            buf_len = 0
    if buf:
        flush(b"".join(buf))

    if not used_encs:
        used_encs.add("utf-8")
    label = "mixed:" + "+".join(sorted(used_encs)) if len(used_encs) > 1 else next(iter(used_encs))
    return "".join(parts), label


def decode_bytes(raw: bytes, path: Path | None = None) -> DecodedFile:
    sha = hashlib.sha256(raw).hexdigest()
    bom_enc, bom_skip = _detect_bom(raw)
    if bom_enc is not None:
        try:
            text = raw[bom_skip:].decode(bom_enc.replace("-sig", ""), errors="strict")
            return _finish_decode(path or Path("<memory>"), sha, bom_enc, text, is_mixed=False)
        except UnicodeDecodeError:
            pass

    candidates: list[tuple[float, str, str]] = []
    enc_list = ["utf-8", "gb18030", "big5"]
    if len(raw) % 2 == 0 and raw[:4096].count(b"\x00") > len(raw[:4096]) * 0.1:
        enc_list.extend(["utf-16-le", "utf-16-be"])

    for enc in enc_list:
        try:
            text = raw.decode(enc, errors="strict")
            candidates.append((_score_decoded(text), enc, text))
        except UnicodeDecodeError:
            continue

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_enc, best_text = candidates[0]
        # Guard against mixed UTF-8 + GB18030 files that strictly decode as GB18030 mojibake
        if best_enc in ("gb18030", "big5") and not any(e == "utf-8" for _, e, _ in candidates):
            if _has_mixed_utf8_line_evidence(raw, best_enc):
                mixed_text, mixed_label = _decode_chunk_fallback(raw)
                if (
                    "+" in mixed_label
                    and mixed_text.count("\ufffd") == 0
                    and _changed_lines_improved(best_text, mixed_text)
                ):
                    return _finish_decode(path or Path("<memory>"), sha, mixed_label, mixed_text, is_mixed=True)
        return _finish_decode(path or Path("<memory>"), sha, best_enc, best_text, is_mixed=False)

    # Try charset_normalizer whole-file
    cn = from_bytes(raw).best()
    if cn is not None:
        text = str(cn)
        if text.count("\ufffd") == 0:
            return _finish_decode(path or Path("<memory>"), sha, cn.encoding, text, is_mixed=False)

    # Mixed-encoding line-aligned fallback
    text, enc_label = _decode_chunk_fallback(raw)
    return _finish_decode(path or Path("<memory>"), sha, enc_label, text, is_mixed=("+" in enc_label))


def _finish_decode(path: Path, sha: str, encoding: str, text: str, *, is_mixed: bool) -> DecodedFile:
    # Strip leading BOM if any remained
    if text.startswith("\ufeff"):
        text = text[1:]
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    cr = text.count("\r") - crlf
    if crlf > lf and crlf > cr:
        line_ending = "\r\n"
    elif cr > lf and cr > 0:
        line_ending = "\r"
    else:
        line_ending = "\n"
    # Normalize internal line endings to logical lines (original physical line structure preserved)
    lines = text.splitlines()
    return DecodedFile(
        path=path,
        sha256=sha,
        encoding=encoding,
        text=text,
        lines=lines,
        line_ending=line_ending,
        is_mixed=is_mixed,
    )


def read_novel_file(path: Path) -> DecodedFile:
    raw = path.read_bytes()
    return decode_bytes(raw, path=path)
