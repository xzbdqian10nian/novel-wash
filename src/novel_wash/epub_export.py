"""Pure-Python EPUB3 generator from cleaned reader text."""

from __future__ import annotations

import html
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from .edit_log import detect_numbered_scheme, is_chapter_heading


@dataclass(frozen=True)
class Chapter:
    index: int
    title: str
    paragraphs: list[str]
    is_synthetic_title: bool = False
    lead_paragraphs: list[str] = field(default_factory=list)


_CH_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CH_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_CHAPTER_NUM_EXTRACT_RE = re.compile(r"^(?:\d{1,5}[.．、]\s*)?第\s*([零〇一二三四五六七八九十百千万两\d]+)\s*章")
_NUMBERED_HEADING_EXTRACT_RE = re.compile(r"^(\d{1,4})\s*[.．、]\s*\S")


def _extract_chapter_num(heading: str, *, allow_numbered: bool = False) -> int | None:
    s = heading.strip()
    m = _CHAPTER_NUM_EXTRACT_RE.match(s)
    if not m:
        if allow_numbered:
            mn = _NUMBERED_HEADING_EXTRACT_RE.match(s)
            if mn:
                return int(mn.group(1))
        return None
    token = m.group(1)
    if token.isdigit():
        return int(token)
    total = 0
    section = 0
    cur = 0
    has_any = False
    for ch in token:
        if ch in _CH_DIGITS:
            cur = _CH_DIGITS[ch]
            has_any = True
        elif ch in _CH_UNITS:
            has_any = True
            unit = _CH_UNITS[ch]
            if unit == 10000:
                section = (section + cur) * unit
                total += section
                section = 0
            else:
                val = cur if cur > 0 else (1 if unit == 10 else 0)
                section += val * unit
            cur = 0
        else:
            return None
    return (total + section + cur) if has_any else None


def _heading_quality(heading: str) -> tuple[int, int, int, int, int]:
    """Score chapter heading completeness so unmasked, titled headings beat masked or bare variants."""
    s = heading.strip()
    no_mask_stars = 0 if ("*" in s or "＊" in s) else 1
    balanced_parens = 1 if (s.count("（") == s.count("）") and s.count("(") == s.count(")")) else 0
    m = _CHAPTER_NUM_EXTRACT_RE.match(s)
    has_title = 0
    has_sep = 0
    if m:
        after = s[m.end() :]
        title_tail = after.lstrip(" \t\u3000：:·-—")
        has_title = 1 if bool(title_tail) else 0
        has_sep = 1 if (bool(after) and after[0] in " \t\u3000：:·-—（(【[") else 0
    else:
        mn = _NUMBERED_HEADING_EXTRACT_RE.match(s)
        if mn:
            has_title = 1
            has_sep = 1
    return (no_mask_stars, balanced_parens, has_title, has_sep, len(s))


def _heading_identity(heading: str, *, allow_numbered: bool = False) -> tuple[str, int | str]:
    num = _extract_chapter_num(heading, allow_numbered=allow_numbered)
    return ("num", num) if num is not None else ("raw", heading)


def split_chapters(text: str, *, book_title: str) -> list[Chapter]:
    """Split cleaned reader text into chapters by Chinese chapter headings without dropping any lines."""
    nonblank_lines = [raw_line.strip() for raw_line in text.splitlines() if raw_line.strip()]
    numbered_indices = detect_numbered_scheme(nonblank_lines) or set()
    use_num = bool(numbered_indices)
    heading_flags = [
        is_chapter_heading(line) or (pos in numbered_indices)
        for pos, line in enumerate(nonblank_lines)
    ]
    # Demote inline TOC blocks (>= 20 consecutive heading-only lines with >= 10 distinct chapters).
    # Keep trailing same-chapter run when followed by body text AND it genuinely restarts after the TOC.
    inline_toc_indices: set[int] = set()
    n_nb = len(nonblank_lines)
    idx = 0
    while idx < n_nb:
        if not heading_flags[idx]:
            idx += 1
            continue
        end_idx = idx
        while end_idx < n_nb and heading_flags[end_idx]:
            end_idx += 1
        if end_idx - idx >= 20:
            run_slice = nonblank_lines[idx:end_idx]
            distinct_chapters = {_heading_identity(h, allow_numbered=use_num) for h in run_slice}
            if len(distinct_chapters) >= 10:
                demote_end = end_idx
                if end_idx < n_nb:
                    last_id = _heading_identity(nonblank_lines[end_idx - 1], allow_numbered=use_num)
                    trail_start = end_idx - 1
                    while trail_start > idx and _heading_identity(nonblank_lines[trail_start - 1], allow_numbered=use_num) == last_id:
                        trail_start -= 1
                    if trail_start > idx:
                        toc_prefix = nonblank_lines[idx:trail_start]
                        trail_num = _extract_chapter_num(nonblank_lines[trail_start], allow_numbered=use_num)
                        prev_num = _extract_chapter_num(nonblank_lines[trail_start - 1], allow_numbered=use_num)
                        toc_nums = {
                            num for h in toc_prefix if (num := _extract_chapter_num(h, allow_numbered=use_num)) is not None
                        }
                        toc_raw = set(toc_prefix)
                        restarts_earlier = (
                            (trail_num is not None and prev_num is not None and trail_num < prev_num)
                            or (trail_num is not None and trail_num in toc_nums)
                            or any(h in toc_raw for h in nonblank_lines[trail_start:end_idx])
                        )
                        if restarts_earlier:
                            demote_end = trail_start
                        elif end_idx - trail_start >= 2:
                            # Same-chapter restart immediately after the final TOC entry (e.g. TOC 1..20
                            # followed by real chapter 20 heading(s) + body prose).
                            demote_end = trail_start + 1
                inline_toc_indices.update(range(idx, demote_end))
        idx = end_idx

    chapters: list[Chapter] = []
    cur_heading_run: list[str] = []
    cur_title = "序"
    cur_is_synthetic = True
    cur_paras: list[str] = []
    cur_has_body_prose = False
    any_real_heading = False

    def _flush_current() -> None:
        if cur_is_synthetic:
            if cur_paras or not chapters:
                final_title = cur_title if (cur_paras or chapters) else book_title
                chapters.append(
                    Chapter(
                        len(chapters) + 1,
                        final_title,
                        cur_paras,
                        is_synthetic_title=True,
                    )
                )
            return
        best_idx = max(range(len(cur_heading_run)), key=lambda i_: (_heading_quality(cur_heading_run[i_]), -i_))
        best_title = cur_heading_run[best_idx]
        lead_ps = cur_heading_run[:best_idx]
        body_ps = cur_heading_run[best_idx + 1 :] + cur_paras
        chapters.append(
            Chapter(
                len(chapters) + 1,
                best_title,
                body_ps,
                is_synthetic_title=False,
                lead_paragraphs=lead_ps,
            )
        )

    for pos, line in enumerate(nonblank_lines):
        if heading_flags[pos] and pos not in inline_toc_indices:
            if not cur_is_synthetic and not cur_has_body_prose and cur_heading_run:
                cur_num = _extract_chapter_num(cur_heading_run[0], allow_numbered=use_num)
                new_num = _extract_chapter_num(line, allow_numbered=use_num)
                if (cur_num is not None and cur_num == new_num) or (
                    cur_num is None and new_num is None and cur_heading_run[-1] == line
                ):
                    cur_heading_run.append(line)
                    continue
            any_real_heading = True
            if not cur_is_synthetic or cur_paras:
                _flush_current()
            cur_heading_run = [line]
            cur_title = line
            cur_is_synthetic = False
            cur_paras = []
            cur_has_body_prose = False
        else:
            cur_paras.append(line)
            if not heading_flags[pos]:
                cur_has_body_prose = True

    if not cur_is_synthetic or cur_paras or not chapters:
        _flush_current()

    # Fallback chunking ONLY if the source had no recognized headings at all and > 400 paragraphs
    if not any_real_heading and len(chapters) == 1 and len(chapters[0].paragraphs) > 400:
        paras = chapters[0].paragraphs
        chapters = []
        chunk_size = 120
        for idx, start in enumerate(range(0, len(paras), chunk_size), start=1):
            chapters.append(
                Chapter(
                    idx,
                    f"{book_title}（第{idx}节）",
                    paras[start : start + chunk_size],
                    is_synthetic_title=True,
                )
            )

    return chapters


_STYLE_CSS = """\
body {
  font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", serif;
  line-height: 1.75;
  margin: 1.2em;
  color: #1f2328;
}
h1 {
  font-size: 1.35em;
  margin: 0.6em 0 1em 0;
  border-bottom: 1px solid #e5e7eb;
  padding-bottom: 0.35em;
}
p {
  text-indent: 2em;
  margin: 0.45em 0;
}
"""


def _render_chapter_xhtml(ch: Chapter) -> str:
    esc_title = html.escape(ch.title)
    lead_block = ("\n" + "\n".join(f"  <p>{html.escape(p)}</p>" for p in ch.lead_paragraphs)) if ch.lead_paragraphs else ""
    h1_tag = f'  <h1 class="synthetic-title">{esc_title}</h1>' if ch.is_synthetic_title else f"  <h1>{esc_title}</h1>"
    body_block = ("\n" + "\n".join(f"  <p>{html.escape(p)}</p>" for p in ch.paragraphs)) if ch.paragraphs else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="zh-CN" lang="zh-CN">\n'
        "<head>\n"
        f"  <title>{esc_title}</title>\n"
        '  <link rel="stylesheet" type="text/css" href="style.css"/>\n'
        "</head>\n"
        "<body>"
        f"{lead_block}\n"
        f"{h1_tag}"
        f"{body_block}\n"
        "</body>\n"
        "</html>\n"
    )


def verify_epub_text_preservation(reader_text: str, epub_path: Path) -> tuple[bool, list[str]]:
    """Verify every non-blank line in `reader_text` appears in exact order inside exported EPUB spine XHTML."""
    expected = [line.strip() for line in reader_text.splitlines() if line.strip()]
    actual: list[str] = []

    with zipfile.ZipFile(epub_path, "r") as zf:
        opf_raw = zf.read("OEBPS/content.opf").decode("utf-8")
        opf_root = ET.fromstring(opf_raw)
        ns = {"opf": "http://www.idpf.org/2007/opf"}
        href_by_id: dict[str, str] = {}
        for item in opf_root.findall("./opf:manifest/opf:item", ns):
            iid = item.attrib.get("id")
            href = item.attrib.get("href")
            if iid and href:
                href_by_id[iid] = href

        for itemref in opf_root.findall("./opf:spine/opf:itemref", ns):
            idref = itemref.attrib.get("idref", "")
            href = href_by_id.get(idref)
            if not href:
                continue
            member = href if href.startswith("OEBPS/") else f"OEBPS/{href}"
            xhtml_root = ET.fromstring(zf.read(member).decode("utf-8"))
            for elem in xhtml_root.iter():
                tag = elem.tag.split("}", 1)[-1]
                if tag == "h1":
                    cls = elem.attrib.get("class", "")
                    if "synthetic-title" in cls.split():
                        continue
                    txt = "".join(elem.itertext()).strip()
                    if txt:
                        actual.append(txt)
                elif tag == "p":
                    txt = "".join(elem.itertext()).strip()
                    if txt:
                        actual.append(txt)

    if expected == actual:
        return True, []

    diffs: list[str] = []
    if len(expected) != len(actual):
        diffs.append(f"line_count_mismatch: expected={len(expected)}, actual={len(actual)}")
    for idx, (e, a) in enumerate(zip(expected, actual), start=1):
        if e != a:
            diffs.append(f"first_diff@{idx}: expected={e!r}, actual={a!r}")
            break
    if not diffs:
        if len(expected) > len(actual):
            diffs.append(f"missing_tail_example={expected[len(actual)]!r}")
        else:
            diffs.append(f"extra_tail_example={actual[len(expected)]!r}")
    return False, diffs


def build_epub(
    reader_text: str,
    out_path: Path,
    *,
    title: str,
    author: str = "未知",
) -> int:
    """Package cleaned reader text into a valid EPUB3 archive; returns chapter count."""
    chapters = split_chapters(reader_text, book_title=title)
    book_uuid = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_DNS, title + ':' + str(len(reader_text)))}"
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    esc_title = html.escape(title)
    esc_author = html.escape(author)

    container_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        "  <rootfiles>\n"
        '    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n'
        "  </rootfiles>\n"
        "</container>\n"
    )

    nav_items = "\n".join(
        f'      <li><a href="ch_{ch.index:04d}.xhtml">{html.escape(ch.title)}</a></li>'
        for ch in chapters
    )
    nav_xhtml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="zh-CN" lang="zh-CN">\n'
        "<head>\n"
        f"  <title>{esc_title} 目录</title>\n"
        '  <link rel="stylesheet" type="text/css" href="style.css"/>\n'
        "</head>\n"
        "<body>\n"
        '  <nav epub:type="toc" id="toc">\n'
        f"    <h1>{esc_title}</h1>\n"
        "    <ol>\n"
        f"{nav_items}\n"
        "    </ol>\n"
        "  </nav>\n"
        "</body>\n"
        "</html>\n"
    )

    ncx_points = "\n".join(
        f'    <navPoint id="navPoint-{ch.index}" playOrder="{ch.index}">\n'
        f"      <navLabel><text>{html.escape(ch.title)}</text></navLabel>\n"
        f'      <content src="ch_{ch.index:04d}.xhtml"/>\n'
        "    </navPoint>"
        for ch in chapters
    )
    toc_ncx = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">\n'
        "  <head>\n"
        f'    <meta name="dtb:uid" content="{book_uuid}"/>\n'
        '    <meta name="dtb:depth" content="1"/>\n'
        '    <meta name="dtb:totalPageCount" content="0"/>\n'
        '    <meta name="dtb:maxPageNumber" content="0"/>\n'
        "  </head>\n"
        f"  <docTitle><text>{esc_title}</text></docTitle>\n"
        "  <navMap>\n"
        f"{ncx_points}\n"
        "  </navMap>\n"
        "</ncx>\n"
    )

    manifest_items = [
        '    <item id="css" href="style.css" media-type="text/css"/>',
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
    ] + [
        f'    <item id="ch_{ch.index:04d}" href="ch_{ch.index:04d}.xhtml" media-type="application/xhtml+xml"/>'
        for ch in chapters
    ]
    spine_items = [f'    <itemref idref="ch_{ch.index:04d}"/>' for ch in chapters]

    content_opf = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="BookId" version="3.0">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        f'    <dc:identifier id="BookId">{book_uuid}</dc:identifier>\n'
        f"    <dc:title>{esc_title}</dc:title>\n"
        f"    <dc:creator>{esc_author}</dc:creator>\n"
        "    <dc:language>zh-CN</dc:language>\n"
        f'    <meta property="dcterms:modified">{modified}</meta>\n'
        "  </metadata>\n"
        "  <manifest>\n"
        + "\n".join(manifest_items)
        + "\n  </manifest>\n"
        '  <spine toc="ncx">\n'
        + "\n".join(spine_items)
        + "\n  </spine>\n"
        "</package>\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        # Per EPUB specification, mimetype MUST be first and stored uncompressed
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/style.css", _STYLE_CSS, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/nav.xhtml", nav_xhtml, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/toc.ncx", toc_ncx, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/content.opf", content_opf, compress_type=zipfile.ZIP_DEFLATED)
        for ch in chapters:
            zf.writestr(
                f"OEBPS/ch_{ch.index:04d}.xhtml",
                _render_chapter_xhtml(ch),
                compress_type=zipfile.ZIP_DEFLATED,
            )
    tmp.replace(out_path)
    return len(chapters)


def infer_title_and_author(path: Path) -> tuple[str, str]:
    stem = path.stem
    stem = re.sub(r"⊙完本$", "", stem).strip()
    if " - " in stem:
        t, a = stem.rsplit(" - ", 1)
        return t.strip("《》 "), a.strip()
    if "-" in stem and len(stem.rsplit("-", 1)[1]) <= 12:
        t, a = stem.rsplit("-", 1)
        return t.strip("《》 "), a.strip()
    return stem.strip("《》 "), "未知"
