"""EPUB archive validator and XML/XHTML repairer (preserves original files)."""

from __future__ import annotations

import html
import re
import zipfile
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

UNESCAPED_AMP_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")
NAMED_ENTITY_RE = re.compile(r"&([A-Za-z][A-Za-z0-9]{1,15});")
XML_BUILTIN_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}
XML_DECL_ENCODING_RE = re.compile(rb"<\?xml[^>]*\bencoding\s*=\s*[\"']([A-Za-z0-9_\-]+)[\"']", re.I)
VOID_HTML_TAGS = {"br", "hr", "img", "meta", "link", "input", "source", "wbr"}


class _XHTMLSanitizer(HTMLParser):
    """Best-effort HTML-to-XHTML serializer when regex fixes still fail XML parsing."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.out: list[str] = []
        self.stack: list[str] = []

    def handle_decl(self, decl: str) -> None:
        self.out.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        self.out.append(f"<?{data}?>")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        attr_str = "".join(
            f' {k}="{html.escape(v or "", quote=True)}"' for k, v in attrs
        )
        if tag_l in VOID_HTML_TAGS:
            self.out.append(f"<{tag_l}{attr_str} />")
        else:
            self.out.append(f"<{tag_l}{attr_str}>")
            self.stack.append(tag_l)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        attr_str = "".join(
            f' {k}="{html.escape(v or "", quote=True)}"' for k, v in attrs
        )
        self.out.append(f"<{tag_l}{attr_str} />")

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in VOID_HTML_TAGS:
            return
        while self.stack:
            top = self.stack.pop()
            self.out.append(f"</{top}>")
            if top == tag_l:
                break

    def handle_data(self, data: str) -> None:
        self.out.append(html.escape(data, quote=False))

    def handle_entityref(self, name: str) -> None:
        if name in {"amp", "lt", "gt", "quot", "apos"}:
            self.out.append(f"&{name};")
        else:
            self.out.append(html.escape(html.unescape(f"&{name};"), quote=False))

    def handle_charref(self, name: str) -> None:
        self.out.append(f"&#{name};")

    def finish(self) -> str:
        while self.stack:
            self.out.append(f"</{self.stack.pop()}>")
        return "".join(self.out)


@dataclass(frozen=True)
class EpubAuditResult:
    file: str
    mb: float
    total_entries: int
    xml_entries: int
    bad_xml_entries: list[str]
    repaired_entries: list[str]
    remaining_bad_entries: list[str]
    fixed_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_xml_bytes(data: bytes, *, strict: bool = False) -> str:
    """Decode XML/XHTML archive member strictly by BOM or declared/fallback encodings before replacement."""
    m = XML_DECL_ENCODING_RE.search(data[:256])
    tried: list[str] = []
    if m:
        decl_enc = m.group(1).decode("ascii", errors="ignore").strip()
        if decl_enc:
            tried.append(decl_enc)

    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return data.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            # Handle stray UTF-8 BOM over GB18030/Big5 payload strictly before falling back
            body = data[3:]
            for enc in [e for e in tried if e.lower() not in ("utf-8", "utf8", "utf-8-sig")] + ["gb18030", "big5"]:
                try:
                    return body.decode(enc, errors="strict")
                except (LookupError, UnicodeDecodeError):
                    continue
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16", errors="strict")
        except UnicodeDecodeError:
            pass

    for enc in tried + ["utf-8", "gb18030", "big5"]:
        try:
            return data.decode(enc, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
    if strict:
        raise UnicodeDecodeError("xml", data[:64], 0, min(len(data), 1), "Unable to decode XML bytes without loss")
    return data.decode("utf-8", errors="replace")


def _resolve_non_xml_entities(raw: str) -> str:
    """Resolve HTML named entities like &nbsp;/&mdash; into Unicode chars while keeping XML built-ins."""
    def _repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name.lower() in XML_BUILTIN_ENTITIES:
            return m.group(0)
        ent = f"&{name};"
        decoded = html.unescape(ent)
        if decoded == ent:
            return ent
        if any(c in decoded for c in "<>&\"'"):
            return html.escape(decoded, quote=True)
        return decoded

    return NAMED_ENTITY_RE.sub(_repl, raw)


def repair_xml_text(raw: str) -> str:
    """Fix HTML entities + unescaped ampersands and fallback-sanitize broken XHTML."""
    resolved = _resolve_non_xml_entities(raw)
    fixed = UNESCAPED_AMP_RE.sub("&amp;", resolved)
    fixed = re.sub(
        r"(<\?xml[^>]*encoding=[\"'])([A-Za-z0-9_\-]+)([\"'])",
        r"\g<1>utf-8\g<3>",
        fixed,
        count=1,
        flags=re.I,
    )
    try:
        ET.fromstring(fixed)
        return fixed
    except ET.ParseError:
        san = _XHTMLSanitizer()
        san.feed(fixed)
        san.close()
        candidate = san.finish()
        if not candidate.lstrip().startswith("<?xml"):
            candidate = '<?xml version="1.0" encoding="utf-8"?>\n' + candidate
        ET.fromstring(candidate)
        return candidate


_TG_EPUB_BLOCK_RE = re.compile(
    r"━{6,}(?:(?!━{6,})[\s\S]){0,1200}?━{6,}(?:\s*<br\s*/?>)*",
    re.I,
)
_XML_ENCODING_DECL_RE = re.compile(r'^(<\?xml\b[^>]*\bencoding\s*=\s*["\'])([^"\']+)(["\'])', re.I)
_META_CHARSET_RE = re.compile(r'(<meta\b[^>]*\bcharset\s*=\s*["\']?)([^"\'\s;/>]+)', re.I)


def _normalize_utf8_declarations(xhtml: str) -> str:
    """Update non-UTF-8 XML/HTML charset declarations before encoding serialized output as UTF-8."""
    out = _XML_ENCODING_DECL_RE.sub(lambda m: f"{m.group(1)}utf-8{m.group(3)}", xhtml, count=1)
    out = _META_CHARSET_RE.sub(lambda m: f"{m.group(1)}utf-8", out)
    return out


_TG_BLOCK_BREAK_RE = re.compile(
    r'''</?(?:[^\s:/>=<]+:)?(?:p|div|br|li|ul|ol|h[1-6]|tr|td|th|blockquote|section|article|hr)(?=[\s/>])(?:[^>"']|"[^"]*"|'[^']*')*>''',
    re.I,
)


def strip_epub_telegram_banners(raw_xhtml: str) -> str:
    """Remove Telegram channel promo blocks embedded inside downloaded XHTML pages."""
    from .group_soup import is_telegram_promo_block

    if "━" not in raw_xhtml or ("t.me/" not in raw_xhtml.lower() and "免责声明" not in raw_xhtml):
        return raw_xhtml

    def _replace_if_promo(match: re.Match[str]) -> str:
        # Collapse XML formatting newlines before breaking blocks so split inline dialogue stays intact
        txt = re.sub(r"\s*[\r\n]+\s*", " ", match.group())
        txt = _TG_BLOCK_BREAK_RE.sub("\n", txt)
        txt = re.sub(r"<[^>]+>", "", txt)
        inner_txt = html.unescape(txt)
        inner_txt = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", inner_txt)
        if is_telegram_promo_block(inner_txt.splitlines()):
            return ""
        return match.group()

    updated = re.sub(
        r"<p\b[^>]*>\s*" + _TG_EPUB_BLOCK_RE.pattern + r"\s*</p>",
        _replace_if_promo,
        raw_xhtml,
        flags=re.I,
    )
    updated = _TG_EPUB_BLOCK_RE.sub(_replace_if_promo, updated)
    # Clean redundant consecutive <br/> left right after stripped block
    updated = re.sub(r"(?:<br\s*/?>\s*){3,}", "<br/><br/>", updated)
    if updated == raw_xhtml:
        return raw_xhtml
    updated = _normalize_utf8_declarations(updated)
    ET.fromstring(updated.encode("utf-8"))
    return updated


def check_and_repair_epub(
    src_path: Path,
    out_dir: Path | None = None,
    *,
    strip_telegram_banner: bool = False,
) -> EpubAuditResult:
    mb = round(src_path.stat().st_size / (1024 * 1024), 2)
    bad_xml: list[str] = []
    repaired: list[str] = []
    remaining_bad: list[str] = []
    fixed_bytes_map: dict[str, bytes] = {}
    xml_count = 0

    with zipfile.ZipFile(src_path, "r") as zf:
        names = zf.namelist()
        for name in names:
            data = zf.read(name)
            if name.lower().endswith((".xhtml", ".html", ".htm", ".xml", ".svg", ".opf", ".ncx")):
                xml_count += 1
                try:
                    # Validate original raw archive bytes so encoding/byte-level XML errors are caught
                    ET.fromstring(data)
                    working_text = _decode_xml_bytes(data, strict=True)
                except (ET.ParseError, ValueError, LookupError, UnicodeDecodeError):
                    bad_xml.append(name)
                    try:
                        strict_text = _decode_xml_bytes(data, strict=True)
                        working_text = _normalize_utf8_declarations(repair_xml_text(strict_text))
                        payload = working_text.encode("utf-8")
                        ET.fromstring(payload)
                        fixed_bytes_map[name] = payload
                        repaired.append(name)
                    except Exception:
                        remaining_bad.append(name)
                        continue
                if strip_telegram_banner and name.lower().endswith((".xhtml", ".html", ".htm")):
                    try:
                        cleaned_xhtml = strip_epub_telegram_banners(working_text)
                        if cleaned_xhtml != working_text:
                            cleaned_xhtml = _normalize_utf8_declarations(cleaned_xhtml)
                            payload = cleaned_xhtml.encode("utf-8")
                            ET.fromstring(payload)
                            fixed_bytes_map[name] = payload
                            if name not in repaired:
                                repaired.append(name)
                    except Exception:
                        pass

        fixed_path: str | None = None
        if out_dir is not None and (bad_xml or fixed_bytes_map):
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"{src_path.stem}.fixed.epub"
            tmp = dest.with_suffix(".epub.tmp")
            with zipfile.ZipFile(tmp, "w") as out_zf:
                # Ensure mimetype first and uncompressed
                if "mimetype" in names:
                    out_zf.writestr("mimetype", zf.read("mimetype"), compress_type=zipfile.ZIP_STORED)
                for name in names:
                    if name == "mimetype":
                        continue
                    payload = fixed_bytes_map.get(name) if name in fixed_bytes_map else zf.read(name)
                    out_zf.writestr(name, payload, compress_type=zipfile.ZIP_DEFLATED)
            tmp.replace(dest)
            fixed_path = str(dest)

    return EpubAuditResult(
        file=str(src_path),
        mb=mb,
        total_entries=len(names),
        xml_entries=xml_count,
        bad_xml_entries=bad_xml,
        repaired_entries=repaired,
        remaining_bad_entries=remaining_bad,
        fixed_path=fixed_path,
    )
