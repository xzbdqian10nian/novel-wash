"""Layer 2: Standalone encoded group-number soup & site-banner cleaning.

Replaces rigid AND-conjunctions and narrow manual character lists with:
1. Shadow-string pinyin + numeral equivalence decoding (without altering original output).
2. Weighted multi-signal scoring (preserving normal sentences, dialogue, and headings).
3. Book-level group-number clustering: recurring 6-11 digit numbers across chapters rescue
   borderline/variant soup lines with near-zero false positives.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlsplit

import jieba
from pypinyin import Style, pinyin

from .edit_log import EditOp, is_chapter_heading
from .shadow import ShadowLine, build_shadow

# Keep jieba quiet during imports/runs if needed
jieba.setLogLevel(60)

CIRCLED_MAP: dict[str, str] = {
    "⓪": "0",
    **{chr(0x2460 + i): str(i + 1) for i in range(20)},  # ①..⑳ -> 1..20
    **{chr(0x2474 + i): str(i + 1) for i in range(20)},  # ⑴..⒇ -> 1..20
    **{chr(0x2488 + i): str(i + 1) for i in range(20)},  # ⒈..⒛ -> 1..20
    **{chr(0x24EB + i): str(i + 11) for i in range(10)},  # ⓫..⓴ -> 11..20
    **{chr(0x24F5 + i): str(i + 1) for i in range(10)},  # ⓵..⓾ -> 1..10
    **{chr(0x2776 + i): str(i + 1) for i in range(10)},  # ❶..❿ -> 1..10
    **{chr(0x2780 + i): str(i + 1) for i in range(10)},  # ➀..➉ -> 1..10
    **{chr(0x278A + i): str(i + 1) for i in range(10)},  # ➊..➓ -> 1..10
    **{chr(0x3021 + i): str(i + 1) for i in range(9)},   # 〡..〩 -> 1..9
}

DIRECT_CJK_DIGIT: dict[str, str] = {
    "零": "0", "〇": "0", "○": "0",
    "一": "1", "壹": "1", "幺": "1", "弌": "1",
    "二": "2", "贰": "2", "弍": "2", "两": "2", "俩": "2",
    "三": "3", "叁": "3", "叄": "3", "参": "3",
    "四": "4", "肆": "4",
    "五": "5", "伍": "5",
    "六": "6", "陆": "6",
    "七": "7", "柒": "7",
    "八": "8", "捌": "8",
    "九": "9", "玖": "9",
}

RADICAL_CJK_DIGIT: dict[str, str] = {
    "氵": "3",
    "彡": "3",
}

PY_STRICT_DIGIT: dict[str, str] = {
    "ling": "0", "lin": "0",
    "yi": "1", "yao": "1",
    "er": "2", "liang": "2",
    "san": "3", "shan": "3",
    "si": "4",
    "wu": "5",
    "liu": "6",
    "qi": "7",
    "ba": "8",
    "jiu": "9",
}

# Broader dialect/homophone pinyins seen in watermark soup (e.g., 是/师/十-ish -> shi, 气->qi)
PY_BROAD_DIGIT: dict[str, str] = {
    **PY_STRICT_DIGIT,
    "shi": "4",
}

# Explicit character overrides for heteronyms / dialect variants in watermark soup
CHAR_DIGIT_OVERRIDE: dict[str, str] = {
    "奇": "7",
    "陆": "6",
    "参": "3",
    "玐": "8",
    "镹": "9",
    "岜": "8",
    "锍": "6",
    "硫": "6",
    "氿": "9",
    "阾": "0",
    "澪": "0",
    "〇": "0",
    "陾": "0",
    "镏": "6",
    "蹴": "7",
    "企": "7",
}

GROUP_KEYWORDS = set("群裙峮裠箘撩聊灵梦猛鹅扣外")
RARE_SOUP_SIGNALS = set(
    "氿紦帬羣裠澪厁阾镹陾宭踆玐儛叄罒倭俬熘峮氵彡児侕箘囷鸸岜麇邬轳鹨衤弌弍覇锍诌"
    "壹贰叄肆伍陸柒捌玖靈夢镏蹴"
)

LATIN_PINYIN_RE = re.compile(r"(?:ling|liang|jiu|san|liu|yao|er|yi|si|wu|qi|ba)", re.I)
ROMAN_TOKEN_RE = re.compile(r"(?<![A-Za-z])(?:VIII|VII|III|VI|IV|IX|II|V|I|X)(?![A-Za-z])")
KEEP_ENGLISH_WORDS = {
    "vip", "bug", "buff", "play", "npc", "dps", "mvp", "xp", "ex", "max",
    "hp", "mp", "lv", "pk", "aoe", "cc", "cg", "gank", "ot", "ok", "skip",
    "boss", "pua", "man", "app", "ps", "cd", "rpg", "up", "qq", "gm", "ai",
}

HARD_NOISE_SYM_CHARS = set("#@!?:;/&|+^`$%\\<>{}[]〃⊙")
NOISE_PUNCTS = set("#@!?:;/&|*+^`$%\\<>[]{}()（）【】·.-～~_=\"'‘’“”〔〕〃⊙…")
SENTENCE_END_PUNCTS = set("。！？")
CHAPTER_SPLIT_MARKER_RE = re.compile(r"^[（(]\s*\d+\s*/\s*\d+\s*更?\s*[）)]$")

PAIRED_DIALOGUE_QUOTE_RE = re.compile(
    r"“[^”\r\n]{1,120}”|「[^」\r\n]{1,120}」|『[^』\r\n]{1,120}』|‘[^’\r\n]{1,120}’"
)
MID_INDENT_PROSE_RE = re.compile(r"\S\u3000+[\u4e00-\u9fff]{2,}[。！？]")

_URL_BODY = (
    r"(?:https?://|www\.|[a-z0-9\-]+\.(?:com|net|org|cc|cn|vip|xyz|top|io|me|la|tv))"
    r"(?:[a-z0-9\-._~:/?#\[\]@!$&'*+=%]*[a-z0-9/])?"
)
_CLOSED_AD_TAIL = r"(?:永久地址|防走丢|请收藏|请记住|手机版|无弹窗|免费阅读|最新地址|备用地址|最新网址|官网)"
_CLOSED_PROVENANCE_TAIL = r"(?:整理|制作|分享|收集|校对|下载|免费分享|仅供试读|版权归原作者所有)"
_KNOWN_SITE_NAMES = (
    r"(?:搜书吧|尚香书院|笔趣阁|顶点小说|飘天文学|全本小说网|"
    r"起点中文网|晋江文学城|刺猬猫|菠萝包|飞卢小说网|月漪|番茄小说|七猫小说)"
)

FULL_LINE_AD_PATTERNS = [
    re.compile(r"^\[img(?:=[^\]]*)?\].*?\[/img\]$", re.I),
    re.compile(
        rf"^本文件来自[ \t]*(?:{_URL_BODY}|{_KNOWN_SITE_NAMES})(?:[，,\s]*{_CLOSED_PROVENANCE_TAIL})*[。！!]?$",
        re.I,
    ),
    re.compile(
        rf"^(?:(?:搜书吧|笔趣阁|顶点小说|飘天文学|尚香书院|全本小说网)[ \t]*)?(?:发布页|来源|最新网址|记住网址|首发网址)[:：]\s*{_URL_BODY}(?:\s*{_CLOSED_AD_TAIL})*[。！!]?$",
        re.I,
    ),
    re.compile(
        rf"^(?:(?:搜书吧|笔趣阁|顶点小说|飘天文学|尚香书院|全本小说网)[ \t]*)?(?:发布页|来源|最新网址|记住网址|首发网址)[:：]\s*(?:搜书吧|笔趣阁|顶点|飘天|尚香|全本小说|小说网|书院|文学网)(?:\s*{_CLOSED_AD_TAIL})*[。！!]?$",
        re.I,
    ),
    re.compile(
        rf"^(?:搜书吧|笔趣阁|顶点小说|飘天文学|尚香书院|全本小说网)(?:[:：\s]+(?:发布页|最新网址|最新地址|记住网址|首发网址|官方网址|下载|{_URL_BODY}|{_CLOSED_AD_TAIL}))+[。！!]?$",
        re.I,
    ),
    re.compile(r"^[-]{5,}$"),
    re.compile(r"^={5,}$"),
    re.compile(r"^\*{5,}$"),
    re.compile(r"^[（(]月[（(]?费[#＃]?群\d{3,8}[)）]?\d{0,8}[)）]?\"?\d{0,6}[)）]?$"),
    re.compile(rf"^(?:记住网址|首发网址|手机阅读)[:：]?\s*{_URL_BODY}[。！!]?$", re.I),
    re.compile(r"^本书由[【\[]月漪[】\]]整理.*(?:版权|试读|删除|正版|盗版|权益).*$"),
    re.compile(r"^[【\[]月漪[】\]]提醒您[：:].*(?:阅读时间|沉迷|免费外群|月漪).*$"),
    re.compile(r"^[【\[]月漪(?:免费外群|中转群|内部群|书友群|外群)\d*号?[】\]][—\-－–_=:：\s]*\d{5,12}$"),
    re.compile(r"^本群免费提取全网平台已购vip章节.*$", re.I),
    re.compile(r"^[【\[][^】\]]{1,24}[群裙羣][】\]][—\-－–_=:：\s]*[\d\W_+`'\"]{6,}$"),
]

_LINGDIAN_TLD = r"(?:com|net|org|cc|cn|vip|xyz|top|io|me|la|tv|ｃｏｍ)"
_LINGDIAN_HOST_TAIL = (
    rf"(?:[a-z0-9ｚzｗwｃcｏoｍm]{{0,12}}\.{_LINGDIAN_TLD}(?![A-Za-z0-9.\-_@])"
    rf"(?:/[a-z0-9.\-_/:?=#&]{{0,30}})?|(?![A-Za-z0-9.\-_@]))"
)
_URL_LEFT_BOUNDARY = r"(?<![A-Za-z0-9.\-_/:?=&%#+~])"
_OBF_LINGDIAN_PATTERN = (
    r"(?<![./:?=&%#+~])"
    r"(?:"
    r"(?:思址|捌h群)(?:https?://|ttps?://)m\."
    r"|"
    r"(?![a-zA-Z:/\.]+lingdian)"
    r"t[^\s。，,！？\"”’」『』/:?=&]{0,2}p[^\s。，,！？\"”’」『』/:?=&]{0,4}:/"
    r"[^\s。，,！？\"”’」『』a-zA-Z0-9]{0,2}/[^\s。，,！？\"”’」『』a-zA-Z0-9]{0,2}m\.[^\x00-\x7f]?"
    r")"
    rf"lingdian{_LINGDIAN_HOST_TAIL}"
)

INLINE_AD_PATTERNS = [
    re.compile(
        rf"(?:^|(?<=[ \t\u3000]))(?:(?:搜书吧|笔趣阁|顶点小说|飘天文学|尚香书院|全本小说网)[ \t]*)?(?:来源|发布页|最新网址|首发网址|记住网址)[:：]\s*{_URL_BODY}(?:[。，,][ \t]*|[ \t]+)?",
        re.I,
    ),
    re.compile(
        rf"记住网址[:：\s]*(?:{_URL_BODY}|m\.lingdian{_LINGDIAN_HOST_TAIL})",
        re.I,
    ),
    re.compile(
        rf"首发[^\s。，,！？\"”’」『』]{{0,8}}网址[:：\s]*(?:{_URL_BODY}|https?://m\.lingdian{_LINGDIAN_HOST_TAIL}|m\.lingdian{_LINGDIAN_HOST_TAIL})",
        re.I,
    ),
    re.compile(_OBF_LINGDIAN_PATTERN, re.I),
    re.compile(rf"{_URL_LEFT_BOUNDARY}https?://m\.lingdian{_LINGDIAN_HOST_TAIL}", re.I),
    re.compile(rf"{_URL_LEFT_BOUNDARY}m\.lingdian{_LINGDIAN_HOST_TAIL}", re.I),
    re.compile(r"\[img(?:=[^\]]*)?\].*?\[/img\]", re.I),
    re.compile(r"[（(]月[（(]?费[#＃]?群\d{3,8}[)）]?\d{0,8}[)）]?\"?\d{0,6}[)）]?"),
    re.compile(r"[（(]看暴爽小说[，,]就上飞卢小说网[！!][）)]"),
    re.compile(r"[（(]飞卢小说网独家签约小说[：:][^）)]{0,80}[）)]"),
    re.compile(r"飞卢小说网提醒您[：:]本小说及人物纯属虚构[，,]如有雷同[，,]纯属巧合[，,]切勿模仿[。！!]?"),
    re.compile(
        r"[ \t\u3000]*\.?[ \t\u3000]*首发[^\s。，,！？\"”’」『』]{0,8}网址[:：\s]*"
        r"[hｈ](?:[tｔ](?:[tｔ](?:[pｐ][sｓ]?)?)?)?(?::/?/?[^\s\u4e00-\u9fff。，,！？\"”’」『』]{0,45})?(?=$|[ \t\u3000]*$)",
        re.I,
    ),
]

_UNPREFIXED_LINGDIAN_INDICES = {3, 4, 5}
_ENCLOSING_URL_RE = re.compile(
    r"(?:[a-z][a-z0-9+.\-]*://|(?<![A-Za-z0-9:/])//)[^\s\u4e00-\u9fff。，、；：！？“”‘’「」『』（）【】《》<>,;'\")\]}/?#]+"
    r"(?:[/\?#][^\s\u4e00-\u9fff。，、；：！？“”‘’「」『』（）【】《》<>]*(?:[\u4e00-\u9fff]+[^\s。，、；：！？“”‘’「」『』（）【】《》<>]*)*)?",
    re.I,
)
_IS_LINGDIAN_HOST_RE = re.compile(
    rf"^m\.lingdian(?:[a-z0-9ｚzｗwｃcｏoｍm]{{0,12}}\.{_LINGDIAN_TLD})?$",
    re.I,
)


def _non_lingdian_enclosing_url_spans(line: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for um in _ENCLOSING_URL_RE.finditer(line):
        raw_u = um.group().rstrip(".,;!?'\")]}")
        parse_u = f"http:{raw_u}" if raw_u.startswith("//") else raw_u
        try:
            parts = urlsplit(parse_u)
            host = (parts.hostname or "").lower()
            netloc = parts.netloc or ""
            has_userinfo = "@" in netloc
            has_port = parts.port is not None or (":" in netloc and not netloc.endswith("]"))
        except ValueError:
            spans.append((um.start(), um.end()))
            continue
        if has_userinfo or has_port or not _IS_LINGDIAN_HOST_RE.fullmatch(host):
            spans.append((um.start(), um.end()))
    return spans


def find_inline_ad_spans(line: str) -> list[tuple[int, int]]:
    """Find non-overlapping inline ad spans with RFC-3986 enclosing-URL host isolation."""
    blocked_url_spans = _non_lingdian_enclosing_url_spans(line)
    sh = build_shadow(line)
    sh_blocked_url_spans = _non_lingdian_enclosing_url_spans(sh.text)
    cand_spans: list[tuple[int, int]] = []
    for p_idx, pat in enumerate(INLINE_AD_PATTERNS):
        is_unprefixed_lingdian = p_idx in _UNPREFIXED_LINGDIAN_INDICES
        for m in pat.finditer(line):
            st_c, en_c = m.start(), m.end()
            if en_c - st_c < 4:
                continue
            if is_unprefixed_lingdian and any(
                not (en_c <= u_st or st_c >= u_en) for u_st, u_en in blocked_url_spans
            ):
                continue
            cand_spans.append((st_c, en_c))
        if sh.text != line:
            for m in pat.finditer(sh.text):
                st_s, en_s = m.start(), m.end()
                if en_s - st_s < 4:
                    continue
                if is_unprefixed_lingdian and any(
                    not (en_s <= u_st or st_s >= u_en) for u_st, u_en in sh_blocked_url_spans
                ):
                    continue
                o_st, o_en = sh.to_orig_span(st_s, en_s)
                if o_en - o_st >= 4:
                    cand_spans.append((o_st, o_en))
    chosen: list[tuple[int, int]] = []
    for st_c, en_c in sorted(cand_spans, key=lambda x: (x[0], -x[1])):
        if any(not (en_c <= st or st_c >= en) for st, en in chosen):
            continue
        chosen.append((st_c, en_c))
    return chosen


@lru_cache(maxsize=8192)
def char_digit_info(shadow_ch: str) -> tuple[str | None, bool]:
    """Return (digit_char_or_none, is_strict) for a simplified shadow CJK char."""
    if shadow_ch in CHAR_DIGIT_OVERRIDE:
        return CHAR_DIGIT_OVERRIDE[shadow_ch], True
    if shadow_ch in DIRECT_CJK_DIGIT:
        return DIRECT_CJK_DIGIT[shadow_ch], True
    if shadow_ch in RADICAL_CJK_DIGIT:
        return RADICAL_CJK_DIGIT[shadow_ch], True
    if not ("\u4e00" <= shadow_ch <= "\u9fff" or "\u3400" <= shadow_ch <= "\u4dbf"):
        return None, False
    py_list = pinyin(shadow_ch, style=Style.NORMAL, heteronym=False, errors="ignore")
    if not py_list or not py_list[0]:
        return None, False
    py = py_list[0][0].lower()
    if py in PY_STRICT_DIGIT:
        return PY_STRICT_DIGIT[py], True
    if py in PY_BROAD_DIGIT:
        return PY_BROAD_DIGIT[py], False
    return None, False


def _roman_to_digits(token: str) -> str:
    table = {
        "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
        "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10",
    }
    return table.get(token, "")


@dataclass(frozen=True)
class SoupAnalysis:
    line_num: int
    shadow_text: str
    decoded_digits: str
    score: float
    signals: tuple[str, ...]
    other_cjk: int
    strict_digits: int
    is_hard_protected: bool
    is_high_conf: bool
    is_borderline: bool


def analyze_soup_line(line_num: int, orig_line: str, shadow: ShadowLine | None = None) -> SoupAnalysis:
    """Score whether a standalone line is an encoded group-number soup line."""
    sh = shadow or build_shadow(orig_line)
    s = sh.text.strip()
    if not s:
        return SoupAnalysis(line_num, s, "", 0.0, (), 0, 0, True, False, False)

    if is_chapter_heading(orig_line.strip()) or is_chapter_heading(s):
        return SoupAnalysis(line_num, s, "", 0.0, (), 0, 0, True, False, False)
    if CHAPTER_SPLIT_MARKER_RE.match(s):
        return SoupAnalysis(line_num, s, "", 0.0, (), 0, 0, True, False, False)
    # Protect legitimate scene dividers (*, ***, --- when short? wait ----- is ad banner below, * and *** are scene dividers)
    if re.fullmatch(r"[\*\s·~—…]+", s):
        return SoupAnalysis(line_num, s, "", 0.0, (), 0, 0, True, False, False)

    # Protect normal long prose, paired dialogue quotes, ellipsis/dash prose, multi-percentage/list lines
    if PAIRED_DIALOGUE_QUOTE_RE.search(orig_line):
        return SoupAnalysis(line_num, s, "", 0.0, (), 99, 0, True, False, False)

    # Known pure banner/ad lines get immediate high-conf score BEFORE length/dash prose guards
    stripped_quotes = s.strip("\"'“”‘’「」 ")
    for pat in FULL_LINE_AD_PATTERNS:
        if pat.match(stripped_quotes):
            return SoupAnalysis(line_num, s, "", 1.0, ("ad_banner",), 0, 0, False, True, False)

    if len(s) > 78:
        return SoupAnalysis(line_num, s, "", 0.0, (), 99, 0, True, False, False)
    if "……" in orig_line or "——" in orig_line:
        return SoupAnalysis(line_num, s, "", 0.0, (), 99, 0, True, False, False)
    if len(re.findall(r"\d+(?:\.\d+)?%", s)) >= 2:
        return SoupAnalysis(line_num, s, "", 0.0, (), 99, 0, True, False, False)
    if MID_INDENT_PROSE_RE.search(orig_line):
        return SoupAnalysis(line_num, s, "", 0.0, (), 99, 0, True, False, False)
    if (s.count("、") >= 2 or ("、" in s and "…" in s)) and not any(
        ch in RARE_SOUP_SIGNALS or ch in CIRCLED_MAP or ch in HARD_NOISE_SYM_CHARS for ch in orig_line
    ):
        return SoupAnalysis(line_num, s, "", 0.0, (), 99, 0, True, False, False)

    # Tokenize left to right on shadow text
    digits: list[str] = []
    signals: set[str] = set()
    strict_digits = 0
    broad_digits = 0
    group_kw_cnt = 0
    other_cjk_chars: list[str] = []
    noise_cnt = 0
    rare_cnt = 0
    sent_punct_cnt = 0

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        orig_ch = orig_line.strip()[i] if i < len(orig_line.strip()) else ch
        if orig_ch in RARE_SOUP_SIGNALS or ch in RARE_SOUP_SIGNALS:
            rare_cnt += 1
            signals.add("rare_cjk")
        if ch.isspace():
            i += 1
            continue
        if ch in CIRCLED_MAP:
            digits.append(CIRCLED_MAP[ch])
            strict_digits += 1
            signals.add("circled")
            i += 1
            continue
        if "0" <= ch <= "9":
            digits.append(ch)
            strict_digits += 1
            signals.add("ascii_digit")
            i += 1
            continue
        # Roman numerals (only if bounded away from normal lowercase letters)
        rm = ROMAN_TOKEN_RE.match(s, i)
        if rm:
            rval = _roman_to_digits(rm.group())
            if rval:
                digits.append(rval)
                strict_digits += 1
                signals.add("roman")
                i = rm.end()
                continue
        # English words vs pinyin digit sequences
        if ch.isascii() and ch.isalpha():
            wm = re.match(r"[A-Za-z]+", s[i:])
            assert wm is not None
            word = wm.group()
            if word.lower() in KEEP_ENGLISH_WORDS:
                i += len(word)
                continue
            # Try greedy pinyin digit segmentation
            pos = 0
            matched_py = False
            while pos < len(word):
                pm = LATIN_PINYIN_RE.match(word, pos)
                if pm:
                    d = PY_STRICT_DIGIT.get(pm.group().lower())
                    if d is not None:
                        digits.append(d)
                        strict_digits += 1
                        signals.add("latin_pinyin")
                        pos = pm.end()
                        matched_py = True
                        continue
                break
            if matched_py and pos == len(word):
                i += len(word)
                continue
            # Single stray Latin letter in noisy short line ('O' often stands for 0)
            if len(word) == 1:
                if word in ("O", "o"):
                    digits.append("0")
                signals.add("stray_latin")
                i += 1
                continue
            i += len(word)
            continue

        if ch in SENTENCE_END_PUNCTS:
            sent_punct_cnt += 1
            i += 1
            continue
        if ch in NOISE_PUNCTS or ch in ",;":
            noise_cnt += 1
            if ch in HARD_NOISE_SYM_CHARS:
                signals.add("hard_sym")
            i += 1
            continue

        # Handle split-character 君...羊 = 群
        if ch == "君" and i + 1 < n and s[i + 1] == "羊":
            group_kw_cnt += 1
            signals.add("group_kw")
            i += 2
            continue
        if ch in GROUP_KEYWORDS:
            group_kw_cnt += 1
            signals.add("group_kw")
            i += 1
            continue

        d, is_strict = char_digit_info(ch)
        if d is not None:
            digits.append(d)
            if is_strict:
                strict_digits += 1
                signals.add("cjk_homophone")
            else:
                broad_digits += 1
                signals.add("cjk_broad")
            i += 1
            continue

        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            other_cjk_chars.append(ch)
            i += 1
            continue

        i += 1

    decoded = "".join(digits)
    total_cjk = strict_digits + broad_digits + group_kw_cnt + len(other_cjk_chars)
    other_cjk = len(other_cjk_chars)

    # Short pure-symbol/rare-char junk check (never clear percentages, dialogue quotes, kaomoji, short ASCII like 1!, or speaker:emote)
    is_pct = bool(re.fullmatch(r"\d+(?:\.\d+)?%", s))
    is_quoted = s.startswith(("“", "\"", "「"))
    is_kaomoji = bool(re.search(r"[\^>_T=][-_][\^<_T=]", s))
    has_standalone_sym = any(c in "^`{}[]|\\<>$#@&〃⊙" for c in s)
    if (
        (len(s) <= 3 and has_standalone_sym and other_cjk == 0 and sent_punct_cnt == 0 and not is_pct and not is_quoted and not is_kaomoji)
        or (
            len(s) <= 8
            and other_cjk <= 1
            and sent_punct_cnt == 0
            and not is_pct
            and not is_quoted
            and not is_kaomoji
            and (rare_cnt >= 1 or group_kw_cnt >= 1 or "circled" in signals)
            and ("hard_sym" in signals or rare_cnt >= 1)
        )
        or (s in {"群", "裙", "羣", "峮", "裠"})
    ):
        return SoupAnalysis(line_num, s, decoded, 0.95, tuple(sorted(signals)), other_cjk, strict_digits, False, True, False)

    # Standalone number/watermark soup scoring
    digit_len = len(decoded)
    if digit_len < 4 and rare_cnt == 0 and group_kw_cnt == 0:
        return SoupAnalysis(line_num, s, decoded, 0.0, tuple(sorted(signals)), other_cjk, strict_digits, False, False, False)

    # Only run jieba segmentation on lines that actually have enough digit/soup signals to be candidates
    cjk_only = "".join(re.findall(r"[\u4e00-\u9fff]+", s))
    if len(cjk_only) >= 4 and other_cjk <= 8:
        multi_word_chars = sum(len(w) for w in jieba.cut(cjk_only) if len(w) >= 2)
        lexicon_ratio = multi_word_chars / len(cjk_only)
    else:
        lexicon_ratio = 0.0

    score = 0.0
    if 6 <= digit_len <= 13:
        score += 0.35
    elif 4 <= digit_len <= 16:
        score += 0.20

    if strict_digits >= 5:
        score += 0.20
    elif strict_digits >= 3:
        score += 0.12

    score += min(0.24, 0.06 * len(signals))
    if rare_cnt >= 1:
        score += min(0.20, 0.08 * rare_cnt)
    if group_kw_cnt >= 1:
        score += 0.12
    if noise_cnt >= 2:
        score += min(0.15, 0.04 * noise_cnt)
    if sent_punct_cnt == 0:
        score += 0.08
    elif len(signals) >= 3 or "circled" in signals or rare_cnt >= 1:
        # Stray sentence punctuation injected into watermark soup (e.g. 斯氿棋！ / 淋私？)
        score -= 0.05 * sent_punct_cnt
    else:
        score -= 0.25 * sent_punct_cnt

    if other_cjk == 0 and (strict_digits + broad_digits + group_kw_cnt) >= 4:
        score += 0.12
    else:
        score -= 0.14 * other_cjk

    # Heavy penalty when line parses as fluent multi-char Chinese prose without strong signals
    if lexicon_ratio >= 0.70 and rare_cnt == 0 and "circled" not in signals and "hard_sym" not in signals:
        score -= 0.45

    score = max(0.0, min(1.0, round(score, 4)))

    # Protect paragraph-indented lines containing genuine non-digit prose + sentence punctuation
    if orig_line.startswith("\u3000\u3000") and other_cjk >= 1 and sent_punct_cnt >= 1:
        return SoupAnalysis(
            line_num=line_num,
            shadow_text=s,
            decoded_digits=decoded,
            score=score,
            signals=tuple(sorted(signals)),
            other_cjk=other_cjk,
            strict_digits=strict_digits,
            is_hard_protected=True,
            is_high_conf=False,
            is_borderline=False,
        )

    # Hard guard: never auto-clear a line with too much non-digit prose or high lexicon coherence w/o mixed signals
    is_high_conf = (
        score >= 0.65
        and other_cjk <= 2
        and len(signals) >= 2
        and (digit_len >= 6 or rare_cnt >= 2 or "ad_banner" in signals or (rare_cnt >= 1 and digit_len >= 4))
    )
    is_borderline = (not is_high_conf) and (
        score >= 0.35 or (digit_len >= 6 and other_cjk <= 3 and len(signals) >= 2)
    )
    return SoupAnalysis(
        line_num=line_num,
        shadow_text=s,
        decoded_digits=decoded,
        score=score,
        signals=tuple(sorted(signals)),
        other_cjk=other_cjk,
        strict_digits=strict_digits,
        is_hard_protected=False,
        is_high_conf=is_high_conf,
        is_borderline=is_borderline,
    )


def _edit_dist_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) <= 1
    if la < lb:
        a, b = b, a
    # la == lb + 1
    i = j = diffs = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            diffs += 1
            if diffs > 1:
                return False
            i += 1
    return True


def _matches_cluster(decoded: str, clusters: list[str]) -> str | None:
    if len(decoded) < 5 or not clusters:
        return None
    for center in clusters:
        if center in decoded or decoded in center:
            if min(len(decoded), len(center)) >= 6:
                return center
        if len(decoded) == len(center) and _edit_dist_le1(decoded, center):
            return center
        # Sliding substring window of len(center) inside decoded
        if len(decoded) > len(center):
            w = len(center)
            for k in range(len(decoded) - w + 1):
                if _edit_dist_le1(decoded[k : k + w], center):
                    return center
    return None


_TG_RULE_RE = re.compile(r"^━{6,}$")
_TG_SIGNAL_URL_RE = re.compile(r"t\.me/[a-z0-9_]{5,}", re.I)
_TG_SIGNAL_DISC_RE = re.compile(r"免责声明|版权.{0,8}删除|仅供.{0,12}(?:交流|试读|学习)|24小时内(?:删除|自觉删除)")
_TG_SIGNAL_CHAN_RE = re.compile(
    r"交流群|搜书机器人|搜索机器人|防失联|菠萝包|刺猬猫|电报群|电报频道|telegram|资源群|整理分享|小说频道|推书频道|群号[:：\s]*\d",
    re.I,
)
_TG_LINE_KW_RE = re.compile(
    r"t\.me/|https?://|免责声明|版权.{0,8}删除|24小时|仅供.{0,12}(?:交流|试读|学习)|交流群|搜书|菠萝包|刺猬猫|轻小说(?:资源|整理)|整理分享|防失联|telegram|电报(?:群|频道)|书友群|资源群|小说频道|推书频道|群号|公众号|@",
    re.I,
)


def is_telegram_promo_block(inner_lines: list[str]) -> bool:
    """Return True only if lines between divider bars structurally form a Telegram promo/disclaimer banner."""
    cleaned = [ln.strip() for ln in inner_lines if ln.strip() and not _TG_RULE_RE.match(ln.strip())]
    if not cleaned:
        return False
    block_txt = "\n".join(cleaned)
    if PAIRED_DIALOGUE_QUOTE_RE.search(block_txt):
        return False
    sig_count = sum(
        bool(rx.search(block_txt))
        for rx in (_TG_SIGNAL_URL_RE, _TG_SIGNAL_DISC_RE, _TG_SIGNAL_CHAN_RE)
    )
    if sig_count < 2:
        return False
    for s in cleaned:
        if _TG_LINE_KW_RE.search(s):
            continue
        if len(s) <= 36 and not s.endswith(("。", "！", "？", "”", "」")):
            continue
        return False
    return True


def _detect_telegram_promo_block_lines(lines: list[str]) -> set[int]:
    """Detect Telegram channel disclaimer blocks bounded by `━━━━━━` matching structural promo criteria."""
    rule_indices = [idx for idx, ln in enumerate(lines) if _TG_RULE_RE.match(ln.strip())]
    to_clear: set[int] = set()
    i = 0
    while i < len(rule_indices) - 1:
        start_idx = rule_indices[i]
        end_idx = rule_indices[i + 1]
        if 2 <= end_idx - start_idx <= 24:
            if is_telegram_promo_block(lines[start_idx + 1 : end_idx]):
                for pos in range(start_idx, end_idx + 1):
                    if lines[pos].strip():
                        to_clear.add(pos + 1)  # 1-based line num
                i += 2
                continue
        i += 1
    return to_clear


def detect_soup_and_ads(lines: list[str]) -> tuple[list[EditOp], list[dict], list[str], set[int]]:
    """Run two-pass group-soup + banner detection across a book.

    Returns:
      edits: approved EditOps (`clear_line` and inline `del` for known banner strings)
      review_rows: borderline soup rows that did not meet auto-clear threshold
      clusters: discovered recurring group-number cluster centers for this book
      soup_line_nums: 1-based line numbers classified as group soup (for book-level enrichment)
    """
    tg_clear_lines = _detect_telegram_promo_block_lines(lines)
    analyses: list[SoupAnalysis] = []
    inline_ad_edits: list[EditOp] = []
    soup_line_nums: set[int] = set()

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if idx in tg_clear_lines:
            continue
        sh = build_shadow(line)
        an = analyze_soup_line(idx, line, shadow=sh)
        analyses.append(an)

        # Check explicit inline ad substrings on non-clear-line candidates (safe even on prose because URL/domain is required)
        if not an.is_high_conf:
            for st_c, en_c in find_inline_ad_spans(line):
                inline_ad_edits.append(
                    EditOp(
                        line=idx,
                        start=st_c,
                        end=en_c,
                        op="del",
                        before=line[st_c:en_c],
                        after="",
                        tier="ad",
                        reason="inline_site_ad_pattern",
                        source="rule:group_soup",
                        status="approved",
                    )
                )

    # Stage 2: Extract recurring group-number clusters from high-confidence soup lines
    seed_counts: Counter[str] = Counter()
    for an in analyses:
        if an.is_high_conf and 6 <= len(an.decoded_digits) <= 12:
            seed_counts[an.decoded_digits] += 1

    # Extract 6-11 char n-grams from high-confidence decoded strings to cluster noisy variants & 6-digit seeds
    ngram_counts: Counter[str] = Counter()
    for an in analyses:
        d = an.decoded_digits
        if an.is_high_conf and len(d) >= 6:
            if 6 <= len(d) <= 11:
                ngram_counts[d] += 1
            for length in (8, 9, 10):
                if len(d) > length:
                    for k in range(len(d) - length + 1):
                        ngram_counts[d[k : k + length]] += 1

    clusters: list[str] = [
        num
        for num, cnt in ngram_counts.most_common(30)
        if (cnt >= 3 and 6 <= len(num) <= 11) or (cnt >= 2 and 8 <= len(num) <= 11)
    ]
    # Deduplicate overlapping substrings keeping highest count
    filtered_clusters: list[str] = []
    for c in clusters:
        if not any(c in existing or existing in c for existing in filtered_clusters):
            filtered_clusters.append(c)

    edits: list[EditOp] = [
        EditOp(
            line=ln,
            start=0,
            end=len(lines[ln - 1]),
            op="clear_line",
            before=lines[ln - 1],
            after="",
            tier="ad",
            reason="telegram_promo_banner",
            source="rule:group_soup",
            status="approved",
        )
        for ln in sorted(tg_clear_lines)
    ]
    review_rows: list[dict] = []
    cleared_lines: set[int] = set(tg_clear_lines)

    for an in analyses:
        if an.is_hard_protected:
            continue
        orig_line = lines[an.line_num - 1]
        if an.is_high_conf:
            cleared_lines.add(an.line_num)
            soup_line_nums.add(an.line_num)
            edits.append(
                EditOp(
                    line=an.line_num,
                    start=0,
                    end=len(orig_line),
                    op="clear_line",
                    before=orig_line,
                    after="",
                    tier="soup" if "ad_banner" not in an.signals else "ad",
                    reason="group_soup_high_conf:" + "+".join(an.signals),
                    source="rule:group_soup",
                    status="approved",
                    confidence=an.score,
                )
            )
            continue

        matched_cluster = _matches_cluster(an.decoded_digits, filtered_clusters)
        exact_bare_cluster = bool(
            matched_cluster
            and an.other_cjk == 0
            and (
                (matched_cluster == an.decoded_digits and len(an.decoded_digits) >= 6)
                or (len(an.decoded_digits) >= 8 and matched_cluster in an.decoded_digits)
            )
        )
        if matched_cluster and (
            exact_bare_cluster
            or (an.other_cjk <= 3 and len(an.signals) >= 2 and an.score >= 0.25)
        ):
            cleared_lines.add(an.line_num)
            soup_line_nums.add(an.line_num)
            edits.append(
                EditOp(
                    line=an.line_num,
                    start=0,
                    end=len(orig_line),
                    op="clear_line",
                    before=orig_line,
                    after="",
                    tier="soup",
                    reason=f"group_soup_cluster_match:{matched_cluster}",
                    source="rule:group_soup_cluster",
                    status="approved",
                    confidence=max(an.score, 0.85),
                )
            )
            continue

        if an.is_borderline:
            soup_line_nums.add(an.line_num)
            review_rows.append(
                {
                    "line": an.line_num,
                    "reason": "borderline_soup_line",
                    "score": an.score,
                    "decoded": an.decoded_digits,
                    "signals": list(an.signals),
                    "original": orig_line,
                }
            )

    # Keep only non-overlapping inline ad edits on lines that weren't cleared
    seen_spans: set[tuple[int, int, int]] = set()
    for e in sorted(inline_ad_edits, key=lambda x: (x.line, x.start, -x.end)):
        if e.line in cleared_lines:
            continue
        # Ensure non-overlapping per line
        if any(e.line == ln and not (e.end <= st or e.start >= en) for ln, st, en in seen_spans):
            continue
        seen_spans.add((e.line, e.start, e.end))
        edits.append(e)

    return edits, review_rows, filtered_clusters, soup_line_nums


def detect_explicit_ads_only(lines: list[str]) -> list[EditOp]:
    """Conservative pass for clean-graded books: remove only explicit site ad lines/banners."""
    tg_clear_lines = _detect_telegram_promo_block_lines(lines)
    edits: list[EditOp] = []
    for idx, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            continue
        if idx in tg_clear_lines:
            edits.append(
                EditOp(
                    line=idx,
                    start=0,
                    end=len(line),
                    op="clear_line",
                    before=line,
                    after="",
                    tier="ad",
                    reason="telegram_promo_banner",
                    source="rule:clean_ads",
                    status="approved",
                )
            )
            continue
        if (
            not PAIRED_DIALOGUE_QUOTE_RE.search(line)
            and not is_chapter_heading(s)
            and any(p.match(s) for p in FULL_LINE_AD_PATTERNS)
        ):
            edits.append(
                EditOp(
                    line=idx,
                    start=0,
                    end=len(line),
                    op="clear_line",
                    before=line,
                    after="",
                    tier="ad",
                    reason="explicit_site_ad_line",
                    source="rule:clean_ads",
                    status="approved",
                )
            )
            continue
        for st_c, en_c in find_inline_ad_spans(line):
            edits.append(
                EditOp(
                    line=idx,
                    start=st_c,
                    end=en_c,
                    op="del",
                    before=line[st_c:en_c],
                    after="",
                    tier="ad",
                    reason="explicit_inline_ad",
                    source="rule:clean_ads",
                    status="approved",
                )
            )
    return edits
