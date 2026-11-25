"""Layer 3: Multi-pass inline anti-censor & watermark pre-peeling, book self-learning, and protection.

Implements:
- Word-internal anti-censor strip (`和^谐` -> `和谐`, `豪`乳` -> `豪乳`).
- Tail group-soup strip (`"我带你去。"外:裙)...` -> `"我带你去。"`).
- Yueyi inline ad/brand strip (`-月!@漪#$首%^发&`) and invisible/Cyrillic anchor strip.
- Pass 1: Non-CJK inline noise (hard symbols, circled numerals, stray non-whitelisted Latin) on dirty lines.
- Pass 2: Tier-R rare/traditional watermark CJK on simplified-majority books (respecting mined proper nouns).
- Pass 3: Book-level self-learning (Group A vs Group B enrichment, 2-4 char proper-noun mining,
  post-peel jieba lexicon & numeral soft protection, and +/-10 char contiguous block window tagging
  around Yueyi anchor clusters -> Tier-N vs Tier-C candidate tagging).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import jieba

from .edit_log import EditOp, chapter_heading_prefix_end, is_chapter_heading
from .group_soup import KEEP_ENGLISH_WORDS, _ENCLOSING_URL_RE, char_digit_info
from .shadow import ShadowLine, build_shadow

jieba.setLogLevel(60)

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")

# Word-internal anti-censor markers between two CJK characters
WORD_INTERNAL_CENSOR_RE = re.compile(r"(?<=[\u4e00-\u9fff\u3400-\u4dbf])[\^`](?=[\u4e00-\u9fff\u3400-\u4dbf])")

# Invisible / Bidi format chars (preserve \u200d ZWJ only when joining two emoji)
YUEYI_INVISIBLE_RE = re.compile(
    r"[\u200b-\u200c\u200e-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad\u034f\u061c\u180e\u0337]|"
    r"(?<![\U0001F300-\U0001FAFF])\u200d|\u200d(?![\U0001F300-\U0001FAFF])"
)

# Exotic non-CJK homoglyph & math-alnum anchors that never appear legitimately in Chinese web novels
YUEYI_HARD_ANCHOR_RE = re.compile(
    r"(?:[\U0001d400-\U0001d7ff\U00010280-\U0001034f"
    r"\u0100-\u024f\u0250-\u02af\u0488\u0489\u0500-\u05ff\u10a0-\u10ff"
    r"\u1200-\u16ff\u19d0-\u19df\u1d00-\u1eff\u1f00-\u1fff"
    r"\u2113\u212f\u2160-\u2188\u2205\u2206\u2373\u2474-\u24ff\u2776-\u2793"
    r"\u3021-\u302f\u3036\u303b\ua720-\uabbf]|"
    r"(?<![\u0400-\u04ff])[\u0400-\u04ff]{1,2}(?![\u0400-\u04ff])|"
    r"[\u0370-\u03ff](?!粒子|射线|衰变|波|值|蛋白|函数))"
)

# Union pattern for backward-compatible callers and taint checks
YUEYI_NOISE_CHAR_RE = re.compile(
    rf"(?:{YUEYI_INVISIBLE_RE.pattern}|{YUEYI_HARD_ANCHOR_RE.pattern})"
)
YUEYI_BRAND_AD_RE = re.compile(
    r"-?月[^\u4e00-\u9fff\r\n]{0,5}漪[^\u4e00-\u9fff\r\n]{0,5}首[^\u4e00-\u9fff\r\n]{0,5}发[&@#%^*!！]*|"
    r"(?:月|栎|悦|玥|越|阅)[·\-_~`^/]{0,2}(?:漪|怡|椅)"
)
YUEYI_DIRTY_BRAND_RE = re.compile(r"月漪|栎怡|悦怡|玥漪|越漪|月椅|阅[·\-_~]?[漪怡]|亻尔|衤三")
SAFE_LATIN_SPAN_RE = re.compile(
    r"(?:T恤|S形|X光|san值|SAN值|维他命[A-Za-z]|[A-Z]级|[A-Z]超|[A-Z]型|"
    r"(?<![A-Za-z])(?:Debuff|BUFF|iPhone|iPad|Android|Mr|Mrs|Ms|Dr|Prof|G1|G2|G3|NBA|CPU|GPU|API|BOSS|NPC|VIP|MVP|XPS?|C\+\+|K-12)(?![A-Za-z]))",
    re.I,
)
TAG_LIST_LINE_RE = re.compile(r"^(?:[\u4e00-\u9fff\w？?…]+(?:\|[\u4e00-\u9fff\w？?…]+){3,})$")

# Valid simplified & traditional words that happen to overlap TIER_R_WATERMARK_CJK
_SIMPLIFIED_LEXICON_TIER_R = set("壹贰叁肆伍柒捌玖鸸熘陸靈夢爾")
_TRAD_VARIANT_TIER_R = set("壹貳贰叁叄肆伍陸柒捌玖靈夢爾飼傘鷗")
_STRICT_NON_NAME_SOUP_CJK = set(
    "亻氿紦帬羣裠厁阾镹陾宭踆玐儛叄罒俬熘峮氵彡児侕箘囷岜麇邬轳鹨衤弌弍覇锍诌鏾镏栮翏逡"
)


def _is_traditional_line(line: str, shadow: ShadowLine) -> bool:
    """Return True when `line` contains genuine non-watermark Traditional CJK prose."""
    trad_other = 0
    for orig_ch, sh_ch in zip(line, shadow.text):
        if _is_cjk(orig_ch) and orig_ch != sh_ch and orig_ch not in TIER_R_WATERMARK_CJK:
            trad_other += 1
            if trad_other >= 2:
                return True
    return False


def _lexicon_tier_r_mask(line: str) -> list[bool]:
    """Return per-char mask of indices protected inside multi-char dictionary words."""
    mask = [False] * len(line)
    if not any(ch in _SIMPLIFIED_LEXICON_TIER_R for ch in line):
        return mask
    for m in CJK_RUN_RE.finditer(line):
        run_start = m.start()
        run = m.group()
        if not any(ch in _SIMPLIFIED_LEXICON_TIER_R for ch in run):
            continue
        cursor = 0
        for tok in jieba.cut(run):
            tlen = len(tok)
            if tlen >= 2 and (tok in PROTECT_WORDS or jieba.get_FREQ(tok, 0) > 0):
                for k in range(run_start + cursor, run_start + cursor + tlen):
                    mask[k] = True
            cursor += tlen
        for pw in PROTECT_WORDS:
            if not any(ch in _SIMPLIFIED_LEXICON_TIER_R for ch in pw):
                continue
            p_pos = 0
            while True:
                hit = run.find(pw, p_pos)
                if hit < 0:
                    break
                for k in range(run_start + hit, run_start + hit + len(pw)):
                    mask[k] = True
                p_pos = hit + 1
    return mask


# High-confidence inline hard symbols between CJK on dirty lines
HARD_INLINE_SYM_SET = set("^`{}[]|\\<>$#@&%〃_")
CIRCLED_INLINE_SET = set(
    "⓪"
    + "".join(chr(0x2460 + i) for i in range(0xA0))  # U+2460..U+24FF (①-⑳, ⑴-⒇, ⒈-⒛, ⒜-ⓩ, ⓪, ⓫-⓴, ⓵-⓾, ⓿)
    + "".join(chr(0x2776 + i) for i in range(30))    # ❶-❿, ➀-➉, ➊-➓
    + "".join(chr(0x3021 + i) for i in range(9))     # Hangzhou numerals 〡..〩
)

# Rare/traditional CJK watermark signals for simplified-majority books (Tier-R).
# Must NEVER contain common simplified characters like 灵/梦/三/八/尔/陆/台/群/败/条/等.
TIER_R_WATERMARK_CJK = set(
    "亻氿紦帬羣裠澪厁阾镹陾宭踆玐儛叄罒俬熘峮氵彡児侕箘囷鸸岜麇邬轳鹨衤弌弍覇锍诌"
    "捌玖壹貳贰叁叄肆伍陸柒靈夢爾飼鏾镏蹴鷗傘栮翏逡"
)

# Everyday simplified characters that happen to overlap watermark homophones;
# always verify them under Tier-C consensus rather than single-model Tier-N.
ULTRA_COMMON_PROSE_CJK = set("一二三四五六七八九十是就师起把吧衣尔")

# Common protected multi-char words & legitimate phrases (supplemented by jieba + mined proper nouns)
PROTECT_WORDS = {
    "凌晨", "凌乱", "凌厉", "凌驾", "凌迟", "凌空", "欺凌", "盛气凌人",
    "灵魂", "生灵", "精灵", "机灵", "灵光", "威灵", "灵敏", "灵气", "灵力",
    "猛然", "猛地", "猛烈", "猛兽", "猛扑",
    "做梦", "梦境", "梦见", "梦里", "梦想", "噩梦", "美梦", "梦魇", "梦话", "春梦", "旧梦", "入梦", "发梦", "托梦", "圆梦", "如梦", "好梦", "梦到", "梦中",
    "队伍", "入伍", "行伍", "军伍", "配伍", "落伍", "退伍", "为伍", "伍长", "酒肆", "茶肆", "市肆", "商肆",
    "肆意妄为", "肆无忌惮", "大肆", "放肆", "肆意", "肆虐",
    "鸸鹋", "滴熘熘", "酸熘熘", "灰熘熘", "滑熘熘", "直熘熘", "圆熘熘", "黑熘熘", "光熘熘",
    "败犬", "三条", "一起", "起来", "冷静", "群体", "群聊", "群众", "人群", "一蹴而就",
    "以前", "以后", "伊始", "零点", "零食", "零件", "零碎", "零星", "零下",
    "衣服", "衣襟", "衣裳", "医院", "医生", "医药", "仪式", "仪器",
    "其余", "其他", "其实", "奇葩", "奇怪", "好奇", "奇迹", "气味", "气氛", "气候",
    "企业", "企图", "起码", "起身", "起床", "起飞", "起来", "留恋", "留意", "停留",
    "世界", "四十五", "时候", "时间", "为什么", "什么", "这么", "那么", "怎么",
    "观察", "琢磨", "攻略", "方法", "肯定", "确定", "唯一", "感觉", "情况", "后果",
    "讨厌", "崇拜", "耍帅", "凝重", "学习", "提升", "喜欢",
}

LEGIT_NUMERAL_UNIT_RE = re.compile(
    r"^[零〇一二三四五六七八九十百千万亿两]+(?:楼|层|岁|天|年|月|日|号|次|个|名|种|位|件|步|章|回|卷|条|只|把|张|根|片|份|颗|枚|株|门|本|座|倍|成|分|点|米|里|斤|两|块|元|万|千|百|十|间|声|下|定|直|样|般|双|对|排|行|阵|道|股|团|丝|缕|抹|滴|口|杯|碗|盒|袋|包|辆|艘|架|匹|头|尾|封|节|段|句|番|遍|趟|顿|场|局|轮|圈|寸|尺|丈)"
)

LATIN_WATERMARK_PINYIN_RE = re.compile(
    r"^(?:ling|liang|jiu|san|liu|yao|er|yi|si|wu|qi|ba|qun)+$", re.I
)

STOP_NGRAM_CHARS = set(
    "的了是在我有他和她它们这那不也都要就说看到去来着过得地会能把被让给从对为与跟并而但可因所如果又还很最更已已经刚才却只真好小大上下前后左右中里内外"
    "一个两几多些什怎么时候事情自己于吗呢吧啊呀哦嗯之没摇笑入叹问回"
)


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"


def _strip_end_group_soup_span(line: str) -> tuple[int, int] | None:
    """Detect trailing group-number soup glued after closing quote or sentence punctuation."""
    s = line.rstrip()
    if len(s) < 6:
        return None
    closer = -1
    for i in range(len(s) - 1, 0, -1):
        if s[i] not in "”」\"'":
            continue
        j = i - 1
        if j >= 0 and s[j] in "。！？":
            j -= 1
        if j >= 0 and _is_cjk(s[j]) and s[j] not in "零〇一二三四五六七八九十外裙":
            closer = i
            break
    if closer >= 0 and closer + 1 < len(s):
        tail = s[closer + 1 :]
        if _looks_like_group_tail(tail):
            return closer + 1, len(line)
    m = re.search(r"([。！？])(外[、.\-·:：)(（]*裙.*)$", s)
    if m and _looks_like_group_tail(m.group(2)):
        return m.start(2), len(line)
    return None


def _looks_like_group_tail(tail: str) -> bool:
    t = tail.lstrip("。")
    t_no_prefix = re.sub(r"^外[、.\-·:：)(（]*裙", "", t)
    for cand in (t, t_no_prefix):
        if len(cand) < 4 or len(cand) > 90:
            continue
        cjk = [c for c in cand if _is_cjk(c)]
        keep_cjk = [c for c in cjk if c in "零〇一二三四五六七八九十外裙群"]
        digits = sum(1 for c in cand if "0" <= c <= "9" or c in CIRCLED_INLINE_SET)
        if len(cjk) == len(keep_cjk) and (digits + len(keep_cjk)) >= 3:
            return True
    return False


def _line_corruption_evidence(orig_line: str, shadow: ShadowLine) -> int:
    """Count independent inline anti-censor corruption signals on a single line."""
    stripped = orig_line.strip()
    if TAG_LIST_LINE_RE.match(stripped) and not YUEYI_NOISE_CHAR_RE.search(stripped):
        return 0

    s = shadow.text
    # Remove legitimate [正常词], kaomoji spans, and safe Latin tokens before scoring
    s_check = _KAOMOJI_SPAN_RE.sub(
        lambda m: "" if any(c in _KAOMOJI_HINT_CHARS for c in m.group()) else m.group(),
        s,
    )
    orig_no_kaomoji = _KAOMOJI_SPAN_RE.sub(
        lambda m: (" " * len(m.group())) if any(c in _KAOMOJI_HINT_CHARS for c in m.group()) else m.group(),
        orig_line,
    )
    s_check = re.sub(r"\[[\u4e00-\u9fff]{1,6}\]", "", s_check)
    s_check = SAFE_LATIN_SPAN_RE.sub("", s_check)
    is_trad = _is_traditional_line(orig_line, shadow)
    lex_mask = _lexicon_tier_r_mask(orig_line)

    score = 0
    if YUEYI_HARD_ANCHOR_RE.search(orig_no_kaomoji) or YUEYI_INVISIBLE_RE.search(orig_no_kaomoji):
        score += 3
    if YUEYI_BRAND_AD_RE.search(orig_line) or YUEYI_DIRTY_BRAND_RE.search(orig_line):
        score += 3
    hard_between = len(re.findall(r"(?<=[\u4e00-\u9fff])[\^`{}\[\]|\\<>$#⊙∞∆∅⍳]+(?=[\u4e00-\u9fff])", s_check))
    if hard_between:
        score += min(3, hard_between)
    circled_between = sum(
        1
        for i, ch in enumerate(s_check)
        if ch in CIRCLED_INLINE_SET
        and ((i > 0 and _is_cjk(s_check[i - 1])) or (i + 1 < len(s_check) and _is_cjk(s_check[i + 1])))
    )
    if circled_between:
        score += min(3, circled_between)
    rare_cnt = sum(
        1
        for i, ch in enumerate(orig_line)
        if ch in TIER_R_WATERMARK_CJK
        and not lex_mask[i]
        and not (is_trad and ch in _TRAD_VARIANT_TIER_R)
    )
    if rare_cnt:
        score += min(4, rare_cnt)
    latin_between = 0
    for m in re.finditer(r"(?<=[\u4e00-\u9fff])[A-Za-z]{1,4}(?=[\u4e00-\u9fff])", s_check):
        tok = m.group()
        if tok.lower() in KEEP_ENGLISH_WORDS:
            continue
        if len(tok) == 1 and (tok.islower() or s_check.count(tok) >= 2):
            latin_between += 1
        elif LATIN_WATERMARK_PINYIN_RE.match(tok):
            # Single short pinyin token like 'san' needs ≥4 chars or another signal before triggering alone
            latin_between += 2 if len(tok) >= 4 else 1
    if latin_between:
        score += min(3, latin_between)
    ascii_p_between = len(re.findall(r"(?<=[\u4e00-\u9fff])[;()\[\]{}](?=[\u4e00-\u9fff])", s_check))
    if ascii_p_between:
        score += min(2, ascii_p_between)
    if re.search(r"月椅|亻尔|衤三", orig_line):
        score += 2
    return score


def detect_simplified_majority(lines: list[str]) -> bool:
    sample = "".join(lines[:3000])[:150_000]
    sh = build_shadow(sample)
    cjk_total = 0
    trad_diff = 0
    for a, b in zip(sample, sh.text):
        if _is_cjk(a):
            cjk_total += 1
            if a != b and not (0xFF10 <= ord(a) <= 0xFF5A):
                trad_diff += 1
    if cjk_total == 0:
        return True
    return (trad_diff / cjk_total) < 0.01


_NARRATIVE_EDGE_CHARS = set("微轻缓忽突猛伸抬转低摇眨皱抿勾嘴唇眉眼手脚头声笑叹看听闻问")


def _mine_proper_nouns(clean_lines: list[str], top_k: int = 200) -> list[str]:
    """Mine recurring book-specific proper nouns (2-4 chars, preferring full names like 顾清霜 / 青云城)."""
    ngram_counts: Counter[str] = Counter()
    # Stride large books so n-gram mining stays sub-second while covering start-to-end chapters evenly
    stride = max(1, len(clean_lines) // 25000)
    sampled = clean_lines[::stride]
    min_hits = 4 if stride > 1 else 6
    for line in sampled:
        if not line.strip() or is_chapter_heading(line.strip()):
            continue
        for run in CJK_RUN_RE.findall(line):
            if len(run) < 2:
                continue
            for length in (2, 3, 4):
                if len(run) < length:
                    continue
                for i in range(len(run) - length + 1):
                    gram = run[i : i + length]
                    if any(ch in STOP_NGRAM_CHARS for ch in gram):
                        continue
                    if any(ch in _STRICT_NON_NAME_SOUP_CJK for ch in gram):
                        continue
                    ngram_counts[gram] += 1

    candidates: list[tuple[int, str]] = []
    for gram, cnt in ngram_counts.most_common(4500):
        if cnt < min_hits:
            continue
        seg = list(jieba.cut(gram))
        single_chars = sum(1 for tok in seg if len(tok) == 1)
        freq = jieba.get_FREQ(gram, 0)
        # Skip common multi-word action/narrative phrases (e.g. 微微点头 / 唇角勾起)
        if len(gram) >= 3 and freq == 0 and len(seg) >= 2 and all(len(t) >= 2 and jieba.get_FREQ(t, 0) >= 200 for t in seg):
            continue
        if len(gram) == 4 and freq == 0 and (gram[0] in _NARRATIVE_EDGE_CHARS or gram[-1] in _NARRATIVE_EDGE_CHARS):
            continue
        if freq > 0 and len(seg) == 1:
            if len(gram) >= 3 and any(c in "伊莉丝莱尔菲亚娜蕾提欧德弗琳斯洛" for c in gram):
                candidates.append((cnt, gram))
            elif (
                len(gram) == 2
                and freq <= 30
                and cnt >= max(25, min_hits * 5)
                and not any(c in "零一二三四五六七八九十群" for c in gram)
            ):
                candidates.append((cnt, gram))
            continue
        if len(gram) >= 3 and freq == 0:
            candidates.append((cnt, gram))
        elif len(gram) == 2 and freq == 0 and (
            (single_chars == 2 and cnt >= min_hits * 2)
            or (len(seg) == 1 and cnt >= max(25, min_hits * 5))
        ) and not any(c in "零一二三四五六七八九十群" for c in gram):
            candidates.append((cnt, gram))

    long_cands = [c for c in candidates if len(c[1]) >= 3]
    short_cands = [c for c in candidates if len(c[1]) == 2]
    long_cands.sort(key=lambda x: (-len(x[1]), -x[0]))
    short_cands.sort(key=lambda x: -x[0])

    max_long = max(0, top_k - min(25, top_k // 4))
    selected: list[str] = []
    for cnt, gram in long_cands:
        if any(gram in existing and cnt < ngram_counts[existing] * 1.45 for existing in selected):
            continue
        selected.append(gram)
        if len(selected) >= max_long:
            break
    for cnt, gram in short_cands:
        if len(selected) >= top_k:
            break
        if any(gram in existing and cnt < ngram_counts[existing] * 1.45 for existing in selected):
            continue
        selected.append(gram)
    return selected


_KAOMOJI_SPAN_RE = re.compile(
    r"(?:[凸oO]\s*)?[（(][^（）()\u4e00-\u9fff\r\n]{1,18}[）)]"
)
_KAOMOJI_HINT_CHARS = set("°ω∀•ᴗ▽ʖ￣╰╯┬╥┗┛∧╭╮﹏︵▂")
_DEC_UNIT_SPAN_RE = re.compile(
    r"\d+\.\d+%?|\d+%|(?<![A-Za-z0-9])\d+(?:m|km|kg|cm|mm|g|h|ms|s|Hz|GHz|MB|GB|KB)(?![A-Za-z])"
)
_CLUSTER_EXTRA_SYM = set("∆∞⊙⍳∅ↀ·-~/_=+'\"")
_EDGE_STOP_PUNCT = set(",，。.!！？?：:；;、…—~“”‘’")
_BRACKET_MATES = {"(": ")", ")": "(", "（": "）", "）": "（", "【": "】", "】": "【", "《": "》", "》": "《"}
_CJK_NUM_CHAR_SET = set("零〇一二三四五六七八九十")


def _heading_protect_end(line: str) -> int:
    """Return protected chapter-heading prefix end (including up to one trailing sep/space), or 0."""
    pend = chapter_heading_prefix_end(line, allow_long_dirty=True)
    if pend <= 0:
        return 0
    n = len(line)
    if pend < n and line[pend] in ":：":
        pend += 1
    if pend < n and line[pend] in " \t\u3000":
        pend += 1
    return pend


_UPPER_ALNUM_PROSE_RE = re.compile(r"(?<![A-Za-z0-9])(?:\d{1,2}[A-Z]{1,3}|[A-Z]{1,4}\d{1,4})(?![A-Za-z0-9])")


def _protected_cluster_mask(line: str, prefix_end: int) -> list[bool]:
    mask = [False] * len(line)
    for k in range(min(prefix_end, len(line))):
        mask[k] = True
    for km in _KAOMOJI_SPAN_RE.finditer(line):
        if any(c in _KAOMOJI_HINT_CHARS for c in km.group()):
            for k in range(km.start(), km.end()):
                mask[k] = True
    for dm in _DEC_UNIT_SPAN_RE.finditer(line):
        for k in range(dm.start(), dm.end()):
            mask[k] = True
    for sm in SAFE_LATIN_SPAN_RE.finditer(line):
        for k in range(sm.start(), sm.end()):
            mask[k] = True
    for am in _UPPER_ALNUM_PROSE_RE.finditer(line):
        for k in range(am.start(), am.end()):
            mask[k] = True
    for um in _ENCLOSING_URL_RE.finditer(line):
        for k in range(um.start(), um.end()):
            mask[k] = True
    return mask


def _peel_yueyi_clusters(
    line: str,
    line_num: int,
    ev: int,
    prefix_end: int,
    tail_end: int,
    is_simplified: bool,
    stable_rare_chars: set[str],
    prot_mask: list[bool],
    tier_r_lex_mask: list[bool],
    is_trad_line: bool,
) -> tuple[list[EditOp], list[int]]:
    """Expand around hard Yueyi exotic anchors to peel adjacent fullwidth digits and mixed ASCII/symbol glue."""
    ops: list[EditOp] = []
    anchors: list[int] = []
    n = min(len(line), tail_end)
    if n <= prefix_end:
        return ops, anchors

    cprot = _protected_cluster_mask(line, prefix_end)
    seeds: list[tuple[int, int]] = []
    for m in YUEYI_HARD_ANCHOR_RE.finditer(line):
        if prefix_end <= m.start() < n and not cprot[m.start()]:
            seeds.append((m.start(), m.end()))
    for m in YUEYI_BRAND_AD_RE.finditer(line):
        if prefix_end <= m.start() < n and not cprot[m.start()]:
            seeds.append((m.start(), m.end()))
    if ev >= 2:
        for m in YUEYI_DIRTY_BRAND_RE.finditer(line):
            if prefix_end <= m.start() < n and not cprot[m.start()] and m.group() not in ("亻尔", "衤三"):
                seeds.append((m.start(), m.end()))
        for i in range(prefix_end, n):
            if not cprot[i] and line[i] in ("∆", "∞", "⊙", "⍳", "∅", "ↀ"):
                seeds.append((i, i + 1))

    if not seeds:
        return ops, anchors

    # Record hard anchors for LLM window expansion
    for st, _ in seeds:
        anchors.append(st)

    line_counts = Counter(line[:n])
    seed_pos_set = {k for st, en in seeds for k in range(st, en)}

    def _is_digit_char(c: str) -> bool:
        return (c.isascii() and c.isdigit()) or ("０" <= c <= "９")

    def _is_unprotected_ascii_alpha_at(idx_a: int) -> bool:
        if not (0 <= idx_a < n) or cprot[idx_a] or not (line[idx_a].isascii() and line[idx_a].isalpha()):
            return False
        lo_a = idx_a
        while lo_a > prefix_end and not cprot[lo_a - 1] and line[lo_a - 1].isascii() and line[lo_a - 1].isalpha():
            lo_a -= 1
        hi_a = idx_a + 1
        while hi_a < n and not cprot[hi_a] and line[hi_a].isascii() and line[hi_a].isalpha():
            hi_a += 1
        tok_a = line[lo_a:hi_a]
        if (
            SAFE_LATIN_SPAN_RE.fullmatch(tok_a)
            or tok_a.lower() in KEEP_ENGLISH_WORDS
            or (tok_a.isupper() and 2 <= len(tok_a) <= 6)
        ):
            return False
        return True

    def _is_mixed_digit_noise(idx_n: int) -> bool:
        if not (0 <= idx_n < n) or cprot[idx_n]:
            return False
        cn = line[idx_n]
        return _is_unprotected_ascii_alpha_at(idx_n) or cn in CIRCLED_INLINE_SET or (0x1D400 <= ord(cn) <= 0x1D7FF)

    def _is_glue_at(pos: int, near_seed_dist: int) -> int:
        """Return span length >=1 if `line[pos]` is absorbable watermark glue, else 0."""
        if pos < prefix_end or pos >= n or cprot[pos]:
            return 0
        ch = line[pos]
        if YUEYI_INVISIBLE_RE.match(ch) or YUEYI_HARD_ANCHOR_RE.match(ch):
            return 1
        # Circled numerals, hard symbols, Yueyi math/symbol glue (full-width digits use noise-adjacency guard below)
        if ch in CIRCLED_INLINE_SET or ch in HARD_INLINE_SYM_SET or ch in _CLUSTER_EXTRA_SYM:
            return 1
        # Isolated `〇`
        if ch == "〇":
            p_c = line[pos - 1] if pos > 0 else ""
            n_c = line[pos + 1] if pos + 1 < n else ""
            if p_c != "〇" and n_c != "〇" and not (p_c in "零〇一二三四五六七八九十百千" and n_c in "零〇一二三四五六七八九十百千"):
                return 1
        # Mixed/stray ASCII letters (never swallow valid/protected English words or uppercase acronyms like IBM/NPC)
        if ch.isascii() and ch.isalpha():
            j = pos
            while j < n and line[j].isascii() and line[j].isalpha() and not cprot[j]:
                j += 1
            tok = line[pos:j]
            if SAFE_LATIN_SPAN_RE.fullmatch(tok) or tok.lower() in KEEP_ENGLISH_WORDS or (tok.isupper() and 2 <= len(tok) <= 6):
                return 0
            if len(tok) <= 6 or pos in seed_pos_set or (pos > 0 and pos - 1 in seed_pos_set):
                return j - pos
            return 0
        # Short digit run (ASCII or full-width 1-2 chars) ONLY when glued to unprotected alpha/circled/math noise
        if _is_digit_char(ch):
            j = pos
            while j < n and _is_digit_char(line[j]) and not cprot[j]:
                j += 1
            adjacent_to_noise = (
                (pos > prefix_end and _is_mixed_digit_noise(pos - 1))
                or (j < n and _is_mixed_digit_noise(j))
            )
            if (j - pos <= 2) and adjacent_to_noise:
                return j - pos
            return 0
        # Multi-char Yueyi brand token or `越已` touching cluster
        for bpat in (YUEYI_BRAND_AD_RE, YUEYI_DIRTY_BRAND_RE):
            bm = bpat.match(line, pos)
            if bm and bm.end() <= n and bm.group() not in ("亻尔", "衤三"):
                return bm.end() - pos
        if line.startswith("越已", pos) and near_seed_dist <= 3:
            return 2
        # Parenthesized single CJK digit like （九） / （三）
        if ch in "（(":
            close_c = "）" if ch == "（" else ")"
            if pos + 2 < n and line[pos + 1] in _CJK_NUM_CHAR_SET and line[pos + 2] == close_c:
                return 3
        # TIER_R watermark char inside/touching cluster
        if (
            is_simplified
            and ch in TIER_R_WATERMARK_CJK
            and ch not in stable_rare_chars
            and not prot_mask[pos]
            and not (ch in _SIMPLIFIED_LEXICON_TIER_R and tier_r_lex_mask[pos])
            and not (is_trad_line and ch in _TRAD_VARIANT_TIER_R)
        ):
            return 1
        # Structural glue chars / brackets inside cluster
        if ch in "·-~/_=+'\",()（）【】《》!！?？":
            return 1
        return 0

    def _is_glue_rev(pos: int, near_seed_dist: int) -> int:
        if pos < prefix_end or pos >= n or cprot[pos]:
            return 0
        # Check parenthesized CJK digit ending at pos
        if line[pos] in "）)" and pos >= prefix_end + 2:
            open_c = "（" if line[pos] == "）" else "("
            if line[pos - 2] == open_c and line[pos - 1] in _CJK_NUM_CHAR_SET:
                return 3
        # Check 2+ char brand ending at pos
        for bpat in (YUEYI_BRAND_AD_RE, YUEYI_DIRTY_BRAND_RE):
            for k in range(max(prefix_end, pos - 12), pos):
                bm = bpat.match(line, k)
                if bm and bm.end() == pos + 1 and bm.group() not in ("亻尔", "衤三"):
                    return bm.end() - bm.start()
        if pos >= prefix_end + 1 and line[pos - 1 : pos + 1] == "越已" and near_seed_dist <= 3:
            return 2
        # Short digit (ASCII or full-width 1-2 chars) ONLY when rev-expanding across unprotected alpha/circled/math noise
        if _is_digit_char(line[pos]):
            k = pos
            while k >= prefix_end and _is_digit_char(line[k]) and not cprot[k]:
                k -= 1
            k += 1
            d_len = pos - k + 1
            adjacent_to_noise = (
                (pos + 1 < n and _is_mixed_digit_noise(pos + 1))
                or (k > prefix_end and _is_mixed_digit_noise(k - 1))
            )
            if d_len <= 2 and adjacent_to_noise:
                return d_len
            return 0
        # Backward ASCII-alpha scan (preserve protected English words and uppercase acronyms like IBM/NPC)
        if line[pos].isascii() and line[pos].isalpha():
            k = pos
            while k >= prefix_end and line[k].isascii() and line[k].isalpha() and not cprot[k]:
                k -= 1
            k += 1
            tok = line[k : pos + 1]
            if SAFE_LATIN_SPAN_RE.fullmatch(tok) or tok.lower() in KEEP_ENGLISH_WORDS or (tok.isupper() and 2 <= len(tok) <= 6):
                return 0
            if (
                len(tok) <= 6
                or any(idx in seed_pos_set for idx in range(k, pos + 1))
                or (pos + 1 < n and pos + 1 in seed_pos_set)
                or (k > prefix_end and (k - 1) in seed_pos_set)
            ):
                return pos - k + 1
            return 0
        # All multi-char backward patterns are handled above; fallback strictly to 1-char glue checks
        return 1 if _is_glue_at(pos, near_seed_dist) == 1 else 0

    raw_intervals: list[tuple[int, int]] = []
    for st, en in seeds:
        if en - st > 1 and YUEYI_BRAND_AD_RE.fullmatch(line[st:en]):
            raw_intervals.append((st, en))
            continue
        l_idx = st
        steps_left = 0
        while l_idx > prefix_end and steps_left < 24:
            if line[l_idx - 1] in " \t\u3000":
                # Bridge space if another glue token lies immediately across 1-2 spaces
                sp = l_idx - 1
                while sp >= prefix_end and line[sp] in " \t\u3000" and (l_idx - sp) <= 3:
                    sp -= 1
                if sp >= prefix_end and _is_glue_rev(sp, st - sp) > 0:
                    l_idx = sp + 1
                    continue
                break
            g_len = _is_glue_rev(l_idx - 1, st - (l_idx - 1))
            if g_len <= 0 or l_idx - g_len < prefix_end:
                break
            l_idx -= g_len
            steps_left += g_len
        r_idx = en
        steps_right = 0
        while r_idx < n and steps_right < 24:
            if line[r_idx] in " \t\u3000":
                sp = r_idx
                while sp < n and line[sp] in " \t\u3000" and (sp - r_idx) < 3:
                    sp += 1
                if sp < n and _is_glue_at(sp, sp - en + 1) > 0:
                    r_idx = sp
                    continue
                break
            g_len = _is_glue_at(r_idx, r_idx - en + 1)
            if g_len <= 0 or r_idx + g_len > n:
                break
            r_idx += g_len
            steps_right += g_len
        raw_intervals.append((l_idx, r_idx))

    # Merge overlapping or space-adjacent clusters
    raw_intervals.sort()
    merged: list[list[int]] = []
    for st, en in raw_intervals:
        if not merged or st > merged[-1][1]:
            merged.append([st, en])
        else:
            merged[-1][1] = max(merged[-1][1], en)

    for st, en in merged:
        # Absorb adjacent inline spaces strictly between CJK chars (e.g. `毒 　ᚱ的话` -> `毒的话`), never indent!
        while st > prefix_end and st > 0 and line[st - 1] in " \t\u3000":
            # Only absorb space if there is a non-space CJK/punct char to the left of the space run
            sp_l = st - 1
            while sp_l >= prefix_end and line[sp_l] in " \t\u3000":
                sp_l -= 1
            if sp_l >= prefix_end and _is_cjk(line[sp_l]) and en < n and _is_cjk(line[en]):
                st = sp_l + 1
            else:
                break
        while en < n and line[en] in " \t\u3000":
            sp_r = en
            while sp_r < n and line[sp_r] in " \t\u3000":
                sp_r += 1
            if st > prefix_end and _is_cjk(line[st - 1]) and sp_r < n and _is_cjk(line[sp_r]):
                en = sp_r
            else:
                break

        # Strict outer-edge pushback: preserve normal punctuation, balanced outer brackets, CJK middle-dot
        changed = True
        while changed and st < en:
            changed = False
            left_surv = line[st - 1] if st > 0 else ""
            right_surv = line[en] if en < n else ""

            # Push back edge spaces unless both outer surviving chars are CJK
            if line[st] in " \t\u3000" and not (_is_cjk(left_surv) and _is_cjk(right_surv)):
                st += 1
                changed = True
                continue
            if en > st and line[en - 1] in " \t\u3000" and not (_is_cjk(left_surv) and _is_cjk(right_surv)):
                en -= 1
                changed = True
                continue

            # Push back outer prose punctuation
            if line[st] in _EDGE_STOP_PUNCT:
                st += 1
                changed = True
                continue
            if en > st and line[en - 1] in _EDGE_STOP_PUNCT:
                en -= 1
                changed = True
                continue

            # Preserve CJK name separator `·` when both sides are CJK
            if line[st] == "·" and _is_cjk(left_surv) and _is_cjk(right_surv):
                st += 1
                changed = True
                continue
            if en > st and line[en - 1] == "·" and _is_cjk(left_surv) and _is_cjk(right_surv):
                en -= 1
                changed = True
                continue

            # Bracket safety: if bracket's mate sits OUTSIDE the cluster on a balanced line, push bracket back!
            for is_left_edge in (True, False):
                bpos = st if is_left_edge else en - 1
                bch = line[bpos]
                if bch in _BRACKET_MATES:
                    mate = _BRACKET_MATES[bch]
                    seg = line[st:en]
                    if line_counts[bch] == line_counts[mate] and seg.count(bch) != seg.count(mate):
                        if is_left_edge:
                            st += 1
                        else:
                            en -= 1
                        changed = True
                        break
            # Also push back isolated trailing/leading `'` or `"` when balanced on the line
            for qch in ("\"", "'"):
                if line[st] == qch and line_counts[qch] % 2 == 0 and line[st:en].count(qch) % 2 == 1:
                    st += 1
                    changed = True
                elif en > st and line[en - 1] == qch and line_counts[qch] % 2 == 0 and line[st:en].count(qch) % 2 == 1:
                    en -= 1
                    changed = True

        if st >= en:
            continue
        sub = line[st:en]
        # Skip single normal ASCII word / digit without any anchor/glue inside
        if sub.isascii() and sub.isalnum() and not any(st <= s_pos < en for s_pos, _ in seeds):
            continue
        ops.append(
            EditOp(
                line=line_num,
                start=st,
                end=en,
                op="del",
                before=sub,
                after="",
                tier="rule_sym",
                reason="yueyi_cluster_glue",
                source="rule:inline_pass1",
                status="approved",
            )
        )
        anchors.append(st)
    return ops, anchors


def cleanup_post_edit_artifacts(lines: list[str], edits: list[EditOp]) -> list[EditOp]:
    """Safely remove empty brackets (`（）`/`()`) and stray CJK spaces created by approved deletions."""
    by_line: dict[int, list[EditOp]] = {}
    for e in edits:
        by_line.setdefault(e.line, []).append(e)

    extra_ops: list[EditOp] = []
    empty_b_re = re.compile(r"(?:\([ \t\u3000\-·—~]*\)|（[ \t\u3000\-·—~]*）|【[ \t\u3000]*】|《[ \t\u3000]*》)")
    stray_sp_re = re.compile(r"[ \t\u3000]{1,4}")

    for ln, ops in by_line.items():
        if ln < 1 or ln > len(lines):
            continue
        orig = lines[ln - 1]
        if any(e.op == "clear_line" and e.status == "approved" for e in ops):
            continue
        del_ops = [e for e in ops if e.status == "approved" and e.op == "del" and e.end > e.start]
        if not del_ops:
            continue
        approved_spans = sorted((e.start, e.end) for e in ops if e.status == "approved" and e.op != "clear_line")
        del_spans = [(e.start, e.end) for e in del_ops]

        # Build rendered line & index map after existing approved edits
        r_chars: list[str] = []
        r_to_orig: list[int] = []  # for each char in rendered, its 0-based start index in `orig` (-1 if inserted)
        cursor = 0
        for e in sorted((x for x in ops if x.status == "approved" and x.op != "clear_line"), key=lambda x: (x.start, x.end)):
            for k in range(cursor, e.start):
                r_chars.append(orig[k])
                r_to_orig.append(k)
            for ch_a in e.after:
                r_chars.append(ch_a)
                r_to_orig.append(-1)
            cursor = e.end
        for k in range(cursor, len(orig)):
            r_chars.append(orig[k])
            r_to_orig.append(k)
        rendered = "".join(r_chars)
        if not rendered:
            continue

        line_claimed = list(approved_spans)
        # 1. Newly emptied / stub brackets whose span contained or touched an approved del edit
        for m in empty_b_re.finditer(rendered):
            r_st, r_en = m.start(), m.end()
            orig_indices = [r_to_orig[k] for k in range(r_st, r_en) if r_to_orig[k] >= 0]
            if len(orig_indices) < 2:
                continue
            o_lo, o_hi = orig_indices[0], orig_indices[-1] + 1
            touched_by_del = any(
                (o_lo <= d_st and d_en <= o_hi) or d_en == o_lo or d_st == o_hi
                for d_st, d_en in del_spans
            )
            if not touched_by_del:
                continue
            # Emit individual del ops over surviving original sub-spans inside [o_lo, o_hi)
            k = o_lo
            while k < o_hi:
                if any(st <= k < en for st, en in line_claimed):
                    k += 1
                    continue
                k2 = k + 1
                while k2 < o_hi and not any(st <= k2 < en for st, en in line_claimed):
                    k2 += 1
                extra_ops.append(
                    EditOp(
                        line=ln,
                        start=k,
                        end=k2,
                        op="del",
                        before=orig[k:k2],
                        after="",
                        tier="rule_sym",
                        reason="post_edit_empty_parens",
                        source="rule:post_cleanup",
                        status="approved",
                    )
                )
                line_claimed.append((k, k2))
                k = k2

        # 2. Stray horizontal spaces between CJK (or before sentence punct) created directly at a del boundary
        prefix_end = _heading_protect_end(orig)
        for m in stray_sp_re.finditer(rendered):
            r_st, r_en = m.start(), m.end()
            if r_st == 0:
                continue
            prev_c = rendered[r_st - 1]
            next_c = rendered[r_en] if r_en < len(rendered) else ""
            if not ((_is_cjk(prev_c) and (_is_cjk(next_c) or next_c in "。！？，；：”」】")) or (prev_c in "“「【" and _is_cjk(next_c))):
                continue
            orig_indices = [r_to_orig[k] for k in range(r_st, r_en) if r_to_orig[k] >= 0]
            if not orig_indices:
                continue
            o_lo, o_hi = orig_indices[0], orig_indices[-1] + 1
            if o_lo < prefix_end:
                continue
            # Preserve 1 space right after paragraph lead or chapter prefix
            if not any(d_en == o_lo or d_st == o_hi or (o_lo <= d_st and d_en <= o_hi) for d_st, d_en in del_spans):
                continue
            if any(not (o_hi <= st or o_lo >= en) for st, en in line_claimed):
                continue
            extra_ops.append(
                EditOp(
                    line=ln,
                    start=o_lo,
                    end=o_hi,
                    op="del",
                    before=orig[o_lo:o_hi],
                    after="",
                    tier="rule_sym",
                    reason="post_edit_stray_space",
                    source="rule:post_cleanup",
                    status="approved",
                )
            )
            line_claimed.append((o_lo, o_hi))

        def _emit_unclaimed_subspans(orig_indices: list[int], reason: str) -> None:
            unclaimed = [idx for idx in orig_indices if not any(st <= idx < en for st, en in line_claimed)]
            if not unclaimed:
                return
            sub_lo = sub_hi = unclaimed[0]
            runs: list[tuple[int, int]] = []
            for idx in unclaimed[1:]:
                if idx == sub_hi + 1:
                    sub_hi = idx
                else:
                    runs.append((sub_lo, sub_hi + 1))
                    sub_lo = sub_hi = idx
            runs.append((sub_lo, sub_hi + 1))
            for s_lo, s_hi in runs:
                extra_ops.append(
                    EditOp(
                        line=ln, start=s_lo, end=s_hi, op="del", before=orig[s_lo:s_hi], after="",
                        tier="rule_sym", reason=reason, source="rule:post_cleanup", status="approved",
                    )
                )
                line_claimed.append((s_lo, s_hi))

        # 3. Stranded glue punctuation (`~`, `-`, `,`, non-CJK-CJK `·`) directly touching or spanning an approved del edit
        _stranded_glue_re = re.compile(r"[~,\-·]+")
        for m in _stranded_glue_re.finditer(rendered):
            r_st, r_en = m.start(), m.end()
            tok = m.group()
            prev_c = rendered[r_st - 1] if r_st > 0 else ""
            next_c = rendered[r_en] if r_en < len(rendered) else ""
            # Never touch `·` between two CJK chars (character names like 艾因诺·芙兰达)
            if tok == "·" and _is_cjk(prev_c) and _is_cjk(next_c):
                continue
            # Only strip when adjacent to CJK or sentence boundary punctuation (never inside English/numbers like K-12)
            if (prev_c.isascii() and prev_c.isalnum()) or (next_c.isascii() and next_c.isalnum()):
                continue
            if not (prev_c in "。！？，；：”」】" or _is_cjk(prev_c) or r_st == 0):
                continue
            if not (next_c in "。！？，；：”」】“「" or _is_cjk(next_c) or r_en == len(rendered)):
                continue
            # Never delete genuine standalone comma between two CJK clauses unless part of `~` / `-` / `·` noise or after sentence punct
            if tok == "," and _is_cjk(prev_c) and _is_cjk(next_c):
                continue
            orig_indices = [r_to_orig[k] for k in range(r_st, r_en) if r_to_orig[k] >= 0]
            if not orig_indices:
                continue
            o_lo, o_hi = orig_indices[0], orig_indices[-1] + 1
            if o_lo < prefix_end:
                continue
            if not any(d_en == o_lo or d_st == o_hi or (d_st < o_hi and o_lo < d_en) for d_st, d_en in del_spans):
                continue
            _emit_unclaimed_subspans(orig_indices, "post_edit_stranded_glue_punct")

        # 4. Unbalanced stray single bracket (`)` / `）` / `(` / `（`) touching an approved del edit
        for b_open, b_close in (("(", ")"), ("（", "）")):
            c_open, c_close = rendered.count(b_open), rendered.count(b_close)
            if c_open == c_close:
                continue
            target_b = b_close if c_close > c_open else b_open
            for r_i, r_ch in enumerate(rendered):
                if r_ch != target_b:
                    continue
                o_idx = r_to_orig[r_i]
                if o_idx < prefix_end or o_idx < 0:
                    continue
                if not any(d_en == o_idx or d_st == o_idx + 1 or abs(d_en - o_idx) <= 2 or abs(d_st - (o_idx + 1)) <= 2 for d_st, d_en in del_spans):
                    continue
                if any(not (o_idx + 1 <= st or o_idx >= en) for st, en in line_claimed):
                    continue
                extra_ops.append(
                    EditOp(
                        line=ln, start=o_idx, end=o_idx + 1, op="del", before=orig[o_idx : o_idx + 1], after="",
                        tier="rule_sym", reason="post_edit_unbalanced_bracket", source="rule:post_cleanup", status="approved",
                    )
                )
                line_claimed.append((o_idx, o_idx + 1))
                break

        # 5. Stranded mixed alphanumeric / circled / single digit (`1iNq`, `差8点`, `i①`) or `越已` glued right next to a watermark del edit
        has_watermark_del = any(
            e.reason.startswith(("yueyi_", "inline_", "tier_r_")) for e in del_ops
        ) or any(e.line == ln and e.reason.startswith("post_edit_") for e in extra_ops)
        if has_watermark_del:
            rendered5_chars: list[str] = []
            r5_to_orig: list[int] = []
            for r_idx, ch in enumerate(rendered):
                o_idx = r_to_orig[r_idx]
                if o_idx >= 0 and any(st <= o_idx < en for st, en in line_claimed):
                    continue
                rendered5_chars.append(ch)
                r5_to_orig.append(o_idx)
            rendered5 = "".join(rendered5_chars)
            wm_spans = [
                (e.start, e.end)
                for e in del_ops
                if e.reason.startswith(("yueyi_", "tier_c_", "eligible_", "tier_r_", "inline_"))
            ] + [
                (e.start, e.end)
                for e in extra_ops
                if e.line == ln
            ]
            _stub_re = re.compile(r"(?<=[\u4e00-\u9fff])(?:[A-Za-z0-9\u2460-\u2473]{1,6}|越已)(?=[\u4e00-\u9fff·。！？，；：”」】])")
            for m in _stub_re.finditer(rendered5):
                tok = m.group()
                r_st, r_en = m.start(), m.end()
                if (
                    SAFE_LATIN_SPAN_RE.fullmatch(tok)
                    or tok.lower() in KEEP_ENGLISH_WORDS
                    or tok.isdigit()
                    or (tok.isupper() and 2 <= len(tok) <= 6)
                    or bool(_UPPER_ALNUM_PROSE_RE.fullmatch(tok))
                ):
                    continue
                orig_indices = [r5_to_orig[k] for k in range(r_st, r_en) if r5_to_orig[k] >= 0]
                if not orig_indices:
                    continue
                o_lo, o_hi = orig_indices[0], orig_indices[-1] + 1
                if o_lo < prefix_end:
                    continue
                # Must directly touch or span a watermark/LLM/post_edit del span
                if not any(d_en == o_lo or d_st == o_hi or (d_st < o_hi and o_lo < d_en) for d_st, d_en in wm_spans):
                    continue
                _emit_unclaimed_subspans(orig_indices, "post_edit_stranded_yueyi_stub")

    return extra_ops


def _peel_pass1_and_2(
    lines: list[str],
    skip_lines: set[int],
    existing_spans_by_line: dict[int, list[tuple[int, int]]],
    is_simplified: bool,
    proper_nouns: set[str],
) -> tuple[list[EditOp], dict[int, list[tuple[int, int]]], set[int], dict[int, list[int]]]:
    """Run deterministic Pass 1 & Pass 2 pre-peeling and record anchor offsets for block windowing."""
    edits: list[EditOp] = []
    deleted_ranges: dict[int, list[tuple[int, int]]] = {}
    dirty_lines: set[int] = set()
    anchor_positions: dict[int, list[int]] = {}

    rare_left_neighbors: dict[str, Counter[str]] = {}
    rare_right_neighbors: dict[str, Counter[str]] = {}
    for idx, line in enumerate(lines, start=1):
        if idx in skip_lines or not line.strip():
            continue
        for i, ch in enumerate(line):
            if ch in TIER_R_WATERMARK_CJK:
                ln = line[i - 1] if i > 0 else "^"
                rn = line[i + 1] if i + 1 < len(line) else "$"
                rare_left_neighbors.setdefault(ch, Counter())[ln] += 1
                rare_right_neighbors.setdefault(ch, Counter())[rn] += 1

    stable_rare_chars: set[str] = set()
    for ch, lcnt in rare_left_neighbors.items():
        if ch == "亻":
            continue
        total = sum(lcnt.values())
        if total >= 5:
            top_l = lcnt.most_common(1)[0][1]
            top_r = rare_right_neighbors[ch].most_common(1)[0][1]
            if top_l / total >= 0.65 or top_r / total >= 0.65:
                stable_rare_chars.add(ch)

    for idx, line in enumerate(lines, start=1):
        if idx in skip_lines or not line.strip():
            continue
        sh = build_shadow(line)
        ev = _line_corruption_evidence(line, sh)
        heading_prefix_end = _heading_protect_end(line)
        if heading_prefix_end > 0:
            tail_str = line[heading_prefix_end:]
            has_dirty_heading_tail = bool(
                YUEYI_NOISE_CHAR_RE.search(tail_str)
                or YUEYI_BRAND_AD_RE.search(tail_str)
                or YUEYI_DIRTY_BRAND_RE.search(tail_str)
                or any(c in CIRCLED_INLINE_SET or c in TIER_R_WATERMARK_CJK for c in tail_str)
                or ev >= 2
            )
            if not has_dirty_heading_tail:
                continue

        line_ops: list[EditOp] = []
        line_anchors: list[int] = []

        # 0a. Tail group soup
        tail_span = _strip_end_group_soup_span(line) if heading_prefix_end == 0 else None
        if tail_span is not None:
            st, en = tail_span
            line_ops.append(
                EditOp(
                    line=idx,
                    start=st,
                    end=en,
                    op="del",
                    before=line[st:en],
                    after="",
                    tier="ad",
                    reason="tail_group_soup",
                    source="rule:inline_rules",
                    status="approved",
                )
            )

        # 0b. Word-internal ^ / ` between CJK
        for m in WORD_INTERNAL_CENSOR_RE.finditer(line):
            if m.start() < heading_prefix_end:
                continue
            if tail_span and m.start() >= tail_span[0]:
                continue
            line_ops.append(
                EditOp(
                    line=idx,
                    start=m.start(),
                    end=m.end(),
                    op="del",
                    before=m.group(),
                    after="",
                    tier="rule_sym",
                    reason="word_internal_censor_symbol",
                    source="rule:inline_rules",
                    status="approved",
                )
            )

        # 0c. Yueyi explicit brand ad (`-月!@漪#$首%^发&`) and invisible/anchor chars
        n_line = len(line)
        kaomoji_mask = [False] * n_line
        for km in _KAOMOJI_SPAN_RE.finditer(line):
            if any(c in _KAOMOJI_HINT_CHARS for c in km.group()):
                for k in range(km.start(), km.end()):
                    kaomoji_mask[k] = True

        for m in YUEYI_BRAND_AD_RE.finditer(line):
            if m.start() < heading_prefix_end:
                continue
            line_ops.append(
                EditOp(
                    line=idx,
                    start=m.start(),
                    end=m.end(),
                    op="del",
                    before=m.group(),
                    after="",
                    tier="ad",
                    reason="yueyi_brand_ad",
                    source="rule:inline_pass1",
                    status="approved",
                )
            )
            line_anchors.append(m.start())

        for m in YUEYI_INVISIBLE_RE.finditer(line):
            if m.start() < heading_prefix_end or kaomoji_mask[m.start()]:
                continue
            line_ops.append(
                EditOp(
                    line=idx,
                    start=m.start(),
                    end=m.end(),
                    op="del",
                    before=m.group(),
                    after="",
                    tier="rule_sym",
                    reason="yueyi_invisible_char",
                    source="rule:inline_pass1",
                    status="approved",
                )
            )
            line_anchors.append(m.start())

        for m in YUEYI_HARD_ANCHOR_RE.finditer(line):
            if m.start() < heading_prefix_end or kaomoji_mask[m.start()]:
                continue
            line_ops.append(
                EditOp(
                    line=idx,
                    start=m.start(),
                    end=m.end(),
                    op="del",
                    before=m.group(),
                    after="",
                    tier="rule_sym",
                    reason="yueyi_invisible_or_cyrillic_noise",
                    source="rule:inline_pass1",
                    status="approved",
                )
            )
            line_anchors.append(m.start())

        if ev >= 2 or heading_prefix_end > 0 or line_anchors:
            if ev >= 2 or line_anchors:
                dirty_lines.add(idx)

            is_trad_line = _is_traditional_line(line, sh)
            prot_mask = [False] * len(line)
            for k in range(min(heading_prefix_end, len(line))):
                prot_mask[k] = True
            tier_r_lex_mask = _lexicon_tier_r_mask(line)
            for pn in proper_nouns:
                start_search = 0
                while True:
                    pos = sh.text.find(pn, start_search)
                    if pos < 0:
                        break
                    o_st, o_en = sh.to_orig_span(pos, pos + len(pn))
                    for k in range(o_st, o_en):
                        prot_mask[k] = True
                    start_search = pos + 1

            safe_lat_mask = [False] * len(line)
            for sm in SAFE_LATIN_SPAN_RE.finditer(line):
                for k in range(sm.start(), sm.end()):
                    safe_lat_mask[k] = True

            # 0d. Yueyi hard-anchor cluster expansion (fullwidth digits + mixed ASCII/symbol glue)
            cluster_ops, cluster_anchors = _peel_yueyi_clusters(
                line,
                idx,
                ev,
                heading_prefix_end,
                tail_span[0] if tail_span else len(line),
                is_simplified,
                stable_rare_chars,
                prot_mask,
                tier_r_lex_mask,
                is_trad_line,
            )
            line_ops.extend(cluster_ops)
            line_anchors.extend(cluster_anchors)

            n = len(line)
            i = heading_prefix_end
            unbalanced_ascii_parens = line.count("(") != line.count(")")
            peeled_open_parens = 0
            peeled_close_parens = 0
            cjk_digits = set("零〇一二三四五六七八九十百千")
            kaomoji_mask = [False] * n
            for km in _KAOMOJI_SPAN_RE.finditer(line):
                if any(c in _KAOMOJI_HINT_CHARS for c in km.group()):
                    for k in range(km.start(), km.end()):
                        kaomoji_mask[k] = True
            while i < n:
                if tail_span and i >= tail_span[0]:
                    break
                if ev < 2 and not line_anchors:
                    break
                ch = line[i]
                prev_ch = line[i - 1] if i > 0 else ""
                next_ch = line[i + 1] if i + 1 < n else ""

                if ch in ("^", "`") and _is_cjk(prev_ch) and _is_cjk(next_ch):
                    i += 1
                    continue

                if (
                    ch == "〇"
                    and prev_ch != "〇"
                    and next_ch != "〇"
                    and not (prev_ch in cjk_digits and next_ch in cjk_digits)
                ):
                    line_ops.append(
                        EditOp(
                            line=idx,
                            start=i,
                            end=i + 1,
                            op="del",
                            before=ch,
                            after="",
                            tier="rule_sym",
                            reason="inline_isolated_circle",
                            source="rule:inline_pass1",
                            status="approved",
                        )
                    )
                    line_anchors.append(i)
                    i += 1
                    continue

                if (ch in HARD_INLINE_SYM_SET or (ch in ("⊙", "∞") and not kaomoji_mask[i] and ev >= 2)) and not (ch == "%" and prev_ch.isdigit()):
                    line_ops.append(
                        EditOp(
                            line=idx,
                            start=i,
                            end=i + 1,
                            op="del",
                            before=ch,
                            after="",
                            tier="rule_sym",
                            reason="inline_hard_symbol",
                            source="rule:inline_pass1",
                            status="approved",
                        )
                    )
                    line_anchors.append(i)
                    i += 1
                    continue

                if ch in CIRCLED_INLINE_SET and (
                    _is_cjk(prev_ch)
                    or _is_cjk(next_ch)
                    or prev_ch in ",，.(（《!！?？:：;；\"“”"
                    or next_ch in ",，.()（）》!！?？:：;；\"“”"
                ):
                    line_ops.append(
                        EditOp(
                            line=idx,
                            start=i,
                            end=i + 1,
                            op="del",
                            before=ch,
                            after="",
                            tier="rule_sym",
                            reason="inline_circled_numeral",
                            source="rule:inline_pass1",
                            status="approved",
                        )
                    )
                    line_anchors.append(i)
                    i += 1
                    continue

                if ch == ";" and _is_cjk(prev_ch) and _is_cjk(next_ch):
                    line_ops.append(
                        EditOp(
                            line=idx,
                            start=i,
                            end=i + 1,
                            op="del",
                            before=ch,
                            after="",
                            tier="rule_sym",
                            reason="inline_semicolon_between_cjk",
                            source="rule:inline_pass1",
                            status="approved",
                        )
                    )
                    line_anchors.append(i)
                    i += 1
                    continue

                if ch in "()[]" and (
                    (_is_cjk(prev_ch) and _is_cjk(next_ch))
                    or (ch == "(" and next_ch in ",，:\"“" and _is_cjk(prev_ch))
                    or (ch == "(" and prev_ch in CIRCLED_INLINE_SET and _is_cjk(next_ch))
                    or (ch == ")" and (prev_ch in ",，." or _is_cjk(prev_ch)) and _is_cjk(next_ch))
                    or (ch == "(" and next_ch == ":" and _is_cjk(prev_ch))
                    or (
                        (unbalanced_ascii_parens or peeled_open_parens > peeled_close_parens)
                        and (
                            (ch == "(" and prev_ch in "!！?？,，" and (_is_cjk(next_ch) or next_ch == "\""))
                            or (ch == ")" and _is_cjk(prev_ch) and next_ch in ",，.。!！?？”")
                        )
                    )
                ):
                    if ch == "(":
                        peeled_open_parens += 1
                    elif ch == ")":
                        peeled_close_parens += 1
                    line_ops.append(
                        EditOp(
                            line=idx,
                            start=i,
                            end=i + 1,
                            op="del",
                            before=ch,
                            after="",
                            tier="rule_sym",
                            reason="inline_ascii_paren_noise",
                            source="rule:inline_pass1",
                            status="approved",
                        )
                    )
                    line_anchors.append(i)
                    i += 1
                    continue

                # Stray Latin letters / short lowercase tokens (ls, ai, nn, yu, qi, etc.) in dirty lines
                if ch.isascii() and ch.isalpha() and not safe_lat_mask[i]:
                    j = i
                    while j < n and line[j].isascii() and line[j].isalpha():
                        j += 1
                    tok = line[i:j]
                    after_tok = line[j] if j < n else ""
                    is_kept_word = tok.lower() in KEEP_ENGLISH_WORDS and not (
                        tok == "ai" and prev_ch in ":：" and after_tok in "“\""
                    )
                    if (
                        not is_kept_word
                        and (_is_cjk(prev_ch) or prev_ch in "\",，:：*/" or prev_ch in CIRCLED_INLINE_SET)
                        and (_is_cjk(after_tok) or after_tok in "\",，:：“”*/" or after_tok in CIRCLED_INLINE_SET)
                        and (
                            len(tok) == 1
                            or LATIN_WATERMARK_PINYIN_RE.match(tok)
                            or (len(tok) == 2 and tok.islower())
                        )
                        and not (tok.isupper() and len(tok) >= 2)
                        and not (len(tok) == 1 and after_tok in "级血型超形")
                    ):
                        line_ops.append(
                            EditOp(
                                line=idx,
                                start=i,
                                end=j,
                                op="del",
                                before=tok,
                                after="",
                                tier="rule_sym",
                                reason="inline_stray_latin",
                                source="rule:inline_pass1",
                                status="approved",
                            )
                        )
                        line_anchors.append(i)
                    i = j
                    continue

                if (
                    is_simplified
                    and ch in TIER_R_WATERMARK_CJK
                    and ch not in stable_rare_chars
                    and not prot_mask[i]
                    and not (ch in _SIMPLIFIED_LEXICON_TIER_R and tier_r_lex_mask[i])
                    and not (is_trad_line and ch in _TRAD_VARIANT_TIER_R)
                ):
                    line_ops.append(
                        EditOp(
                            line=idx,
                            start=i,
                            end=i + 1,
                            op="del",
                            before=ch,
                            after="",
                            tier="rule_tier_r",
                            reason="tier_r_watermark_cjk",
                            source="rule:inline_pass2",
                            status="approved",
                        )
                    )
                    line_anchors.append(i)
                i += 1

        claimed = list(existing_spans_by_line.get(idx, []))
        if line_ops:
            line_ops.sort(key=lambda e: (e.start, -(e.end - e.start)))
            accepted: list[EditOp] = []
            for e in line_ops:
                if any(not (e.end <= st or e.start >= en) for st, en in claimed):
                    continue
                accepted.append(e)
                claimed.append((e.start, e.end))
            if accepted:
                edits.extend(accepted)
        if claimed:
            deleted_ranges[idx] = sorted(claimed)
        if line_anchors:
            anchor_positions[idx] = sorted(set(line_anchors))

    return edits, deleted_ranges, dirty_lines, anchor_positions


@dataclass
class InlineCandidateSite:
    cid: int
    orig_start: int
    orig_end: int
    char: str
    tier: str  # "N" | "C"
    reason: str


@dataclass
class DirtyLineTask:
    line_num: int
    orig_line: str
    pre_peeled_line: str
    peeled_to_orig_offsets: list[int]
    tagged_text: str
    candidates: list[InlineCandidateSite] = field(default_factory=list)


@dataclass
class InlineAnalysisResult:
    pre_edits: list[EditOp]
    proper_nouns: list[str]
    eligible_cjk: set[str]
    tasks: list[DirtyLineTask]
    dirty_line_nums: set[int]


def _render_peeled_with_map(orig_line: str, ranges: list[tuple[int, int]]) -> tuple[str, list[int]]:
    if not ranges:
        return orig_line, list(range(len(orig_line)))
    skip: set[int] = set()
    for st, en in ranges:
        skip.update(range(st, en))
    chars: list[str] = []
    omap: list[int] = []
    for idx, ch in enumerate(orig_line):
        if idx not in skip:
            chars.append(ch)
            omap.append(idx)
    return "".join(chars), omap


def _compute_book_eligible_cjk(
    lines: list[str],
    skip_lines: set[int],
    dirty_lines: set[int],
    soup_lines: set[int],
) -> set[str]:
    cnt_a: Counter[str] = Counter()
    cnt_b: Counter[str] = Counter()
    tot_a = tot_b = 0

    for idx, line in enumerate(lines, start=1):
        if idx in skip_lines or not line.strip():
            continue
        cjk = [c for c in line if _is_cjk(c)]
        if not cjk:
            continue
        if idx in dirty_lines or idx in soup_lines:
            cnt_b.update(cjk)
            tot_b += len(cjk)
        else:
            cnt_a.update(cjk)
            tot_a += len(cjk)

    eligible: set[str] = set()
    for idx in soup_lines:
        if 1 <= idx <= len(lines):
            for ch in lines[idx - 1]:
                if _is_cjk(ch):
                    d, _ = char_digit_info(build_shadow(ch).text)
                    if d is not None or ch in set("群裙峮裠箘灵梦猛撩聊"):
                        eligible.add(ch)

    if tot_b > 0 and tot_a > 0:
        min_dirty = max(1, len(dirty_lines))
        high_enr_min_fb = max(6, min_dirty // 80)
        freq_enr_min_fb = max(18, min_dirty // 50)
        for ch, fb in cnt_b.items():
            if fb < 3:
                continue
            pb = (fb + 0.5) / (tot_b + 10.0)
            pa = (cnt_a.get(ch, 0) + 0.5) / (tot_a + 10.0)
            enrichment = pb / pa
            d, _ = char_digit_info(build_shadow(ch).text)
            is_hom = d is not None or ch in set("群裙峮裠箘灵梦猛撩聊")
            if is_hom:
                if enrichment >= 1.6:
                    eligible.add(ch)
            elif (fb >= high_enr_min_fb and enrichment >= 6.0) or (fb >= freq_enr_min_fb and enrichment >= 3.0):
                eligible.add(ch)

    if len(dirty_lines) >= 3:
        core_watermarks = set(
            "零灵玲龄澪陵磷凌泠淋霖一壹医伊衣仪异艺易亦亿二贰尔"
            "三叁伞散删衫杉珊四肆思司斯私似死师五伍午舞武无吴吾呜捂物务"
            "六陆留刘柳琉溜遛榴硫七柒齐气起器旗奇企妻弃漆祁八捌巴把吧扒芭罢坝疤"
            "九玖久酒究韭揪鸠咎就群裙峮裠灵梦猛撩聊掺"
        )
        eligible.update(core_watermarks)

    return eligible


def _soft_protected_indices(peeled_line: str, proper_nouns: set[str]) -> tuple[set[int], set[int]]:
    hard: set[int] = set()
    soft: set[int] = set()

    sh_peeled = build_shadow(peeled_line)
    for pn in proper_nouns:
        pos = 0
        while True:
            hit = sh_peeled.text.find(pn, pos)
            if hit < 0:
                break
            for k in range(hit, hit + len(pn)):
                hard.add(k)
            pos = hit + 1

    for m in CJK_RUN_RE.finditer(peeled_line):
        run_start = m.start()
        run = m.group()
        for idx_in_run in range(len(run)):
            nm = LEGIT_NUMERAL_UNIT_RE.match(run[idx_in_run:])
            if nm:
                for k in range(run_start + idx_in_run, run_start + idx_in_run + nm.end()):
                    soft.add(k)
        cursor = 0
        for tok in jieba.cut(run):
            tlen = len(tok)
            if tlen >= 2 and (tok in PROTECT_WORDS or jieba.get_FREQ(tok, 0) > 0):
                for k in range(run_start + cursor, run_start + cursor + tlen):
                    soft.add(k)
            cursor += tlen
        for pw in PROTECT_WORDS:
            p_pos = 0
            while True:
                hit = run.find(pw, p_pos)
                if hit < 0:
                    break
                for k in range(run_start + hit, run_start + hit + len(pw)):
                    soft.add(k)
                p_pos = hit + 1

    return hard, soft


def _window_cjk_indices(
    orig_line: str,
    peeled_line: str,
    omap: list[int],
    anchors: list[int],
    window_radius: int = 3,
) -> set[int]:
    """Return peeled indices tightly inside Yueyi contiguous-block anchor clusters only.

    Never expands on regular scatter-poisoned books, avoiding tagging ordinary prose.
    """
    has_yueyi_specific = bool(
        YUEYI_NOISE_CHAR_RE.search(orig_line)
        or YUEYI_DIRTY_BRAND_RE.search(orig_line)
        or any(ord(c) >= 0x2474 and c in CIRCLED_INLINE_SET for c in orig_line)
        or re.search(r"月椅|亻尔|月漪", orig_line)
    )
    if not has_yueyi_specific:
        return set()

    extra_anchors = list(anchors)
    for pat in ("月椅", "亻尔", "月漪", "栎怡", "悦怡", "玥漪", "越漪"):
        pos = 0
        while True:
            hit = orig_line.find(pat, pos)
            if hit < 0:
                break
            extra_anchors.append(hit)
            extra_anchors.append(hit + len(pat) - 1)
            pos = hit + 1
    for m in YUEYI_NOISE_CHAR_RE.finditer(orig_line):
        extra_anchors.append(m.start())
    for i_ch, ch in enumerate(orig_line):
        if ord(ch) >= 0x2474 and ch in CIRCLED_INLINE_SET:
            extra_anchors.append(i_ch)
    extra_anchors = sorted(set(extra_anchors))
    if not extra_anchors:
        return set()

    spans: list[tuple[int, int]] = []
    for a in extra_anchors:
        spans.append((a - window_radius, a + window_radius))
    for i in range(len(extra_anchors) - 1):
        a, b = extra_anchors[i], extra_anchors[i + 1]
        if b - a <= 15:
            spans.append((a, b))

    window_p: set[int] = set()
    for p_idx, o_idx in enumerate(omap):
        if _is_cjk(peeled_line[p_idx]) and any(lo <= o_idx <= hi for lo, hi in spans):
            window_p.add(p_idx)
    return window_p


def analyze_inline_pollution(
    lines: list[str],
    soup_edits: list[EditOp],
    soup_line_nums: set[int],
) -> InlineAnalysisResult:
    """Run Pass 1+2 deterministic pre-peeling, self-learning, and candidate tagging for Pass 3/4."""
    skip_lines = {e.line for e in soup_edits if e.op == "clear_line"}
    existing_spans_by_line: dict[int, list[tuple[int, int]]] = {}
    for e in soup_edits:
        if e.op != "clear_line":
            existing_spans_by_line.setdefault(e.line, []).append((e.start, e.end))
    is_simp = detect_simplified_majority(lines)

    # Fast pre-filter before shadow building: skip shadow for obviously plain clean lines when scanning huge books
    _fast_dirty_hint_re = re.compile(
        r"[\^`{}\[\]|\\<>$#@&%〃_⊙∞∆∅⍳\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad\u0337"
        r"\u0100-\u02af\u0370-\u03ff\u0488\u0489\u0500-\u05ff\u10a0-\u10ff\u1200-\u16ff\u19d0-\u19df"
        r"\u1d00-\u1fff\u2113\u212f\u2160-\u2188\u2205\u2206\u2373\u3021-\u302f\u3036\u303b\ua720-\uabbf"
        r"\U00010280-\U0001034f\U0001d400-\U0001d7ff①-⓿➀-➓]|"
        r"(?<![\u0400-\u04ff])[\u0400-\u04ff]{1,2}(?![\u0400-\u04ff])|"
        r"(?<=[\u4e00-\u9fff])[A-Za-z;()](?=[\u4e00-\u9fff])|月椅|亻尔|月漪|栎怡|悦怡|玥漪|越漪"
    )
    initial_clean: list[str] = []
    initial_dirty: set[int] = set()
    for idx, line in enumerate(lines, start=1):
        if idx in skip_lines or not line.strip():
            continue
        if not _fast_dirty_hint_re.search(line) and not any(c in TIER_R_WATERMARK_CJK for c in line):
            initial_clean.append(line)
            continue
        sh = build_shadow(line)
        if _line_corruption_evidence(line, sh) >= 2:
            initial_dirty.add(idx)
        else:
            initial_clean.append(line)

    proper_nouns = set(_mine_proper_nouns(initial_clean))

    pre_edits, deleted_ranges, dirty_lines, anchor_positions = _peel_pass1_and_2(
        lines, skip_lines, existing_spans_by_line, is_simp, proper_nouns
    )
    dirty_lines |= initial_dirty

    post_peel_all: list[str] = []
    for idx, line in enumerate(lines, start=1):
        if idx in skip_lines:
            post_peel_all.append("")
            continue
        if idx in deleted_ranges:
            peeled, _ = _render_peeled_with_map(line, deleted_ranges[idx])
            post_peel_all.append(peeled)
        else:
            post_peel_all.append(line)
    if dirty_lines:
        proper_nouns.update(_mine_proper_nouns([post_peel_all[i - 1] for i in dirty_lines if post_peel_all[i - 1]]))

    eligible_cjk = _compute_book_eligible_cjk(post_peel_all, skip_lines, dirty_lines, soup_line_nums)

    tasks: list[DirtyLineTask] = []
    for idx, orig_line in enumerate(lines, start=1):
        if idx in skip_lines or idx not in dirty_lines:
            continue
        heading_prefix_end = _heading_protect_end(orig_line)
        if is_chapter_heading(orig_line.strip()) and heading_prefix_end == 0:
            continue
        if heading_prefix_end > 0 and idx not in deleted_ranges and not YUEYI_NOISE_CHAR_RE.search(orig_line[heading_prefix_end:]):
            continue
        peeled, omap = _render_peeled_with_map(orig_line, deleted_ranges.get(idx, []))
        win_indices = _window_cjk_indices(orig_line, peeled, omap, anchor_positions.get(idx, []))
        cjk_chars = [c for c in peeled if _is_cjk(c)]
        if len(cjk_chars) < 4:
            continue
        elig_cnt = sum(1 for c in cjk_chars if c in eligible_cjk)
        if elig_cnt < 2 and len(win_indices) < 2:
            continue

        hard, soft = _soft_protected_indices(peeled, proper_nouns)
        if heading_prefix_end > 0:
            for p_idx, o_idx in enumerate(omap):
                if o_idx < heading_prefix_end:
                    hard.add(p_idx)
        candidates: list[InlineCandidateSite] = []
        tagged_parts: list[str] = []
        cid = 0
        for p_idx, ch in enumerate(peeled):
            is_cand = (
                _is_cjk(ch)
                and p_idx not in hard
                and (ch in eligible_cjk or p_idx in win_indices)
            )
            if is_cand:
                cid += 1
                orig_idx = omap[p_idx]
                if p_idx in soft or ch in ULTRA_COMMON_PROSE_CJK:
                    tier = "C"
                    reason = (
                        "soft_protected_lexicon" if p_idx in soft else "ultra_common_prose_cjk"
                    )
                elif ch in eligible_cjk:
                    tier = "N"
                    reason = "eligible_watermark_homophone"
                else:
                    tier = "C"  # Non-enriched window char stays Tier-C so cross-check protects prose!
                    reason = "anchor_window_candidate"
                candidates.append(
                    InlineCandidateSite(
                        cid=cid,
                        orig_start=orig_idx,
                        orig_end=orig_idx + 1,
                        char=orig_line[orig_idx],
                        tier=tier,
                        reason=reason,
                    )
                )
                tagged_parts.append(f"⟦{cid}{ch}⟧")
            else:
                tagged_parts.append(ch)

        if candidates:
            tasks.append(
                DirtyLineTask(
                    line_num=idx,
                    orig_line=orig_line,
                    pre_peeled_line=peeled,
                    peeled_to_orig_offsets=omap,
                    tagged_text="".join(tagged_parts),
                    candidates=candidates,
                )
            )

    return InlineAnalysisResult(
        pre_edits=pre_edits,
        proper_nouns=sorted(proper_nouns),
        eligible_cjk=eligible_cjk,
        tasks=tasks,
        dirty_line_nums=dirty_lines,
    )
