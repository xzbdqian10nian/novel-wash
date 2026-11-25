"""Layer 3b: Censored placeholder (`**` / `〇〇`) slot restoration.

Provides deterministic dictionary + strict slot-only invariance restoration as Python `EditOp`s.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from .edit_log import EditOp
from .llm_client import RouterConfigError

PLACE_RE = re.compile(r"\*{2,}|[〇○]{2}")
CJK_ONE_RE = re.compile(r"[\u3400-\u9fff]")

_P = r"(?:\*{2,}|[〇○]{2})"

KNOWN_RESTORE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"突{_P}况"), "突发状况"),
    (re.compile(rf"感{_P}彩"), "感情色彩"),
    (re.compile(rf"双{_P}错"), "双剑交错"),
    (re.compile(rf"男{_P}望"), "男性欲望"),
    (re.compile(rf"水{_P}融"), "水乳交融"),
    (re.compile(rf"长{_P}女"), "长幼男女"),
    (re.compile(rf"纵{_P}欲"), "纵情纵欲"),
    (re.compile(rf"奸夫{_P}"), "奸夫淫妇"),
    (re.compile(rf"情情{_P}"), "情情爱爱"),
    (re.compile(rf"雪{_P}腻"), "雪白细腻"),
    (re.compile(rf"雪{_P}肉"), "雪白嫩肉"),
    (re.compile(rf"雪{_P}团"), "雪乳团"),
    (re.compile(rf"纯{_P}净"), "纯洁干净"),
    (re.compile(rf"畅{_P}"), "畅快感"),
    (re.compile(rf"爽{_P}"), "爽快感"),
    (re.compile(rf"锦{_P}燕"), "锦玉飞燕"),
    (re.compile(rf"秋{_P}近"), "秋季临近"),
    (re.compile(rf"所想{_P}行"), "所想即行"),
    (re.compile(rf"摸了{_P}口"), "摸了摸胸口"),
    (re.compile(rf"随{_P}谈"), "随意畅谈"),
    (re.compile(rf"来一{_P}感"), "来一场情感"),
    (re.compile(rf"双{_P}叠"), "双脚交叠"),
    (re.compile(rf"气吞{_P}"), "气吞山河"),
    (re.compile(rf"昏{_P}、"), "昏睡药、"),
    (re.compile(rf"{_P}之欢"), "鱼水之欢"),
    (re.compile(rf"{_P}大发"), "兽性大发"),
    (re.compile(rf"被{_P}绑起来"), "被五花大绑起来"),
    (re.compile(rf"那{_P}绑"), "那绳索绑"),
    (re.compile(rf"聊以{_P}"), "聊以自慰"),
    (re.compile(rf"嫩{_P}滑"), "嫩白细滑"),
    (re.compile(rf"雪{_P}嫩"), "雪白粉嫩"),
]

BANNED_FILLS = {"情况", "关系", "咬", "那个", "这样", "那样", "什么", "这个"}


def has_placeholder(s: str) -> bool:
    return bool(PLACE_RE.search(s))


def leftover_count(s: str) -> int:
    return len(PLACE_RE.findall(s))


def is_fill_word(w: str) -> bool:
    return bool(re.fullmatch(r"[\u3400-\u9fff]{1,4}", w or ""))


def is_book_title(line: str) -> bool:
    return bool(re.search(r"《[^》]*[*〇○]{2}[^》]*》", line))


def is_uncertain_proper(line: str) -> bool:
    if re.search(r"[颗枚株门本座]\s*[*〇○]{2}", line):
        return True
    if "毒" in line and re.search(r"[乃则若]是\s*[*〇○]{2}", line):
        return True
    return False


def is_false_positive(line: str) -> bool:
    if not has_placeholder(line):
        return True
    if re.search(r"特[〇○]{2}", line):
        return True
    if re.search(r"[的把了]\s*[〇○]{2}", line) and not re.search(r"\*{2,}", line):
        return True
    if re.search(r"[xX]{2,}", line) and not re.search(r"\*{2,}", line):
        return True
    if re.search(r"口口相传|一口口|口水", line) and not re.search(r"\*{2,}", line):
        return True
    if re.fullmatch(r"[+\-_=.\s]*\*{2,}[+\-_=.\s]*", line.strip()):
        return True
    return False


def classify_line(line: str) -> str:
    if not has_placeholder(line):
        return "none"
    if is_false_positive(line):
        return "false_positive"
    if is_book_title(line):
        return "book_title"
    if leftover_count(line) >= 3:
        return "multi_slot"
    if is_uncertain_proper(line):
        return "proper"
    return "fill"


def _is_stacked_fill(word: str, left: str, right: str) -> bool:
    last_l = left[-1:]
    first_r = right[:1]
    if last_l and CJK_ONE_RE.match(last_l) and word[0] == last_l:
        return True
    if first_r and CJK_ONE_RE.match(first_r) and word[-1] == first_r:
        return True
    return False


def _is_banned_fill(word: str, left: str) -> bool:
    if word in BANNED_FILLS:
        return True
    if word == "东西" and not left.endswith("的"):
        return True
    return False


def fills_ok(prev: str, words: list[str]) -> bool:
    parts = PLACE_RE.split(prev)
    slots = len(parts) - 1
    if len(words) < slots:
        return False
    for i in range(slots):
        w = words[i]
        if not is_fill_word(w):
            return False
        if _is_stacked_fill(w, parts[i], parts[i + 1]):
            return False
        if _is_banned_fill(w, parts[i]):
            return False
    return True


def apply_fills(prev: str, fills: list[str]) -> str | None:
    """Slot-only stitching: guarantees characters outside `**` / `〇〇` are untouched."""
    slots = leftover_count(prev)
    if not slots or len(fills) < slots:
        return None
    words = fills[:slots]
    if not fills_ok(prev, words):
        return None
    idx = 0

    def _repl(_m: re.Match[str]) -> str:
        nonlocal idx
        w = words[idx]
        idx += 1
        return w

    nxt = PLACE_RE.sub(_repl, prev)
    return nxt if leftover_count(nxt) < leftover_count(prev) else None


def accept_restore(prev: str, nxt: str) -> str | None:
    """Verify a full-line restore string matches `prev` everywhere outside slot spans."""
    t = nxt.replace("\r", "").rstrip()
    if not t or t == prev:
        return None
    parts = PLACE_RE.split(prev)
    if len(parts) < 2:
        return None
    extracted: list[str] = []
    rest = t
    for i, p in enumerate(parts):
        if i == 0:
            if not rest.startswith(p):
                return None
            rest = rest[len(p) :]
            continue
        if len(p) == 0:
            if not is_fill_word(rest):
                return None
            extracted.append(rest)
            rest = ""
            continue
        idx = rest.find(p)
        if idx < 1 or idx > 4:
            return None
        filled = rest[:idx]
        if not is_fill_word(filled):
            return None
        extracted.append(filled)
        rest = rest[idx + len(p) :]
    if rest != "":
        return None
    if leftover_count(t) >= leftover_count(prev):
        return None
    if not fills_ok(prev, extracted):
        return None
    return t


def restore_known_slots(
    lines: list[str],
    skip_lines: set[int] | None = None,
    llm_fill: Callable[[str, str], str] | None = None,
    *,
    max_llm_lines: int = 120,
) -> list[EditOp]:
    """Produce slot-level EditOps (`op="sub"`, `tier="harmony"`) for `**` / `〇〇` placeholders.

    Only generates edits for unambiguous dictionary hits (plus optional batched slot-only LLM fills).
    """
    skip = skip_lines or set()
    edits: list[EditOp] = []
    llm_candidates: list[tuple[int, str, list[tuple[int, int]]]] = []

    for line_num, line in enumerate(lines, start=1):
        if line_num in skip or classify_line(line) != "fill":
            continue
        # Step 1: Deterministic known pattern matches -> emit exact slot EditOps
        covered_slots: set[int] = set()
        for pat, full_repl in KNOWN_RESTORE_PATTERNS:
            for m in pat.finditer(line):
                sub_match = PLACE_RE.search(m.group())
                if not sub_match:
                    continue
                slot_start = m.start() + sub_match.start()
                slot_end = m.start() + sub_match.end()
                prefix_len = sub_match.start()
                suffix_len = len(m.group()) - sub_match.end()
                fill_word = full_repl[prefix_len : len(full_repl) - suffix_len if suffix_len else None]
                if not is_fill_word(fill_word):
                    continue
                edits.append(
                    EditOp(
                        line=line_num,
                        start=slot_start,
                        end=slot_end,
                        op="sub",
                        before=line[slot_start:slot_end],
                        after=fill_word,
                        tier="harmony",
                        reason="known_harmony_patch",
                        source="rule:harmony_restore",
                        status="approved",
                    )
                )
                covered_slots.add(slot_start)

        # Collect remaining uncovered slots if llm_fill is provided
        if llm_fill is not None:
            rem_slots = [(m.start(), m.end()) for m in PLACE_RE.finditer(line) if m.start() not in covered_slots]
            if rem_slots and len(llm_candidates) < max_llm_lines:
                llm_candidates.append((line_num, line, rem_slots))

    # Optional batched slot-only LLM fill with book pattern caching
    if llm_fill is not None and llm_candidates:
        sys_prompt = (
            "你是中文网文被屏蔽词还原器。每行用 ⟦SLOTk⟧ 标出了被平台打码成 ** 或 〇〇 的位置。\n"
            "请结合上下文，为每个槽位输出 1 个 1~4 字最准确的简体汉字还原词。不要输出整句。\n"
            "严格输出 JSON，每行有几个 ⟦SLOT⟧ 槽位就输出几个词（键为 \"L行号\"）：{\"L行号\":[\"词1\",...]}"
        )
        pattern_cache: dict[str, str] = {}
        batch_size = 20
        for b_start in range(0, len(llm_candidates), batch_size):
            batch = llm_candidates[b_start : b_start + batch_size]
            prompt_items = []
            for ln, line, slots in batch:
                ctx_prev = lines[ln - 2].strip() if ln >= 2 else ""
                # Tag slots
                marked = []
                cur = 0
                for si, (st, en) in enumerate(slots, start=1):
                    marked.append(line[cur:st])
                    marked.append(f"⟦SLOT{si}⟧")
                    cur = en
                marked.append(line[cur:])
                prompt_items.append(f"[ctx] {ctx_prev[:60]}\n[L{ln}] {''.join(marked)}")
            try:
                raw = llm_fill(sys_prompt, "\n".join(prompt_items))
            except RouterConfigError:
                raise
            except Exception:
                continue
            s = raw.strip()
            lo, hi = s.find("{"), s.rfind("}")
            if lo >= 0 and hi > lo:
                try:
                    parsed = json.loads(s[lo : hi + 1])
                except Exception:
                    parsed = {}
            else:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}

            for ln, line, slots in batch:
                fills = parsed.get(f"L{ln}")
                if fills is None:
                    fills = parsed.get(str(ln))
                if not isinstance(fills, list) or len(fills) < len(slots):
                    continue
                for (st, en), word in zip(slots, fills):
                    w = str(word).strip()
                    left = line[max(0, st - 2) : st]
                    right = line[en : en + 2]
                    if is_fill_word(w) and not _is_stacked_fill(w, left, right) and not _is_banned_fill(w, left):
                        pattern_key = f"{left}{line[st:en]}{right}"
                        pattern_cache.setdefault(pattern_key, w)
                        edits.append(
                            EditOp(
                                line=ln,
                                start=st,
                                end=en,
                                op="sub",
                                before=line[st:en],
                                after=w,
                                tier="harmony",
                                reason="llm_harmony_slot_fill",
                                source="llm:harmony_restore",
                                status="approved",
                                confidence=0.85,
                            )
                        )

    return edits
