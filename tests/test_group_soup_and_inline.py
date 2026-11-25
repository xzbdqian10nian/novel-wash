import json
import os
from pathlib import Path

import pytest

from novel_wash.edit_log import apply_and_verify
from novel_wash.encoding import decode_bytes, read_novel_file
from novel_wash.group_soup import detect_explicit_ads_only, detect_soup_and_ads
from novel_wash.inline_rules import analyze_inline_pollution
from novel_wash.triage import triage_decoded


def test_group_soup_clears_hard_variants_and_preserves_headings_and_prose():
    lines = [
        "第10章 风波",
        "　　“我带你去。”外:裙)1:2。3·四、五'六“7:八,9,0",
        "饵衫*丝捂}⑹qi八久医",
        "壹饵衫④-捂 硫琦岜 氿|君羊",
        "%箘⑨岜企陆呜④衫?饵依",
        "234567891",
        "234567891",
        "234567891",
        "贰⑶⑷捂陸琦把玖依",
        "（1/3）",
        "*",
        "　　九百五十三，九百五十四。",
    ]
    edits, review, clusters, _ = detect_soup_and_ads(lines)
    cleared = {e.line for e in edits if e.op == "clear_line"}
    assert {3, 4, 5, 6, 7, 8, 9}.issubset(cleared)
    assert 1 not in cleared  # chapter heading
    assert 2 not in cleared  # dialogue line with tail soup
    assert 10 not in cleared  # （1/3）
    assert 11 not in cleared  # * scene divider
    assert 12 not in cleared  # 九百五十三，九百五十四。


def test_inline_rules_pass1_pass2_and_hard_protections():
    lines = [
        "第一章 青云城",
        "青云城是一座古老边城。青云城有很多巡逻队。" * 10,
        "和^谐与温`柔平`静需要清洗。",
        "所以三靈位夢守陾卫希霖望趁着巴最后⑸这段淋夜色咎,停衫留|在青云;城中V,好好I的揪查探、商议一下三十六层高塔的攻略方法,",
        "“我带你去。”外:裙)1:2。3·四、五'六“7:八,9,0",
        "木屑纷飞中，令人san值狂掉的恐怖身影出现了。",
        "剑批|拳批|掌批|刀批|枪批|药批|分霸",
        "他穿着旧T恤，开着S形走位，拿到了A级评价和维他命C，说着玩的就是XP和卡bug。",
    ]
    se, _, _, sn = detect_soup_and_ads(lines)
    res = analyze_inline_pollution(lines, se, sn)
    peeled, _ = apply_and_verify(lines, se + res.pre_edits)

    # Proper noun protection + Pass 1/2 peeling
    assert "青云城" in res.proper_nouns
    assert "和谐与温柔平静需要清洗。" == peeled[2]
    assert "所以三位守卫希霖望趁着巴最后这段淋夜色咎,停衫留在青云城中,好好的揪查探、商议一下三十六层高塔的攻略方法," == peeled[3]
    assert peeled[4] == "“我带你去。”"
    # Clean traps untouched
    assert peeled[5] == lines[5]
    assert peeled[6] == lines[6]
    assert peeled[7] == lines[7]
    assert "卫" not in res.eligible_cjk
    task_lines = {t.line_num for t in res.tasks}
    assert 6 not in task_lines and 7 not in task_lines and 8 not in task_lines


def test_yueyi_block_window_tagging_and_brand_strip():
    lines = [
        "第一章 北境学院",
        "“别急！真是的-月!@漪#$首%^发&，前辈，你过来，行囊也不带吗？”",
        "所以她很讨栮亦⑶伍霓氿６衫贰厌北境学院，因为这是规定。",
        "现在уu ꬲ漪（二）蹴泣鹨九①珊\u2060坝流舒展一下筋骨之后就好多了。",
    ]
    se, _, _, sn = detect_soup_and_ads(lines)
    res = analyze_inline_pollution(lines, se, sn)
    peeled, _ = apply_and_verify(lines, se + res.pre_edits)
    assert peeled[1] == "“别急！真是的，前辈，你过来，行囊也不带吗？”"
    assert "\u2060" not in peeled[3]
    assert "у" not in peeled[3] and "ꬲ" not in peeled[3]
    task_map = {t.line_num: t for t in res.tasks}
    assert 4 in task_map
    # Pass 1/2 already peeled Cyrillic/invisible/rare chars on line 4; window captures adjacent block chars without spilling into 筋骨/好了
    tagged_chars = {c.char for c in task_map[4].candidates}
    assert {"流", "漪"}.issubset(tagged_chars) or {"九", "珊"}.issubset(tagged_chars)
    assert "骨" not in tagged_chars and "宿" not in tagged_chars


def test_optional_local_soup_regression():
    corpus_env = os.environ.get("NOVEL_WASH_SOUP_CORPUS_FILE")
    rep_env = os.environ.get("NOVEL_WASH_SOUP_REPORT_FILE")
    if not (corpus_env and rep_env):
        pytest.skip("Optional local soup regression env vars not set")
    corpus_path = Path(corpus_env)
    rep_path = Path(rep_env)
    if not (corpus_path.exists() and rep_path.exists()):
        pytest.skip("Optional local soup corpus fixture not present")
    corpus_doc = read_novel_file(corpus_path)
    edits, _, _, _ = detect_soup_and_ads(corpus_doc.lines)
    cleared = {e.line for e in edits if e.op == "clear_line"}
    old_rr = {r["line"] for r in json.loads(rep_path.read_text())["review_rows"]}
    caught = sum(1 for ln in old_rr if ln in cleared)
    assert caught == len(old_rr)


def test_inline_pass1_peels_circle_and_radical_and_guards_tier_c():
    norm_lines = (
        ["第一章 顾言的旅程"]
        + ["顾言说：“二〇〇八年我和她去了青云城，他穿着长衫站在门外。”"] * 30
        + [
            "“顾言先<生!⑤(在〇想什!么)啊,她轻轻咳嗽了一声,能够〇在说:ai“小i⑺声,唯六一),亻尔是灵梦的”",
            "顾言在客厅亿里坐下铃来,握了妻握师她的手,这<是四六一,亻尔知道吗?",
            "顾言点<头道,这是^个问题,五九七,亻尔怎么看?",
        ]
    )
    se, _, _, sn = detect_soup_and_ads(norm_lines)
    res = analyze_inline_pollution(norm_lines, se, sn)
    peeled, _ = apply_and_verify(norm_lines, se + res.pre_edits)
    # 二〇〇八 preserved on clean lines; isolated 〇, !⑤(, :ai“, i⑺, ), 亻 all peeled on dirty lines
    assert "二〇〇八" in peeled[1]
    dirty_peeled = peeled[31]
    assert "〇" not in dirty_peeled
    assert "⑤" not in dirty_peeled
    assert ":ai“" not in dirty_peeled
    assert "i⑺" not in dirty_peeled
    assert "一)," not in dirty_peeled
    assert "亻" not in dirty_peeled
    assert "咳嗽" in dirty_peeled
    # 顾言 is mined as a proper noun; 咳 is NOT enriched into eligible_cjk; 一/六/是/尔 remain Tier-C
    assert "顾言" in res.proper_nouns
    assert "咳" not in res.eligible_cjk
    t = next(t for t in res.tasks if t.line_num == 32)
    # First 尔 (if any) stays out of proper noun 顾言; later 尔 (from 亻尔) & 一/六/是 are Tier-C
    assert all(c.orig_start > 2 for c in t.candidates if c.char == "尔")
    tier_by_char = {c.char: c.tier for c in t.candidates}
    for common in ("一", "六", "是", "尔"):
        if common in tier_by_char:
            assert tier_by_char[common] == "C"


def test_clean_book_traps_and_tier_r_lexicon_protection(tmp_path: Path):
    from novel_wash.cli import wash_book

    # 1. group_soup must never clear percentages, quoted ASCII, speaker kaomoji, or 1!
    safe_lines = [
        "第一章 日常",
        "1.12%",
        "93.3%",
        "1!",
        "“what?”",
        "林美：^-^",
        "舅舅：_#",
    ]
    se, _, _, _ = detect_soup_and_ads(safe_lines)
    assert not any(e.op == "clear_line" for e in se)

    # 2. TIER_R Pass 2 must protect 队伍 / 退伍 / 肆虐 even on dirty lines
    dirty_with_words = [
        "第一章 战斗",
        "他<们分好队伍，准备退伍，然而魔力仍然在肆虐。" * 12,
        "他<们分^好队伍，确保退伍前不要肆虐起来，五九七。",
    ]
    se2, _, _, sn2 = detect_soup_and_ads(dirty_with_words)
    ir2 = analyze_inline_pollution(dirty_with_words, se2, sn2)
    peeled2, _ = apply_and_verify(dirty_with_words, se2 + ir2.pre_edits)
    assert "队伍" in peeled2[2]
    assert "退伍" in peeled2[2]
    assert "肆虐" in peeled2[2]

    # 3. Clean-graded book in wash_book leaves normal prose/kaomoji/tags untouched (total_approved_edits == 0)
    clean_book = tmp_path / "clean_novel.txt"
    clean_body = "\n".join(
        [
            "第一章 起点",
            "1.12%",
            "今天她穿了件哆啦A梦T恤，走向X学院，A，B，C，D四列，b类股。",
            "保护区|药桶|高速法力",
            "B、(o>ε(o>u(≧∩≦)(￣ー￣)X(^▽^)",
            "“what?”",
        ]
        + ["正常小说正文段落，没有任何防盗水印或乱码，大家安心阅读。"] * 40
    )
    clean_book.write_text(clean_body + "\n", encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    res = wash_book(clean_book, project_root=project_root, out_dir=tmp_path / "out", use_model=False, render_epub=False)
    assert res["stats"]["total_approved_edits"] == 0
    assert Path(res["ai_path"]).read_text(encoding="utf-8") == clean_body + "\n"
    reader_out = Path(res["reader_path"]).read_text(encoding="utf-8")
    for needle in ("1.12%", "哆啦A梦T恤", "保护区|药桶|高速法力", "(^▽^)", "“what?”"):
        assert needle in reader_out


def test_blocker1_dialogue_prose_and_ad_regex_and_six_digit_cluster():
    lines = [
        "第一章 对话保护",
        "\u3000\u3000“啊，是你。”Iı玲爾⑵１叄玲吧侕",
        "\u3000\u3000鹨壹⑺①⑵把逝师𝞺拔 yuе漪“噗嗤。”",
        "酒溜ӏIꭒ师⑥妻⑧洱八\u3000\u3000“嗯~”",
        "\u3000２（九）七镏久уi氵疤硫\u3000好帅。",
        "\u3000\u300022%…18%……12%…10%（电量提示）…",
        "\u3000\u3000灵依、灵儿…01、02…",
        "依霖医企咝无玖іV九拔\u3000\u3000要么……",
        "来源：这是他写下的案情笔记。",
        "他轻声说：“记住网址和密码再走。”",
        "来源：https://example.com",
        "她说：“记住网址https://abc.com 就行。”",
        "群234567891",
        "群234567891",
        "群234567891",
        "“234567891”",
        "234567891……",
    ]
    se, _, _, _ = detect_soup_and_ads(lines)
    cleared = {e.line for e in se if e.op == "clear_line"}
    # Protected dialogue / ellipsis / multi-pct / prose must NEVER be cleared (not even by cluster rescue)
    for ln in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17):
        assert ln not in cleared, f"Protected line {ln} was cleared: {lines[ln - 1]!r}"
    assert 11 in cleared  # explicit URL source ad

    applied, _ = apply_and_verify(lines, se)
    assert applied[8] == "来源：这是他写下的案情笔记。"
    assert applied[9] == "他轻声说：“记住网址和密码再走。”"
    assert applied[11] == "她说：“ 就行。”"

    # Explicit clean-safe ad pass also spares prose/dialogue
    ad_only = detect_explicit_ads_only(lines)
    ad_applied, _ = apply_and_verify(lines, ad_only)
    assert ad_applied[8] == lines[8]
    assert ad_applied[9] == lines[9]
    assert ad_applied[10] == ""
    assert ad_applied[11] == "她说：“ 就行。”"

    # Standalone URL source ad clears; source prefix glued to dialogue strips only prefix while keeping prose
    mixed_src = [
        "第一章 开始",
        "来源：https://example.com。她说：“这是假的。”",
        "来源：https://example.com。她说：‘这是假的。’",
        "然后她合上笔记。",
    ]
    clean_mixed = detect_explicit_ads_only(mixed_src)
    applied_mixed, _ = apply_and_verify(mixed_src, clean_mixed)
    assert applied_mixed[1] == "她说：“这是假的。”"
    assert applied_mixed[2] == "她说：‘这是假的。’"
    full_mixed, _, _, _ = detect_soup_and_ads(mixed_src)
    applied_full_mixed, _ = apply_and_verify(mixed_src, full_mixed)
    assert applied_full_mixed[1] == "她说：“这是假的。”"
    assert applied_full_mixed[2] == "她说：‘这是假的。’"

    # Six-digit cluster rescue works for both ASCII and circled seeds
    for seq in (["群123456"] * 3 + ["123456"], ["群①②③④⑤⑥"] * 3 + ["123456"]):
        s6, _, clusters6, _ = detect_soup_and_ads(seq)
        assert "123456" in clusters6
        assert {e.line for e in s6 if e.op == "clear_line"} == {1, 2, 3, 4}


def test_foreign_cyrillic_dialogue_not_stripped():
    lines = ["他轻声说：Привет，同志。"] * 5
    tri = triage_decoded(decode_bytes("\n".join(lines).encode("utf-8")))
    assert tri.grade == "clean"
    se, _, _, sn = detect_soup_and_ads(lines)
    ir = analyze_inline_pollution(lines, se, sn)
    assert ir.pre_edits == []


def test_round4_source_ad_comma_and_space_variants_preserve_trailing_prose():
    src = [
        "来源：https://example.com，后来他走了。",
        "来源：https://example.com 后来他走了。",
        "　　来源：https://example.com 后来他走了。",
        "发布页：https://example.com，后来他走了。",
        "来源：https://example.com",
        "来源：https://example.com。",
    ]
    clean_ops = detect_explicit_ads_only(src)
    clean_out, clean_cleared = apply_and_verify(src, clean_ops)
    assert clean_out == [
        "后来他走了。",
        "后来他走了。",
        "　　后来他走了。",
        "后来他走了。",
        "",
        "",
    ]
    assert clean_cleared == {5, 6}

    full_ops, _, _, _ = detect_soup_and_ads(src)
    full_out, full_cleared = apply_and_verify(src, full_ops)
    assert full_out == [
        "后来他走了。",
        "后来他走了。",
        "　　后来他走了。",
        "后来他走了。",
        "",
        "",
    ]
    assert full_cleared == {5, 6}


def test_round5_site_brand_remember_url_and_visit_prefix_preserve_prose():
    src = [
        "搜书吧发布页：https://example.com，后来他走了。",
        "　　记住网址：https://example.com,后来他走了。",
        "来源：https://example.com 欢迎访问我们的新家。",
        "正文记住网址：https://example.com,后来他走了。",
        "搜书吧发布页：https://example.com",
    ]
    clean_ops = detect_explicit_ads_only(src)
    clean_out, clean_cleared = apply_and_verify(src, clean_ops)
    assert clean_out == [
        "后来他走了。",
        "　　后来他走了。",
        "欢迎访问我们的新家。",
        "正文,后来他走了。",
        "",
    ]
    assert clean_cleared == {5}

    full_ops, _, _, _ = detect_soup_and_ads(src)
    full_out, full_cleared = apply_and_verify(src, full_ops)
    assert full_out == [
        "后来他走了。",
        "　　后来他走了。",
        "欢迎访问我们的新家。",
        "正文,后来他走了。",
        "",
    ]
    assert full_cleared == {5}


def test_round6_lingdian_and_provenance_prose_boundaries():
    src = [
        "https://m.lingdian.com，后来他走了。",
        "本文件来自网站后来他去了论坛。",
        "本文件来自尚香书院。",
    ]
    clean_ops = detect_explicit_ads_only(src)
    clean_out, clean_cleared = apply_and_verify(src, clean_ops)
    assert clean_out == ["，后来他走了。", "本文件来自网站后来他去了论坛。", ""]
    assert clean_cleared == {3}

    full_ops, _, _, _ = detect_soup_and_ads(src)
    full_out, full_cleared = apply_and_verify(src, full_ops)
    assert full_out == ["，后来他走了。", "本文件来自网站后来他去了论坛。", ""]
    assert full_cleared == {3}


def test_round7_closed_provenance_and_lingdian_scheme_anchoring():
    # 1. Arbitrary provenance sentence must never be cleared; closed piracy site + URL provenance still clear
    prov_lines = [
        "本文件来自出版社并转发至论坛。",
        "本文件来自尚香书院。",
        "本文件来自https://example.com",
    ]
    c_ops = detect_explicit_ads_only(prov_lines)
    c_out, c_cleared = apply_and_verify(prov_lines, c_ops)
    assert c_out == ["本文件来自出版社并转发至论坛。", "", ""]
    assert c_cleared == {2, 3}

    f_ops, _, _, _ = detect_soup_and_ads(prov_lines)
    f_out, f_cleared = apply_and_verify(prov_lines, f_ops)
    assert f_out == ["本文件来自出版社并转发至论坛。", "", ""]
    assert f_cleared == {2, 3}

    # 2. Naked domain in prose only deletes m.lingdian.com while all 6 obfuscated variants strip cleanly
    lingdian_lines = [
        "他打开m.lingdian.com，后来他走了。",
        "记住网址m.lingdianｚｗw.com正文",
        "首发网址https://m.lingdian正文",
        "tt陸ps⑨://逡m.lingdian正文",
        "t飼p诌s:/虾/m.宭lingdian正文",
        "思址https://m.lingdian正文",
        "捌h群ttps://m.lingdian正文",
    ]
    expected_after = [
        "他打开，后来他走了。",
        "正文",
        "正文",
        "正文",
        "正文",
        "正文",
        "正文",
    ]
    c_ling_ops = detect_explicit_ads_only(lingdian_lines)
    c_ling_out, c_ling_clr = apply_and_verify(lingdian_lines, c_ling_ops)
    assert c_ling_out == expected_after
    assert c_ling_clr == set()

    f_ling_ops, _, _, _ = detect_soup_and_ads(lingdian_lines)
    f_ling_out, f_ling_clr = apply_and_verify(lingdian_lines, f_ling_ops)
    assert f_ling_out == expected_after
    assert f_ling_clr == set()


def test_round8_lingdian_host_and_query_boundaries():
    unrelated = [
        "他打开https://notlingdian.com，后来他走了。",
        "她写下system.lingdian.example.com，然后离开。",
        "他访问https://m.lingdian.example.com，然后离开。",
        "他访问https://m.lingdian.com.evil.org，然后离开。",
        "他访问https://example.com/?url=m.lingdian.com，然后离开。",
        "他访问https://example.com/?url=https://m.lingdian.com，然后离开。",
        "他访问https://example.com/path/m.lingdian.com，然后离开。",
    ]
    assert detect_explicit_ads_only(unrelated) == []
    full_unrel_ops, _, _, _ = detect_soup_and_ads(unrelated)
    assert full_unrel_ops == []

    target_url = ["他打开https://m.lingdian.com/book/123.html，后来他走了。"]
    c_out, c_clr = apply_and_verify(target_url, detect_explicit_ads_only(target_url))
    assert c_out == ["他打开，后来他走了。"] and c_clr == set()
    f_ops, _, _, _ = detect_soup_and_ads(target_url)
    f_out, f_clr = apply_and_verify(target_url, f_ops)
    assert f_out == ["他打开，后来他走了。"] and f_clr == set()


def test_round9_lingdian_authority_userinfo_and_quoted_query_isolation():
    nested_and_userinfo = [
        "他访问https://m.lingdian.com@docs.example.org/book，然后离开。",
        "他访问https://docs.example.org/?next='https://m.lingdian.com'，然后离开。",
        '他访问https://docs.example.org/?next="https://m.lingdian.com"，然后离开。',
        "他访问https://docs.example.org/?next='m.lingdian.com'，然后离开。",
        "他访问https://docs.example.org/#https://m.lingdian.com，然后离开。",
        "他写下m.lingdian.com@docs.example.org，然后离开。",
        "他访问https://user@m.lingdian.com/book，然后离开。",
    ]
    assert detect_explicit_ads_only(nested_and_userinfo) == []
    full_ops, _, _, _ = detect_soup_and_ads(nested_and_userinfo)
    assert full_ops == []


def test_quality_fix_proper_noun_sujiuli_trad_guard_and_ads():
    from novel_wash.inline_rules import cleanup_post_edit_artifacts

    # 1. 苏玖璃 mined into proper_nouns even though 玖 is in TIER_R, while 队伍/肆意/鸸鹋 stay protected
    lines = (
        ["第一章 相遇"]
        + ["苏玖璃站在灵气浓郁的山巅，队伍里的众人肆意大笑，旁边的鸸鹋滴熘熘地看着苏玖璃。"] * 12
        + [
            "苏玖璃^再一次被他虐得体无完肤，巴三七。",
            "處大陸北境，需要從精靈之森出發，繼續向著暴風雪前進，巴三七^。",
            "她的队伍在肆意狂风中看到了鸸鹋和滴熘熘的眼睛^，巴三七。",
        ]
    )
    se, _, _, sn = detect_soup_and_ads(lines)
    res = analyze_inline_pollution(lines, se, sn)
    assert "苏玖璃" in res.proper_nouns
    peeled, _ = apply_and_verify(lines, se + res.pre_edits)
    assert "苏玖璃" in peeled[13]
    # All candidate sites inside 苏玖璃 must NOT be tagged
    t14 = next(t for t in res.tasks if t.line_num == 14)
    assert all(c.orig_start >= 3 for c in t14.candidates)
    # Traditional line preserves 靈
    assert "精靈之森" in peeled[14]
    # Protected words intact
    assert "队伍" in peeled[15] and "肆意" in peeled[15] and "鸸鹋" in peeled[15] and "滴熘熘" in peeled[15]

    # 2. Yueyi 10-line banner, 【%测_试...】——..., Telegram promo block, Faloo, fullwidth/truncated 首发网址
    ad_lines = [
        "━━━━━━━━━━━━━━━",
        "📚 测试向小说频道",
        "　 👉 https://t.me/example_promo_chan",
        "⚠️ 免责声明：本书收集自互联网，仅供学习交流使用",
        "━━━━━━━━━━━━━━━",
        "本书由【月漪】整理，小说版权归原作者所有，文本仅供个人学习和试读，请在下载后24小时内删除，想看请去支持订阅正版小说，拒绝盗版！",
        "【月漪】提醒您：合理安排阅读时间，杜绝沉迷网络小说！更多全网小说尽在【月漪免费外群】。",
        "【月漪免费外群1号】——123456789",
        "本群免费提取全网平台已购vip章节，制成txt等格式。有想提取的私聊群主。",
        "【%测_试@小#说*群】——1+2`3*45\"67:89.0",
        "（看暴爽小说，就上飞卢小说网！）",
        "事到如今， 首发网址ｈｔｔps://m.ｌｉｎｇdian",
        "林若雪额头的印记越发滚烫。. 首发网址h",
        "正文正常句子。（）原本空括号要保留。",
    ]
    se_ad, _, _, _ = detect_soup_and_ads(ad_lines)
    out_ad, clr_ad = apply_and_verify(ad_lines, se_ad)
    assert {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.issubset(clr_ad)
    assert out_ad[10] == ""
    assert out_ad[11] == "事到如今，"
    assert out_ad[12] == "林若雪额头的印记越发滚烫。"
    assert out_ad[13] == ad_lines[13]
    # Post-edit cleanup preserves originally empty parens
    assert cleanup_post_edit_artifacts(ad_lines, se_ad) == []


def test_quality_fix_yueyi_clusters_and_negative_controls():
    from novel_wash.inline_rules import cleanup_post_edit_artifacts

    lines = [
        "第216章 在测试之前",
        "甚至衫咝〇⑺２洱斯⑧思有些单薄。",
        "艾因诺l𝝞Ngi①②𝙥𝗱·芙兰达。",
        "（𝜐）（）𝕵ԃⓧV恶魔不是全天候都在的。",
        "差𝜶《𝗎8点一屁股坐在地上。",
        "只要拿1iNqṁ月漪到日记。",
        "第216章 侕球陾‌іɩ(易U（三⁤）‏7球紦迩‌‎在猎户座之下",
        "3.2亿退赛，1.12%利率，3、2、1开战！",
        "C++与K-12，还有Mr.Smith、α粒子、①首先、他买了2个苹果，以及(=ﾟωﾟ)ﾉ。",
        "原本就有的（）不能删，刚被清空的（𝜐）要删。",
    ]
    se, _, _, sn = detect_soup_and_ads(lines)
    res = analyze_inline_pollution(lines, se, sn)
    merged = se + res.pre_edits
    post = cleanup_post_edit_artifacts(lines, merged)
    peeled, _ = apply_and_verify(lines, merged + post)

    assert peeled[2] == "艾因诺·芙兰达。"
    assert peeled[3] == "恶魔不是全天候都在的。"
    assert peeled[4] == "差点一屁股坐在地上。"
    assert peeled[5] == "只要拿到日记。"
    # Poisoned heading preserved `第216章 ` prefix while stripping exotic/rare watermark glue
    assert peeled[6].startswith("第216章 ") and "在猎户座之下" in peeled[6] and "\u200c" not in peeled[6]
    # Window candidates on L2 (衫咝洱斯思) and L7 (球易球) are tagged for LLM consensus
    t2 = next(t for t in res.tasks if t.line_num == 2)
    assert {"衫", "咝", "洱", "斯", "思"}.issubset({c.char for c in t2.candidates})
    # Negative controls untouched
    assert peeled[7] == lines[7]
    assert peeled[8] == lines[8]
    assert peeled[9] == "原本就有的（）不能删，刚被清空的要删。"

    # Stranded post-edit glue cleanup (。·”, 整个）人, 只~,-越已是, 拿1iNq到, 差8点)
    stub_lines = [
        "“……交给我吧。ṁ·”",
        "整个（三）人嵌进了石头里。",
        "整个𝜶）人嵌进了石头里。",
        "只要拿1iNq\u200c到日记。",
        "差点𝜶8点摔倒；今天18点吃饭。",
    ]
    se_s, _, _, sn_s = detect_soup_and_ads(stub_lines)
    res_s = analyze_inline_pollution(stub_lines, se_s, sn_s)
    merged_s = se_s + res_s.pre_edits
    post_s = cleanup_post_edit_artifacts(stub_lines, merged_s)
    out_s, _ = apply_and_verify(stub_lines, merged_s + post_s)
    assert out_s[0] == "“……交给我吧。”"
    assert out_s[2] == "整个人嵌进了石头里。"
    assert out_s[3] == "只要拿到日记。"
    assert "今天18点吃饭。" in out_s[4]

    # Additional stranded glue cleanup across simulated LLM/Yueyi span deletions
    from novel_wash.edit_log import EditOp
    stub_lines2 = [
        "只~,\u200c-越已是",          # `\u200c` yueyi del + `,` llm del -> `~,-` & `越已` peeled cleanly
        "拿1iNq\u200c到",            # `1iNq` glued next to `\u200c` yueyi del
        "艾因诺丝i\u200c①·芙兰达",   # `i` and `①` split across `\u200c` yueyi del, keep `·`
    ]
    se_2, _, _, sn_2 = detect_soup_and_ads(stub_lines2)
    res_2 = analyze_inline_pollution(stub_lines2, se_2, sn_2)
    sim_dels = [
        EditOp(line=1, start=2, end=3, op="del", before=",", after="", tier="tier_c", reason="tier_c_llm", source="test", status="approved"),
    ]
    merged_2 = se_2 + res_2.pre_edits + sim_dels
    post_2 = cleanup_post_edit_artifacts(stub_lines2, merged_2)
    out_2, _ = apply_and_verify(stub_lines2, merged_2 + post_2)
    assert out_2[0] == "只是"
    assert out_2[1] == "拿到"
    assert out_2[2] == "艾因诺丝·芙兰达"

    # Regression guard: reverse ASCII-alpha expansion near start of line must stay in bounds [0, len(line)]
    edge_lines = ["ab〡c无云测设⒏例试（ 六）样="]
    se_e, _, _, sn_e = detect_soup_and_ads(edge_lines)
    res_e = analyze_inline_pollution(edge_lines, se_e, sn_e)
    merged_e = se_e + res_e.pre_edits
    for op in merged_e:
        assert 0 <= op.start <= op.end <= len(edge_lines[op.line - 1])
    out_e, _ = apply_and_verify(edge_lines, merged_e + cleanup_post_edit_artifacts(edge_lines, merged_e))
    assert "ab〡c" not in out_e[0] and "⒏" not in out_e[0]



def test_round9_lingdian_scheme_port_iri_isolation():
    extra = [
        "他访问//example.org/m.lingdian.com/foo，然后离开。",
        "他访问https://m.lingdian.com:8443/book，然后离开。",
        "他访问https://example.com/路径/m.lingdian.com，然后离开。",
    ]
    assert detect_explicit_ads_only(extra) == []
    full_ops, _, _, _ = detect_soup_and_ads(extra)
    assert full_ops == []


def test_codex_blockers_1_and_2a_prose_digits_acronyms_and_telegram_structure():
    from novel_wash.inline_rules import cleanup_post_edit_artifacts

    # Blocker 1: standalone ASCII/full-width digits and model/acronym tokens (`IBM`, `GPT4`, `5G`, `A320`) survive while `8a` peels
    lines = [
        "他还差8\uaac2点才到终点。",
        "他还差８\uaac2点才到终点。",
        "他穿旧IBM\uaac2月漪牌外套。",
        "他穿旧GPT4\uaac2牌外套。",
        "信号切换到了5G\uaac2网络。",
        "手机显示\uaac25G信号，窗外飞过\uaac2A320客机。",
        "他还差8a\uaac2点才到终点。",
    ]
    se, _, _, sn = detect_soup_and_ads(lines)
    res = analyze_inline_pollution(lines, se, sn)
    merged = se + res.pre_edits
    post = cleanup_post_edit_artifacts(lines, merged)
    peeled, _ = apply_and_verify(lines, merged + post)
    assert peeled[0] == "他还差8点才到终点。"
    assert peeled[1] == "他还差８点才到终点。"
    assert peeled[2] == "他穿旧IBM牌外套。"
    assert peeled[3] == "他穿旧GPT4牌外套。"
    assert peeled[4] == "信号切换到了5G网络。"
    assert peeled[5] == "手机显示5G信号，窗外飞过A320客机。"
    assert peeled[6] == "他还差点才到终点。"

    # Blocker 2a: story paragraphs between dividers with prose `机器人`/`频道` + stray t.me link must NOT be erased
    story_window = [
        "━━━━━━━━━━━━━━━",
        "第一段正常剧情，主角看着街角坏掉的机器人和电视里的频道叹了口气。",
        "https://t.me/example_promo_chan",
        "第二段正常剧情，他推开木门走了进去，灯光瞬间亮了起来。",
        "━━━━━━━━━━━━━━━",
    ]
    assert detect_explicit_ads_only(story_window) == []
    se2, _, _, _ = detect_soup_and_ads(story_window)
    assert se2 == []



