import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from novel_wash.epub_export import build_epub, verify_epub_text_preservation
from novel_wash.epub_fix import check_and_repair_epub, repair_xml_text, strip_epub_telegram_banners
from novel_wash.harmony_restore import accept_restore, apply_fills, classify_line, restore_known_slots
from novel_wash.pua_solver import solve_pua_cipher


def test_harmony_slot_invariants():
    assert classify_line("应对突**况的紧急预案") == "fill"
    assert classify_line("不是每一滴牛奶，都叫特〇〇。") == "false_positive"
    assert classify_line("《如影**》") == "book_title"
    assert apply_fills("高产似**。", ["母猪"]) == "高产似母猪。"
    assert accept_restore("高产似**。", "高产似母猪。") == "高产似母猪。"
    assert accept_restore("高产似**。", "高产似母猪！") is None
    assert accept_restore("雪**嫩的脖子", "雪白的脖子") is None

    edits = restore_known_slots(["应对突**况，保持冷静。"])
    assert len(edits) == 1
    assert edits[0].before == "**" and edits[0].after == "发状"

    # Full run of asterisks (`***`) & circles (`〇〇`) restored cleanly without trailing `*`
    from novel_wash.edit_log import apply_and_verify
    sample_lines = ["心情畅***。", "突〇〇况发生了。"]
    h_edits = restore_known_slots(sample_lines)
    out, _ = apply_and_verify(sample_lines, h_edits)
    assert out == ["心情畅快感。", "突发状况发生了。"]

    # LLM slot fill supports numeric string keys ("1") as well as "L2" and rejects stacked/banned fills
    llm_lines = ["我喜欢**和**。", "高产似**。", "这是**的。"]
    def fake_harmony_llm(sys_p: str, usr_p: str) -> str:
        return '{"1": ["重文", "轻理"], "L2": ["母猪"], "3": ["什么"]}'

    llm_edits = restore_known_slots(llm_lines, llm_fill=fake_harmony_llm)
    out2, _ = apply_and_verify(llm_lines, llm_edits)
    assert out2 == ["我喜欢重文和轻理。", "高产似母猪。", "这是**的。"]


def test_pua_cache_roundtrip_and_failure_gates(tmp_path: Path):
    # Encoded version of "他们发现阴暗巷子里有一把锋利剪刀。" -> breaks 2-char jieba words before decode
    lines = ["“他\ue3e8\ue3e9现\ue3ea暗巷子里有一把锋利\ue3eb刀。”"]

    def mock_llm(sys_p: str, usr_p: str, model: str | None) -> str:
        import json
        out = {}
        for line in usr_p.splitlines():
            pid = line.split()[0]
            if "他⟦?⟧" in line:
                out[pid] = "们"
            elif "⟦?⟧现" in line:
                out[pid] = "发"
            elif "⟦?⟧暗" in line:
                out[pid] = "阴"
            else:
                out[pid] = "剪"
        return json.dumps(out, ensure_ascii=False)

    cache_dir = tmp_path / "cache"
    res = solve_pua_cipher(lines * 20, cache_dir, llm_call=mock_llm)
    assert res.unique_codepoints == 4
    assert res.cache_hit is False
    assert len(res.edits) == 80

    # Reload cache cleanly with ZERO LLM calls allowed
    def fail_if_called(sys_p: str, usr_p: str, model: str | None) -> str:
        raise AssertionError("Cache roundtrip should not invoke LLM")

    res_cached = solve_pua_cipher(lines * 20, cache_dir, llm_call=fail_if_called)
    assert res_cached.cache_hit is True
    assert len(res_cached.edits) == 80

    # Wrong book sharing the exact same PUA codepoint set must NOT blindly hit swapped cipher cache
    swapped_lines = ["“他\ue3eb\ue3ea现\ue3e9暗巷子里有一把锋利\ue3e8刀。”"] * 20
    res_other = solve_pua_cipher(swapped_lines, cache_dir, llm_call=None)
    assert res_other.cache_hit is False
    assert res_other.edits == []

    # Large-font (>=16 codepoints) 2-codepoint swap must ALSO miss cache instead of blindly approving wrong cipher
    import json
    from collections import Counter
    from novel_wash.pua_solver import _fingerprint_codepoints, _source_pua_fingerprint

    large_truth = {
        "\ue700": "他", "\ue701": "们", "\ue702": "在", "\ue703": "阴",
        "\ue704": "暗", "\ue705": "巷", "\ue706": "子", "\ue707": "里",
        "\ue708": "发", "\ue709": "现", "\ue70a": "了", "\ue70b": "一",
        "\ue70c": "把", "\ue70d": "锋", "\ue70e": "利", "\ue70f": "剪",
        "\ue710": "刀", "\ue711": "的",
    }
    plain_rich = ["“他们在阴暗巷子里发现了一把锋利的剪刀，真的发现了一把的。”"] * 20
    book_a_lines = [l.translate(str.maketrans({v: k for k, v in large_truth.items()})) for l in plain_rich]
    swapped_truth = dict(large_truth)
    swapped_truth["\ue70a"], swapped_truth["\ue711"] = swapped_truth["\ue711"], swapped_truth["\ue70a"]
    book_b_lines = [l.translate(str.maketrans({v: k for k, v in swapped_truth.items()})) for l in plain_rich]

    large_cache = tmp_path / "large_cache"
    large_cache.mkdir()
    cps_a = sorted(large_truth.keys())
    fp_a = _fingerprint_codepoints(cps_a)
    sfp_a = _source_pua_fingerprint(book_a_lines, cps_a, Counter({cp: 20 for cp in cps_a}))
    (large_cache / f"pua_map_{fp_a}_{sfp_a[:8]}.json").write_text(
        json.dumps({"fingerprint": fp_a, "source_fingerprint": sfp_a, "char_map": large_truth, "uncertain": []}),
        encoding="utf-8",
    )
    res_b = solve_pua_cipher(book_b_lines, large_cache, llm_call=None)
    assert res_b.cache_hit is False
    assert res_b.edits == []

    # Blocker 4: Failed coherence (0.0 -> 0.0) & duplicate mapping MUST emit zero approved edits and no cache
    bad_lines = ["\ue400 \ue401"] * 5
    def dup_llm(sys_p: str, usr_p: str, model: str | None) -> str:
        return '{"P001": "甲", "P002": "甲"}'

    bad_res = solve_pua_cipher(bad_lines, tmp_path / "bad_cache", llm_call=dup_llm)
    assert bad_res.edits == []
    assert any("gate_failed" in x for x in bad_res.uncertain_codepoints)
    assert any(":unmapped" in x for x in bad_res.uncertain_codepoints)
    assert list((tmp_path / "bad_cache").glob("*.json")) == []


def test_pua_paragraph_bijective_solver_resolves_collisions(tmp_path: Path):
    import json

    # 18 distinct PUA codepoints (>15 triggers paragraph bijective mode)
    truth = {
        "\ue600": "他", "\ue601": "们", "\ue602": "在", "\ue603": "阴",
        "\ue604": "暗", "\ue605": "巷", "\ue606": "子", "\ue607": "里",
        "\ue608": "发", "\ue609": "现", "\ue60a": "了", "\ue60b": "一",
        "\ue60c": "把", "\ue60d": "锋", "\ue60e": "利", "\ue60f": "剪",
        "\ue610": "刀", "\ue611": "啊",
    }
    rev = {v: f"{ord(k):04X}" for k, v in truth.items()}
    enc_map = {v: k for k, v in truth.items()}
    plain_lines = [
        "第一章 暗巷",
        "“他们在阴暗巷子里发现了一把锋利剪刀啊。”",
        "阴暗巷子里有一把锋利剪刀，他们发现了啊。",
    ] * 12
    pua_lines = [l.translate(str.maketrans(enc_map)) for l in plain_lines]

    calls = 0

    def fake_llm(sys_p: str, usr_p: str, model: str | None) -> str:
        nonlocal calls
        calls += 1
        if "[E6" in usr_p:
            # Return paragraph hex mapping; deliberately inject 1 collision on E611 -> '一' so bijection picks E60B and Pass 2 fixes E611
            ans = {f"{ord(k):04X}": v for k, v in truth.items() if k != "\ue611"}
            ans["E611"] = "一"
            return json.dumps(ans, ensure_ascii=False)
        # Hole-fill pass
        out = {}
        for line in usr_p.splitlines():
            if line.startswith("P") and "当前猜测=" not in line:
                pid = line.split()[0]
                out[pid] = "啊"
        return json.dumps(out, ensure_ascii=False)

    res = solve_pua_cipher(pua_lines, tmp_path / "cache", llm_call=fake_llm, primary_model="glm-5.3-flash")
    assert calls >= 2
    assert len(set(res.char_map.values())) == len(res.char_map) == 18
    assert res.char_map["\ue60b"] == "一"
    assert res.char_map["\ue611"] == "啊"
    assert res.jieba_ratio_after > res.jieba_ratio_before + 0.15
    assert list((tmp_path / "cache").glob("pua_map_*.json"))



def test_epub_export_produces_valid_epub3_and_preserves_consecutive_headings_and_prose(tmp_path: Path):
    reader_text = (
        "第一卷\n\n"
        "序章\n\n"
        "　　“你好，世界 & 冒险。”\n\n"
        "第303章: 卷末总结\n\n"
        "第三章已经结束了。\n\n"
        "第三卷也算是圆满结束了。\n\n"
        "第一节理论课，老师走了进来。\n\n"
        "第一回：他匆忙赶回家，又想起有人敲门。\n\n"
        "第02集：陨落仙尊，从搬砖开始\n\n"
        "第03集：打脸中介\n\n"
        "番外\n"
    )
    out = tmp_path / "test_book.epub"
    cnt = build_epub(reader_text, out, title="测试小说", author="测试作者")
    assert cnt == 6
    assert out.exists()

    ok, diffs = verify_epub_text_preservation(reader_text, out)
    assert ok, diffs

    with zipfile.ZipFile(out, "r") as zf:
        infos = zf.infolist()
        assert infos[0].filename == "mimetype"
        assert infos[0].compress_type == zipfile.ZIP_STORED
        assert zf.read("mimetype") == b"application/epub+zip"
        for name in zf.namelist():
            if name.endswith((".xml", ".xhtml", ".opf", ".ncx")):
                ET.fromstring(zf.read(name).decode("utf-8"))
        # Prose lines beginning with 第...章/回/卷/节 must be <p>, never swallowed as <h1>
        ch3 = zf.read("OEBPS/ch_0003.xhtml").decode("utf-8")
        assert "<p>第三章已经结束了。</p>" in ch3
        assert "<p>第三卷也算是圆满结束了。</p>" in ch3
        assert "<p>第一节理论课，老师走了进来。</p>" in ch3
        assert "<p>第一回：他匆忙赶回家，又想起有人敲门。</p>" in ch3


def test_epub_fix_repairs_unescaped_ampersand_entities_and_encoding(tmp_path: Path):
    broken_xhtml = '<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml"><body><p>A&nbsp;B &amp; “*/@#￥%……&”</p></body></html>'
    fixed = repair_xml_text(broken_xhtml)
    root = ET.fromstring(fixed)
    assert "&amp;" in fixed
    assert "&amp;nbsp;" not in fixed
    p_text = "".join(root.find(".//{http://www.w3.org/1999/xhtml}p").itertext())  # type: ignore[union-attr]
    assert "\u00a0" in p_text

    gb_broken_xml = '<?xml version="1.0" encoding="gb18030"?><root><t>中文&测试</t></root>'.encode("gb18030")
    broken_epub = tmp_path / "broken.epub"
    with zipfile.ZipFile(broken_epub, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("OEBPS/ch1.html", broken_xhtml)
        zf.writestr("OEBPS/meta.xml", gb_broken_xml)

    res = check_and_repair_epub(broken_epub, tmp_path / "out")
    assert set(res.bad_xml_entries) == {"OEBPS/ch1.html", "OEBPS/meta.xml"}
    assert set(res.repaired_entries) == {"OEBPS/ch1.html", "OEBPS/meta.xml"}
    assert res.remaining_bad_entries == []
    assert res.fixed_path is not None
    with zipfile.ZipFile(res.fixed_path, "r") as zf:
        assert ET.fromstring(zf.read("OEBPS/meta.xml").decode("utf-8")).findtext("t") == "中文&测试"


def test_round4_pua_full_digest_beyond_sample_and_inline_toc_epub(tmp_path: Path):
    import json
    from collections import Counter
    from novel_wash.pua_solver import _fingerprint_codepoints, _source_pua_fingerprint

    # Part A: Swapping two PUA chars on line 60 (>40-line sample) with identical codepoint counts must miss cache
    truth = {
        "\ue700": "他", "\ue701": "们", "\ue702": "在", "\ue703": "阴",
        "\ue704": "暗", "\ue705": "巷", "\ue706": "子", "\ue707": "里",
        "\ue708": "发", "\ue709": "现", "\ue70a": "了", "\ue70b": "一",
        "\ue70c": "把", "\ue70d": "锋", "\ue70e": "利", "\ue70f": "剪",
        "\ue710": "刀", "\ue711": "的",
    }
    plain = ["“他们在阴暗巷子里发现了一把锋利的剪刀，真的发现了一把的。”"] * 65
    enc_map = {v: k for k, v in truth.items()}
    book_lines = [l.translate(str.maketrans(enc_map)) for l in plain]

    cache_dir = tmp_path / "pua_full_cache"
    cache_dir.mkdir()
    cps = sorted(truth.keys())
    fp = _fingerprint_codepoints(cps)
    sfp = _source_pua_fingerprint(book_lines, cps, Counter({c: 65 for c in cps}))
    assert len(sfp) == 64
    (cache_dir / f"pua_map_{fp}_{sfp[:8]}.json").write_text(
        json.dumps({"fingerprint": fp, "source_fingerprint": sfp, "char_map": truth, "uncertain": []}),
        encoding="utf-8",
    )

    hit_res = solve_pua_cipher(book_lines, cache_dir, llm_call=None)
    assert hit_res.cache_hit is True and len(hit_res.edits) > 0

    # Swap \ue700 and \ue701 only on line 60 -> codepoint counts and first 40 lines stay identical!
    mutated = list(book_lines)
    mutated[59] = mutated[59].replace("\ue700", "X").replace("\ue701", "\ue700").replace("X", "\ue701")
    miss_res = solve_pua_cipher(mutated, cache_dir, llm_call=None)
    assert miss_res.cache_hit is False
    assert miss_res.edits == []

    # Part B: Inline numbered TOC run (>= 20 consecutive headings) demotes TOC items to <p> while preserving real chapters
    toc_lines = [f"{i}.第{i}章 标题{i}" for i in range(1, 26)]
    reader_text = (
        "内容简介：\n\n这是一段简介。\n\n章节目录：\n\n"
        + "\n\n".join(toc_lines)
        + "\n\n1.第1章 诈尸\n\n第一段正文。\n\n2.第2章 倒霉蛋\n\n第二段正文。\n\n25.第25章 标题25\n\n结尾正文。\n"
    )
    ep_out = tmp_path / "inline_toc.epub"
    cnt = build_epub(reader_text, ep_out, title="目录测试", author="作者")
    assert cnt == 4  # 序(含内嵌目录) + 第1章 + 第2章 + 第25章
    ok, diffs = verify_epub_text_preservation(reader_text, ep_out)
    assert ok, diffs
    with zipfile.ZipFile(ep_out, "r") as zf:
        ch1 = zf.read("OEBPS/ch_0001.xhtml").decode("utf-8")
        ch2 = zf.read("OEBPS/ch_0002.xhtml").decode("utf-8")
        ch4 = zf.read("OEBPS/ch_0004.xhtml").decode("utf-8")
        assert "<p>2.第2章 标题2</p>" in ch1
        assert "<p>25.第25章 标题25</p>" in ch1
        assert "<h1>1.第1章 诈尸</h1>" in ch2
        assert "<h1>25.第25章 标题25</h1>" in ch4


def test_round5_inline_toc_before_marker_and_duplicate_same_num_headings(tmp_path: Path):
    # Case 1: 20-entry inline TOC followed by "以上是目录。" must demote ALL 20 entries to <p>
    toc_20 = "\n\n".join(f"第{i}章 标题{i}" for i in range(1, 21))
    reader_toc = f"目录\n\n{toc_20}\n\n以上是目录。\n\n第1章 标题1\n\n正文。\n"
    ep1 = tmp_path / "toc20.epub"
    cnt1 = build_epub(reader_toc, ep1, title="目录测试20", author="作者")
    assert cnt1 == 2  # 序 (with TOC + 以上是目录。) + 第1章 标题1
    ok1, diffs1 = verify_epub_text_preservation(reader_toc, ep1)
    assert ok1, diffs1
    with zipfile.ZipFile(ep1, "r") as zf:
        ch1 = zf.read("OEBPS/ch_0001.xhtml").decode("utf-8")
        ch2 = zf.read("OEBPS/ch_0002.xhtml").decode("utf-8")
        assert "<p>第20章 标题20</p>" in ch1
        assert "<p>以上是目录。</p>" in ch1
        assert "<h1>第1章 标题1</h1>" in ch2
        assert "<p>正文。</p>" in ch2

    # Case 2: Consecutive duplicate same-number heading (3520.第3520章 震惊 + 第3520章震惊) must not emit empty chapter
    reader_dup = "3520.第3520章 震惊\n\n第3520章震惊\n\n光明之翼正处于震撼之中。\n"
    ep2 = tmp_path / "dup3520.epub"
    cnt2 = build_epub(reader_dup, ep2, title="重章测试", author="作者")
    assert cnt2 == 1
    ok2, diffs2 = verify_epub_text_preservation(reader_dup, ep2)
    assert ok2, diffs2
    with zipfile.ZipFile(ep2, "r") as zf:
        only_ch = zf.read("OEBPS/ch_0001.xhtml").decode("utf-8")
        assert "<h1>3520.第3520章 震惊</h1>" in only_ch
        assert "<p>第3520章震惊</p>" in only_ch
        assert "<p>光明之翼正处于震撼之中。</p>" in only_ch


def test_round6_triple_duplicate_headings_and_masked_title_upgrade(tmp_path: Path):
    # Case 1: 3 consecutive same-number headings fold into 1 chapter
    reader3 = "3520.第3520章 震惊\n\n第3520章震惊\n\n第三千五百二十章 震惊\n\n正文。\n"
    ep3 = tmp_path / "triple3520.epub"
    cnt3 = build_epub(reader3, ep3, title="三重章测试", author="作者")
    assert cnt3 == 1
    ok3, diffs3 = verify_epub_text_preservation(reader3, ep3)
    assert ok3, diffs3
    with zipfile.ZipFile(ep3, "r") as zf:
        ch = zf.read("OEBPS/ch_0001.xhtml").decode("utf-8")
        assert "<h1>3520.第3520章 震惊</h1>" in ch
        assert "<p>第3520章震惊</p>" in ch
        assert "<p>第三千五百二十章 震惊</p>" in ch

    # Case 2: Masked **** heading followed by unmasked heading picks unmasked title for <h1> + nav while keeping order
    masked_reader = (
        "第850章：张副会长！您终于回来了！（7400，****\n\n"
        "第850章：张副会长！您终于回来了！（7400+，求月票）\n\n"
        "“哭了？”\n"
    )
    ep_m = tmp_path / "masked850.epub"
    cnt_m = build_epub(masked_reader, ep_m, title="防截断测试", author="作者")
    assert cnt_m == 1
    ok_m, diffs_m = verify_epub_text_preservation(masked_reader, ep_m)
    assert ok_m, diffs_m
    with zipfile.ZipFile(ep_m, "r") as zf:
        ch = zf.read("OEBPS/ch_0001.xhtml").decode("utf-8")
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        assert "<p>第850章：张副会长！您终于回来了！（7400，****</p>" in ch
        assert "<h1>第850章：张副会长！您终于回来了！（7400+，求月票）</h1>" in ch
        assert "<title>第850章：张副会长！您终于回来了！（7400+，求月票）</title>" in ch
        assert "第850章：张副会长！您终于回来了！（7400+，求月票）" in nav
        assert "（7400，****" not in nav


def test_round7_duplicate_20_run_and_bare_vs_titled_heading(tmp_path: Path):
    # Case 1: 1 masked + 19 duplicate headings for the same chapter (20 consecutive lines) must fold into 1 chapter
    dup20_reader = (
        "第850章：张副会长****\n\n"
        + "\n\n".join(["第850章：张副会长"] * 19)
        + "\n\n正文。\n"
    )
    ep1 = tmp_path / "dup20_same_ch.epub"
    cnt1 = build_epub(dup20_reader, ep1, title="二十重章测试", author="作者")
    assert cnt1 == 1
    ok1, diffs1 = verify_epub_text_preservation(dup20_reader, ep1)
    assert ok1, diffs1
    with zipfile.ZipFile(ep1, "r") as zf:
        ch1 = zf.read("OEBPS/ch_0001.xhtml").decode("utf-8")
        nav1 = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        assert "<h1>第850章：张副会长</h1>" in ch1
        assert "<p>正文。</p>" in ch1
        assert ">序<" not in nav1

    # Case 2: Bare chapter heading ("第850章") followed by unspaced titled heading ("第850章张副会长回来了") picks titled heading
    bare_reader = "第850章\n\n第850章张副会长回来了\n\n正文。\n"
    ep2 = tmp_path / "bare_vs_titled.epub"
    cnt2 = build_epub(bare_reader, ep2, title="有题标题优先测试", author="作者")
    assert cnt2 == 1
    ok2, diffs2 = verify_epub_text_preservation(bare_reader, ep2)
    assert ok2, diffs2
    with zipfile.ZipFile(ep2, "r") as zf:
        ch2 = zf.read("OEBPS/ch_0001.xhtml").decode("utf-8")
        assert "<p>第850章</p>" in ch2
        assert "<h1>第850章张副会长回来了</h1>" in ch2
        assert "<p>正文。</p>" in ch2


def test_round8_toc_followed_by_duplicate_first_chapter_headings(tmp_path: Path):
    toc_20 = "\n\n".join(f"第{i}章 标题{i}" for i in range(1, 21))
    reader_text = (
        f"目录\n\n{toc_20}\n\n"
        "第1章 正式标题\n\n"
        "第1章\n\n"
        "第一章正文。\n\n"
        "第2章 下一章\n\n"
        "第二章正文。\n"
    )
    ep = tmp_path / "toc_then_dup_ch1.epub"
    cnt = build_epub(reader_text, ep, title="目录后重章测试", author="作者")
    assert cnt == 3
    ok, diffs = verify_epub_text_preservation(reader_text, ep)
    assert ok, diffs
    with zipfile.ZipFile(ep, "r") as zf:
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        ch1 = zf.read("OEBPS/ch_0001.xhtml").decode("utf-8")
        ch2 = zf.read("OEBPS/ch_0002.xhtml").decode("utf-8")
        ch3 = zf.read("OEBPS/ch_0003.xhtml").decode("utf-8")
        assert "第1章 正式标题" in nav
        assert "第2章 下一章" in nav
        assert "第一章正文。" not in ch1
        assert "<h1>第1章 正式标题</h1>" in ch2
        assert "<p>第1章</p>" in ch2
        assert "<p>第一章正文。</p>" in ch2
        assert "<h1>第2章 下一章</h1>" in ch3
        assert "<p>第二章正文。</p>" in ch3


def test_round9_toc_followed_by_same_last_chapter_restart(tmp_path: Path):
    toc_20 = "\n\n".join(f"第{i}章 目录标题{i}" for i in range(1, 21))
    reader_text = (
        f"目录\n\n{toc_20}\n\n"
        "第20章 正式标题\n\n"
        "第20章\n\n"
        "这里是正文。\n\n"
        "第21章 下一章\n\n"
        "这是下一章的正文。\n"
    )
    ep = tmp_path / "toc_then_dup_ch20.epub"
    cnt = build_epub(reader_text, ep, title="目录末章重开测试", author="作者")
    assert cnt == 3
    ok, diffs = verify_epub_text_preservation(reader_text, ep)
    assert ok, diffs
    with zipfile.ZipFile(ep, "r") as zf:
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        ch1 = zf.read("OEBPS/ch_0001.xhtml").decode("utf-8")
        ch2 = zf.read("OEBPS/ch_0002.xhtml").decode("utf-8")
        ch3 = zf.read("OEBPS/ch_0003.xhtml").decode("utf-8")
        assert "第20章 正式标题" in nav
        assert "第21章 下一章" in nav
        assert "<p>第20章 目录标题20</p>" in ch1
        assert "这里是正文。" not in ch1
        assert "<h1>第20章 正式标题</h1>" in ch2
        assert "<p>第20章</p>" in ch2
        assert "<p>这里是正文。</p>" in ch2
        assert "<h1>第21章 下一章</h1>" in ch3


def test_quality_fix_numbered_chapter_scheme_and_spaced_heading(tmp_path: Path):
    from novel_wash.edit_log import detect_numbered_scheme, is_chapter_heading

    assert is_chapter_heading("第1 章 穿书畜生") is True

    # Numbered chapters with volume reset + prose traps like 3.2亿 / 1.12% / 3、2、1开战！ / 1. 首先
    body = "　　这是一段足够长的正文描写。" * 25
    parts = ["简介：\n\n3、2、1开战！\n\n3.2亿退赛。\n\n1. 首先说明一下。"]
    for i in range(1, 13):
        parts.append(f"{i}.第{i}回的冒牌标题{i}\n\n{body}")
    parts.append("第一卷\n")
    for i in range(1, 6):
        parts.append(f"{i}.第二卷标题{i}\n\n{body}")
    text = "\n\n".join(parts)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    scheme = detect_numbered_scheme(lines)
    assert scheme is not None
    assert len(scheme) == 17  # 12 + 5 real numbered headings; excludes 3、2、1 / 3.2亿 / 1. 首先

    ep = tmp_path / "numbered.epub"
    cnt = build_epub(text, ep, title="编号分章测试", author="作者")
    assert cnt == 19  # 序 + 12 + 第一卷 + 5
    ok, diffs = verify_epub_text_preservation(text, ep)
    assert ok, diffs


def test_quality_fix_pua_hungarian_swap_and_3cycle():
    from novel_wash.pua_solver import _max_weight_bipartite_matching

    # 2-swap + 3-cycle assignment check
    weights = [
        [1.0, 9.0, 2.0, 0.5, 0.5, 0.5],  # row 0 prefers col 1
        [9.0, 1.0, 2.0, 0.5, 0.5, 0.5],  # row 1 prefers col 0
        [1.0, 1.0, 8.0, 0.5, 0.5, 0.5],  # row 2 prefers col 2
    ]
    ans = _max_weight_bipartite_matching(weights)
    assert ans == [1, 0, 2]

    # 3-cycle + unmapped dummy eviction
    weights2 = [
        [1.0, 8.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5],  # row 0 -> 1
        [1.0, 1.0, 8.0, 2.0, 0.5, 0.5, 0.5, 0.5],  # row 1 -> 2
        [8.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],  # row 2 -> 0
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 3.0],  # row 3 prefers dummy col 7
    ]
    ans2 = _max_weight_bipartite_matching(weights2)
    assert ans2 == [1, 2, 0, 7]


def test_quality_fix_epub_strip_telegram_banner(tmp_path: Path):
    ep = tmp_path / "tg.epub"
    xhtml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<p>━━━━━━━━━━━━━━━<br/>📚 测试向小说频道<br/>👉 https://t.me/example_promo_chan<br/>'
        '⚠️ 免责声明：本书收集自互联网，仅供学习交流使用<br/>━━━━━━━━━━━━━━━</p>'
        '<p>第一章 正文在这里。</p>'
        '</body></html>'
    )
    with zipfile.ZipFile(ep, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("OEBPS/ch1.xhtml", xhtml)

    res = check_and_repair_epub(ep, tmp_path / "fixed", strip_telegram_banner=True)
    assert res.fixed_path is not None
    with zipfile.ZipFile(res.fixed_path, "r") as zf:
        cleaned = zf.read("OEBPS/ch1.xhtml").decode("utf-8")
        ET.fromstring(cleaned)
        assert "t.me/" not in cleaned
        assert "第一章 正文在这里。" in cleaned


def test_codex_blockers_2_6_7_epub_encoding_pua_evidence_and_std_chapter_guard(tmp_path: Path):
    from collections import Counter
    from novel_wash.epub_export import split_chapters
    from novel_wash.pua_solver import _solve_pua_matching_round

    # Blocker 2a + 2b: legacy gb18030 XHTML declaration + caf\xe9 + story window preservation
    ep2 = tmp_path / "legacy.epub"
    xhtml_legacy = (
        '<?xml version="1.0" encoding="gb18030"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        '<meta http-equiv="Content-Type" content="text/html; charset=gb18030"/>'
        '</head><body>'
        '<p>━━━━━━━━━━━━━━━<br/>测试向小说频道<br/>https://t.me/example_promo_chan<br/>'
        '免责声明：本书收集自互联网，仅供学习交流使用<br/>━━━━━━━━━━━━━━━</p>'
        '<p>━━━━━━━━━━━━━━━<br/>主角走进街角的 café 准备喝杯咖啡休息片刻。<br/>'
        'https://t.me/example_promo_chan<br/>随后他翻开笔记本开始记录线索，窗外下起了小雨。<br/>━━━━━━━━━━━━━━━</p>'
        '</body></html>'
    ).encode("gb18030")
    with zipfile.ZipFile(ep2, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("OEBPS/ch1.xhtml", xhtml_legacy)
    res2 = check_and_repair_epub(ep2, tmp_path / "fixed2", strip_telegram_banner=True)
    assert res2.fixed_path is not None
    with zipfile.ZipFile(res2.fixed_path, "r") as zf:
        raw_out = zf.read("OEBPS/ch1.xhtml")
        ET.fromstring(raw_out)
        txt_out = raw_out.decode("utf-8")
        assert 'encoding="utf-8"' in txt_out and 'charset=utf-8' in txt_out
        assert "café" in txt_out
        assert "免责声明：本书收集自互联网" not in txt_out
        assert "主角走进街角的 café 准备喝杯咖啡休息片刻。" in txt_out

    # Blocker 2c: numeric entity + newline + inline markup dialogue inside divider block + BOM+GB18030 + spaced big5 decl
    ep3 = tmp_path / "entity_and_mismatch.epub"
    xhtml_ent = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<p>━━━━━━━━━━━━━━━<br/>交流群防失联<br/>https://t.me/example_promo_chan<br/>'
        '免责声明：本书收集自互联网，仅供学习交流使用<br/>'
        '&#12300;快看\n<em>前面</em>！&#12301;他举起手大喊一声。<br/>━━━━━━━━━━━━━━━</p>'
        '</body></html>'
    ).encode("utf-8")
    xhtml_bom_gb = b"\xef\xbb\xbf" + (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>测试中文声明错配</p></body></html>'
    ).encode("gb18030")
    xhtml_big5_spaced = (
        '<?xml version="1.0" encoding = "big5"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>中文測試</p></body></html>'
    ).encode("big5")
    with zipfile.ZipFile(ep3, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("OEBPS/ent.xhtml", xhtml_ent)
        zf.writestr("OEBPS/bom_gb.xhtml", xhtml_bom_gb)
        zf.writestr("OEBPS/big5.xhtml", xhtml_big5_spaced)
    res3 = check_and_repair_epub(ep3, tmp_path / "fixed3", strip_telegram_banner=True)
    assert "OEBPS/bom_gb.xhtml" in res3.bad_xml_entries
    assert "OEBPS/bom_gb.xhtml" in res3.repaired_entries
    assert res3.remaining_bad_entries == [] and res3.fixed_path is not None
    with zipfile.ZipFile(res3.fixed_path, "r") as zf:
        bom_fixed = zf.read("OEBPS/bom_gb.xhtml").decode("utf-8")
        ET.fromstring(zf.read("OEBPS/bom_gb.xhtml"))
        assert "\ufffd" not in bom_fixed and "测试中文声明错配" in bom_fixed
        ent_txt = zf.read("OEBPS/ent.xhtml").decode("utf-8")
        assert "&#12300;快看\n<em>前面</em>！&#12301;" in ent_txt
        big5_txt = zf.read("OEBPS/big5.xhtml").decode("utf-8")
        ET.fromstring(zf.read("OEBPS/big5.xhtml"))
        assert "中文測試" in big5_txt

    ent_ns = strip_epub_telegram_banners(
        '<x.y:html xmlns:x.y="http://www.w3.org/1999/xhtml"><x.y:body><x.y:p>'
        '━━━━━━━━━━━━━━━<x.y:br/>https://t.me/example_promo_chan<x.y:br/>'
        '免责声明：本书收集自互联网，仅供学习交流使用<x.y:br/>他举起手大喊一声。<x.y:br/>━━━━━━━━━━━━━━━'
        '</x.y:p></x.y:body></x.y:html>'
    )
    assert "他举起手大喊一声。" in ent_ns
    promo_ns_inline = strip_epub_telegram_banners(
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:p.foo="http://www.w3.org/1999/xhtml"><body><p>'
        '━━━━━━━━<br/>https://t.me/example_promo_chan<br/>免责声明：<p.foo:span>本书来自互联网。</p.foo:span><br/>━━━━━━━━'
        '</p></body></html>'
    )
    assert "t.me/" not in promo_ns_inline

    ep_corrupt = tmp_path / "corrupt.epub"
    with zipfile.ZipFile(ep_corrupt, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("OEBPS/bad.xhtml", b"<?xml version='1.0'?><html\x80\xff>bad</html>")
    res_corrupt = check_and_repair_epub(ep_corrupt, tmp_path / "fixed_corrupt")
    assert "OEBPS/bad.xhtml" in res_corrupt.remaining_bad_entries
    assert "OEBPS/bad.xhtml" not in res_corrupt.repaired_entries

    # Blocker 6/7: full _polish_pua_map_globally preserves unresolved Pass-3 disputes even if Pass 4 emits a rejected alternate vote
    from novel_wash.pua_solver import _build_pua_edits, _polish_pua_map_globally, _solve_pua_matching_round, _uncertain_codepoints

    cps = [chr(0xE000 + i) for i in range(51)]
    cur_map = {cps[0]: "的", cps[1]: "习"}
    lines_51 = ["他看着手里" + cps[0] + "旧书，认真做完这道练" + cps[1] + "题。"] + [
        "这里出现了一句包含" + cp + "的短句。" for cp in cps[2:]
    ]
    lines_by_cp = {cp: [ln for ln in lines_51 if cp in ln] for cp in cps}
    cnt_51 = Counter({cp: 1 for cp in cps})
    matched = _solve_pua_matching_round(
        cps, {cps[0]: "的"}, lines_by_cp, cnt_51, {cps[0]: {"的": 3.0}}, {}, Counter(), Counter(), Counter()
    )
    assert cps[0] in matched and len(matched) == 1
    hx_1 = f"{ord(cps[1]):04X}"
    r_calls = [0]

    def _sole_rejected_vote(_s: str, _u: str, _m: str | None) -> str:
        r_calls[0] += 1
        return json.dumps({hx_1: ["虫", 0.1]}) if r_calls[0] == 1 else "{}"

    polished_map, polished_unc = _polish_pua_map_globally(
        lines_51,
        cps,
        cur_map,
        lines_by_cp,
        cnt_51,
        _sole_rejected_vote,
        primary_model="m1",
        secondary_model="m2",
        prior_uncertain=[f"U+{hx_1}:?/习"],
    )
    assert polished_map[cps[0]] == "的" and polished_map[cps[1]] == "习"
    assert any(u.startswith(f"U+{hx_1}:") for u in polished_unc)
    edits_51 = _build_pua_edits(lines_51, polished_map, source="test", exclude_cps=_uncertain_codepoints(polished_unc))
    assert [e.before for e in edits_51] == [cps[0]]

    # Blocker 8: 2-chapter 第X章 book with 20 sequential numbered items separated by >=200 chars prose hits median guard -> stays 2 chapters
    long_prose = "第一段记录日常经过的普通正文，字数刻意凑到两百余字以确保通过一百五十字符候选间距阈值，进而真正触发中位数间距检查而不是提前被候选过滤掉。" * 3
    std_book = ["第1章 初遇"]
    for idx_i in range(1, 11):
        std_book.append(f"{idx_i}. 清单条目{idx_i}")
        std_book.append(long_prose)
    std_book.append("第2章 启程")
    for idx_i in range(11, 21):
        std_book.append(f"{idx_i}. 清单条目{idx_i}")
        std_book.append(long_prose)
    chapters = split_chapters("\n".join(std_book), book_title="test_std")
    assert len(chapters) == 2
    assert chapters[0].title == "第1章 初遇" and chapters[1].title == "第2章 启程"



