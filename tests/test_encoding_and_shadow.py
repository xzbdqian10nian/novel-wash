import os
from pathlib import Path

import pytest

from novel_wash.encoding import decode_bytes, read_novel_file
from novel_wash.shadow import build_shadow
from novel_wash.triage import triage_file


def test_gb18030_and_utf8_decoding():
    text = "第一章 开始\n　　“你好，世界！”她笑道。\n"
    gb_bytes = text.encode("gb18030")
    dec = decode_bytes(gb_bytes)
    assert dec.encoding == "gb18030"
    assert not dec.is_mixed
    assert dec.lines[0] == "第一章 开始"
    assert "“你好，世界！”" in dec.lines[1]


def test_mixed_utf8_and_gb18030_within_one_chunk_no_mojibake():
    raw = "你好\n".encode("utf-8") + "世界\n".encode("gb18030")
    dec = decode_bytes(raw)
    assert dec.is_mixed
    assert dec.text == "你好\n世界\n"
    assert "浣犲ソ" not in dec.text
    assert "\ufffd" not in dec.text


def test_sparse_mixed_utf8_inside_predominantly_gb18030_file():
    raw = (
        ("这是一段正常的国标中文正文测试内容。\n" * 20).encode("gb18030")
        + "你好\n".encode("utf-8")
        + "这是国标结尾。\n".encode("gb18030")
    )
    dec = decode_bytes(raw)
    assert dec.is_mixed
    assert "你好" in dec.text
    assert "浣犲ソ" not in dec.text
    assert "\ufffd" not in dec.text


def test_mixed_encoding_chunk_fallback():
    chunk1 = ("第一章 测试\n" + "这是正常的简体中文测试内容，包含标点符号。\n" * 150).encode("utf-8")
    chunk2 = ("第二章 混合\n" + "这里是混合拼接的国标编码段落内容，应该能分别正确解码。\n" * 150).encode("gb18030")
    dec = decode_bytes(chunk1 + chunk2)
    assert dec.is_mixed
    assert "第一章 测试" in dec.text
    assert "第二章 混合" in dec.text
    assert "\ufffd" not in dec.text


def test_shadow_preserves_fullwidth_punctuation_and_exact_spans():
    orig = "“靈夢：测试ＡＢＣ１２３，怎么样？！”"
    sh = build_shadow(orig)
    assert sh.text == "“灵梦：测试ABC123，怎么样？！”"
    # Verify exact offset projection for '灵梦' -> '靈夢'
    pos = sh.text.find("灵梦")
    o_st, o_en = sh.to_orig_span(pos, pos + 2)
    assert orig[o_st:o_en] == "靈夢"
    # Fullwidth punctuation unchanged in shadow and original
    assert "：" in sh.text and "，" in sh.text and "？" in sh.text and "！" in sh.text


def test_optional_gb18030_vs_utf8_parity():
    gb_env = os.environ.get("NOVEL_WASH_PARITY_GB_FILE")
    u8_env = os.environ.get("NOVEL_WASH_PARITY_U8_FILE")
    if not (gb_env and u8_env):
        pytest.skip("Optional local GB18030/UTF-8 parity env vars not set")
    gb_path = Path(gb_env)
    u8_path = Path(u8_env)
    if not (gb_path.exists() and u8_path.exists()):
        pytest.skip("Optional local GB18030/UTF-8 parity fixtures not present")
    d_gb = read_novel_file(gb_path)
    d_u8 = read_novel_file(u8_path)
    assert d_gb.encoding == "gb18030"
    assert len(d_gb.lines) == len(d_u8.lines)
    assert d_gb.lines[0] == d_u8.lines[0]


def test_triage_clean_traps():
    corpus_dir = os.environ.get("NOVEL_WASH_CLEAN_CORPUS_DIR")
    if not corpus_dir or not Path(corpus_dir).is_dir():
        pytest.skip("Optional NOVEL_WASH_CLEAN_CORPUS_DIR not set")
    txts = sorted(Path(corpus_dir).glob("*.txt"))
    if not txts:
        pytest.skip("No clean corpus txt fixtures found")
    for p in txts[:5]:
        res = triage_file(p)
        assert res.grade == "clean"
        assert res.hard_cjk_lines == 0


def test_round4_mixed_encoding_cr_only_line_resolution():
    raw = (
        "测试国标。\r".encode("gb18030")
        + "你好\r".encode("utf-8")
        + "测试国标。\r".encode("gb18030")
    )
    dec = decode_bytes(raw)
    assert dec.is_mixed is True
    assert dec.lines == ["测试国标。", "你好", "测试国标。"]
    assert dec.line_ending == "\r"
    assert "浣犲ソ" not in dec.text
    assert "\ufffd" not in dec.text
