"""Build gold benchmark dataset from a local corpus manifest and external spec file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from novel_wash.encoding import read_novel_file


def _load_book_map(manifest_path: Path | None) -> dict[str, Path]:
    path = manifest_path or (
        Path(os.environ["NOVEL_WASH_CORPUS_MANIFEST"]) if "NOVEL_WASH_CORPUS_MANIFEST" in os.environ else None
    )
    if path is None or not path.is_file():
        raise SystemExit(
            "Provide --manifest <corpus.json> (see config/corpus.example.json) or set NOVEL_WASH_CORPUS_MANIFEST."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["key"]): Path(item["path"]) for item in raw.get("books", []) if "key" in item and "path" in item}


def _write_synthetic_demo(out_path: Path) -> None:
    """Write a small synthetic, non-copyrighted gold demo file for offline M0 smoke checks."""
    demo_records = [
        {
            "id": "demo:L2",
            "book": "demo",
            "line_num": 2,
            "kind": "inline",
            "orig": "她走上三靈层高塔,准备观察地形。",
            "gold_clean": "她走上三层高塔,准备观察地形。",
            "notes": "synthetic watermark char + clean common CJK",
        },
        {
            "id": "demo:L3",
            "book": "demo",
            "line_num": 3,
            "kind": "hard_neg",
            "orig": "“两条道路都在前方等着我们，一步都不能出错。”",
            "gold_clean": "“两条道路都在前方等着我们，一步都不能出错。”",
            "notes": "clean hard negative",
        },
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in demo_records) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    bench_dir = Path(__file__).resolve().parent
    project_root = bench_dir.parent
    parser = argparse.ArgumentParser(description="Build gold benchmark JSONL from manifest + private specs.")
    parser.add_argument("--manifest", type=Path, default=None, help="JSON corpus manifest (see config/corpus.example.json)")
    parser.add_argument("--specs", type=Path, default=None, help="Private JSON list of {book, line_num, kind, gold_clean, ...}")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (defaults to output/benchmarks/synthetic_gold.jsonl for demo, benchmarks/gold_set.jsonl for --specs)",
    )
    parser.add_argument("--write-synthetic-demo", action="store_true", help="Emit synthetic non-copyrighted demo records")
    parser.add_argument("--force", action="store_true", help="Overwrite --out if it already exists")
    args = parser.parse_args(argv)

    out_path = args.out or (
        project_root / "output" / "benchmarks" / "synthetic_gold.jsonl"
        if args.write_synthetic_demo
        else bench_dir / "gold_set.jsonl"
    )
    if out_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing {out_path}; pass --force or a different --out.")

    if args.write_synthetic_demo:
        _write_synthetic_demo(out_path)
        print(f"Wrote synthetic demo gold records to {out_path}")
        return

    if args.specs is None or not args.specs.is_file():
        raise SystemExit("Provide --specs <gold_specs.json> (or pass --write-synthetic-demo).")

    bmap = _load_book_map(args.manifest)
    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    if not isinstance(specs, list):
        raise SystemExit("Specs file must be a JSON array.")

    lines_cache: dict[str, list[str]] = {}
    records: list[dict] = []
    for idx, sp in enumerate(specs):
        book = str(sp["book"])
        ln = int(sp["line_num"])
        if book not in bmap:
            raise SystemExit(f"Spec #{idx}: book key {book!r} missing from manifest.")
        book_path = bmap[book]
        if not book_path.is_file():
            raise SystemExit(f"Spec #{idx}: source file does not exist: {book_path}")
        if book not in lines_cache:
            lines_cache[book] = read_novel_file(book_path).lines
        book_lines = lines_cache[book]
        if not (1 <= ln <= len(book_lines)):
            raise SystemExit(f"Spec #{idx}: line_num {ln} out of bounds for {book} (1..{len(book_lines)}).")
        orig_line = book_lines[ln - 1]
        expected_orig = sp.get("expected_orig")
        if expected_orig is not None and orig_line != expected_orig:
            raise SystemExit(
                f"Spec #{idx}: source mismatch at {book}:L{ln}; file does not match expected edition."
            )
        records.append(
            {
                "id": f"{book}:L{ln}",
                "book": book,
                "line_num": ln,
                "kind": str(sp["kind"]),
                "orig": orig_line,
                "gold_clean": str(sp["gold_clean"]),
                "notes": str(sp.get("notes", "")),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} gold records to {out_path}")


if __name__ == "__main__":
    main()
