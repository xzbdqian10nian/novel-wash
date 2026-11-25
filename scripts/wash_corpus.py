"""Portable manifest-driven batch scanner & washer for novel-wash."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novel_wash.cli import wash_book
from novel_wash.encoding import read_novel_file
from novel_wash.epub_fix import check_and_repair_epub
from novel_wash.group_soup import detect_soup_and_ads
from novel_wash.inline_rules import analyze_inline_pollution
from novel_wash.llm_client import LLMRouter
from novel_wash.triage import triage_file


@dataclass(frozen=True)
class BookEntry:
    key: str
    category: str
    path: Path
    use_model: bool = True
    harmony_llm: bool = False
    group: str | None = None


def _load_entries(
    *,
    manifest_path: Path | None,
    book_paths: list[Path],
    epub_paths: list[Path],
) -> tuple[list[BookEntry], list[Path]]:
    books: list[BookEntry] = []
    epubs: list[Path] = []
    if manifest_path is not None:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for idx, item in enumerate(raw.get("books", [])):
            p = Path(item["path"])
            books.append(
                BookEntry(
                    key=str(item.get("key") or p.stem or f"book_{idx}"),
                    category=str(item.get("category") or "corpus"),
                    path=p,
                    use_model=bool(item.get("use_model", True)),
                    harmony_llm=bool(item.get("harmony_llm", False)),
                    group=item.get("group"),
                )
            )
        epubs.extend(Path(x) for x in raw.get("epubs", []))
    for p in book_paths:
        books.append(BookEntry(key=p.stem, category="cli", path=p))
    epubs.extend(epub_paths)
    return books, epubs


def dry_run_scan(books: list[BookEntry], epubs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in books:
        tr = triage_file(entry.path)
        dec = read_novel_file(entry.path)
        se, _, _, sn = detect_soup_and_ads(dec.lines)
        ir = analyze_inline_pollution(dec.lines, se, sn)
        cleared = sum(1 for e in se if e.op == "clear_line")
        row = {
            "category": entry.category,
            "key": entry.key,
            "file": entry.path.name,
            "mb": tr.mb,
            "encoding": tr.encoding,
            "lines": tr.lines,
            "grade": tr.grade,
            "pua_unique": tr.pua_unique,
            "pua_count": tr.pua_count,
            "soup_cleared": cleared,
            "rule_pre_edits": len(ir.pre_edits),
            "llm_tasks": len(ir.tasks),
            "harmony_slots": tr.harmony_slots,
        }
        rows.append(row)
        print(
            f"[{entry.category:9s}] {entry.key:11s} | {tr.grade:16s} | enc={tr.encoding:9s} | "
            f"lines={tr.lines:6d} | pua={tr.pua_unique:3d}/{tr.pua_count:5d} | "
            f"soup={cleared:4d} | pre={len(ir.pre_edits):5d} | llm_tasks={len(ir.tasks):4d} | "
            f"harmony={tr.harmony_slots:3d}"
        )
    for ep in epubs:
        res = check_and_repair_epub(ep, out_dir=None)
        row = {
            "category": "epub",
            "key": ep.stem,
            "file": ep.name,
            "mb": res.mb,
            "total_entries": res.total_entries,
            "xml_entries": res.xml_entries,
            "bad_xml": len(res.bad_xml_entries),
        }
        rows.append(row)
        print(
            f"[epub     ] {ep.name} | entries={res.total_entries} xml={res.xml_entries} bad_xml={len(res.bad_xml_entries)}"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Portable batch washer using a JSON manifest or explicit file paths.")
    parser.add_argument("--manifest", type=Path, default=None, help="JSON corpus manifest (see config/corpus.example.json)")
    parser.add_argument("--books", nargs="*", type=Path, default=[], help="TXT novel paths")
    parser.add_argument("--epubs", nargs="*", type=Path, default=[], help="EPUB paths to check/repair")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", help="Filter by key substring")
    parser.add_argument("--primary", default=None)
    parser.add_argument("--secondary", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--use-pools", action="store_true")
    parser.add_argument("--strip-telegram", action="store_true")
    args = parser.parse_args(argv)

    books, epubs = _load_entries(manifest_path=args.manifest, book_paths=args.books, epub_paths=args.epubs)
    if not books and not epubs:
        print(
            "Error: provide --manifest <corpus.json> (see config/corpus.example.json) or --books/--epubs paths.",
            file=sys.stderr,
        )
        return 2

    project_root = Path(__file__).resolve().parent.parent
    out_dir = args.out or (project_root / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        rows = dry_run_scan(books, epubs)
        (out_dir / "dry_run_scan.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    router = LLMRouter(project_root) if any(b.use_model for b in books) else None
    primary = args.primary or (router.default_primary if router is not None else "deepseek-v4.1-flash-1017")
    if args.secondary is None:
        secondary: str | None = router.default_secondary if router is not None else None
    elif args.secondary.strip().lower() in ("", "none", "off", "null"):
        secondary = None
    else:
        secondary = args.secondary

    summary: dict[str, Any] = {
        "primary_model": primary,
        "secondary_model": secondary,
        "books": [],
        "epubs": [],
    }

    def match(key: str) -> bool:
        return not args.only or any(o in key for o in args.only)

    try:
        print(f"Running corpus wash with primary={primary}, secondary={secondary}", flush=True)
        for entry in books:
            if not match(entry.key) and not (args.only and entry.category in args.only):
                continue
            print(f"\n>>> [{entry.category}] {entry.key} (provider={entry.group or 'default'}): {entry.path.name}", flush=True)
            r = wash_book(
                entry.path,
                project_root=project_root,
                out_dir=out_dir,
                use_model=entry.use_model,
                primary_model=primary,
                secondary_model=secondary,
                group_override=entry.group,
                render_epub=True,
                restore_harmony_with_llm=entry.harmony_llm,
                use_pools=args.use_pools,
                router=router if entry.use_model else None,
            )
            r["key"] = entry.key
            r["category"] = entry.category
            summary["books"].append(r)
            print(f"    done: {json.dumps(r['stats'], ensure_ascii=False)}", flush=True)

        if not args.only or "epub" in args.only:
            for ep in epubs:
                print(f"\n>>> [epub_fix] {ep.name}", flush=True)
                er = check_and_repair_epub(ep, out_dir / "epub_fixed", strip_telegram_banner=args.strip_telegram)
                summary["epubs"].append(er.to_dict())
                print(f"    done: bad={er.bad_xml_entries}, remaining={er.remaining_bad_entries}", flush=True)

        summary["ledger"] = router.ledger.summary() if router is not None else {"calls": 0, "fallbacks": 0, "total_cost_cny": 0.0, "by_route": {}}
        out_name = "corpus_summary.json" if not args.only else f"corpus_summary_{'_'.join(args.only)}.json"
        (out_dir / out_name).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        total_cost = router.ledger.total_cost_cny() if router is not None else 0.0
        print(f"\nTotal ledger cost: ¥{total_cost:.6f}", flush=True)
    finally:
        if router is not None:
            router.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
