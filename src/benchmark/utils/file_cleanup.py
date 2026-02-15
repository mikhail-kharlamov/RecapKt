from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeleteResult:
    scanned: int
    matched: int
    deleted: int


def delete_files_by_pattern(
    *,
    root_dir: Path | str,
    extension: str,
    name_contains: str,
    dry_run: bool = True,
) -> DeleteResult:
    """Delete files recursively under `root_dir` that match extension and substring.

    Args:
        root_dir: Root directory to scan recursively.
        extension: File extension to match (e.g. `.json`). Case-insensitive.
        name_contains: Substring that must be present in the filename (e.g. `tokens`). Case-insensitive.
        dry_run: If True, only prints the files that would be deleted.

    Returns:
        DeleteResult: basic counters.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    ext = extension.lower().strip()
    if ext and not ext.startswith("."):
        ext = f".{ext}"

    needle = name_contains.lower()

    scanned = 0
    matched = 0
    deleted = 0

    # Use rglob('*') to avoid depending on the extension being well-formed.
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        scanned += 1

        filename_lower = path.name.lower()
        if needle not in filename_lower:
            continue

        if ext and path.suffix.lower() != ext:
            continue

        matched += 1

        if dry_run:
            print(f"[DRY-RUN] Would delete: {path}")
            continue

        path.unlink()
        deleted += 1
        print(f"Deleted: {path}")

    return DeleteResult(scanned=scanned, matched=matched, deleted=deleted)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Delete benchmark artifact files by extension and filename substring")
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to scan recursively (e.g. src/benchmark/tool_plan_benchmarking/logs/memory)",
    )
    parser.add_argument(
        "--ext",
        default="",
        help="Extension to match (e.g. .json). If empty, matches any extension.",
    )
    parser.add_argument(
        "--contains",
        required=True,
        help="Substring that must be present in filename (case-insensitive), e.g. tokens",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete; only print what would be deleted.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    result = delete_files_by_pattern(
        root_dir=args.root,
        extension=args.ext,
        name_contains=args.contains,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(
            "Summary (dry-run): scanned=%s matched=%s",
            result.scanned,
            result.matched,
        )
    else:
        print(
            "Summary: scanned=%s matched=%s deleted=%s",
            result.scanned,
            result.matched,
            result.deleted,
        )


if __name__ == "__main__":
    main()
