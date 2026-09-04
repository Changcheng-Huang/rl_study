#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.algorithm_registry.package import iter_package_files, validate_source_directory


def build_archive(source: Path, output_directory: Path) -> Path:
    source = source.resolve()
    output_directory = output_directory.resolve()
    report = validate_source_directory(source)
    if not report.valid or report.manifest is None:
        for issue in report.issues:
            print(f"{issue.level.upper()} [{issue.code}] {issue.message}", file=sys.stderr)
        raise ValueError("algorithm package source failed validation")

    for warning in report.warnings:
        print(f"WARNING [{warning.code}] {warning.message}", file=sys.stderr)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_directory
        / f"{report.manifest.algorithm_id}-{report.manifest.version}.zip"
    )
    with zipfile.ZipFile(
        output_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for file_path in iter_package_files(source):
            archive.write(file_path, file_path.relative_to(source).as_posix())
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an algorithm teaching package directory and build a ZIP."
    )
    parser.add_argument("source", type=Path, help="package source directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist",
        help="output directory (default: project dist/)",
    )
    args = parser.parse_args()

    try:
        output_path = build_archive(args.source, args.output)
    except ValueError:
        return 1
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
