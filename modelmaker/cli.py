"""`modelmaker check-blocks [DIR]`: validate block files, check their
requirements and run every `_example_` self-check. Exits 1 on any failure,
so a team's library repo can run it in CI.

With no DIR it checks the built-ins and the user's block library
(~/.modelmaker/blocks, or MODELMAKER_BLOCK_LIBRARY)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import block_library as bl
from .blocks.base import BLOCK_REGISTRY


def _check_file(path: Path, taken: set[str], run: bool, out) -> int:
    source = path.read_text(encoding="utf-8")
    parsed = bl.validate(bl.parse_source(source, str(path)), taken)
    failures = 0
    print(f"{path}", file=out)
    for err in parsed.file_errors:
        print(f"  error: {err}", file=out)
        failures += 1
    examples = bl.run_examples(source, str(path)) if run and not parsed.file_errors else {}
    if "__file__" in examples:
        print(f"  error: {examples['__file__']['error']}", file=out)
        failures += 1
    for d in parsed.definitions:
        problems = list(parsed.errors.get(d.id, []))
        problems += [f"needs {req}" for req in bl.check_requirements(d)]
        ex = examples.get(d.id)
        if ex is not None and ex["ok"] is False:
            problems.append(f"example failed: {ex['error']}")
        mark = "ok" if not problems else "FAIL"
        note = " (example passed)" if ex is not None and ex["ok"] else ""
        print(f"  {mark:4} {d.id} v{d.version}{note}", file=out)
        for p in problems:
            print(f"       - {p}", file=out)
        failures += bool(problems)
        taken.add(d.id)
    return failures


def check_blocks(directory: Path | None, run_examples: bool = True, out=sys.stdout) -> int:
    """The number of failing files/blocks (0 means all good)."""
    failures = 0
    taken = set(BLOCK_REGISTRY)
    if directory is None:
        for bid, errs in bl.check_builtin().items():
            if errs:
                print(f"built-in {bid}: {'; '.join(errs)}", file=out)
                failures += 1
        print(f"{len(BLOCK_REGISTRY)} built-in blocks checked", file=out)
        from .library_store import library_dir

        directory = library_dir()
    paths = sorted(directory.glob("*.py")) + sorted((directory / "drafts").glob("*.py")) if directory.is_dir() else []
    if not paths:
        print(f"no block files in {directory}", file=out)
    for path in paths:
        failures += _check_file(path, taken, run_examples, out)
    return failures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="modelmaker")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-blocks", help="validate block files and run their examples")
    check.add_argument("directory", nargs="?", type=Path, help="a folder of block files (default: built-ins + your library)")
    check.add_argument("--no-examples", action="store_true", help="validate only; don't run _example_ functions")
    args = parser.parse_args(argv)
    if args.command == "check-blocks":
        failures = check_blocks(args.directory, run_examples=not args.no_examples)
        print("all good" if not failures else f"{failures} problem(s)")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
