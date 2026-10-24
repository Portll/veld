#!/usr/bin/env python3
"""Check that each optional integration imports cleanly at both ends of its declared range.

The adapters in python/veld/integrations/ import their third-party packages inside
try/except ImportError, so a renamed or removed name fails quietly at runtime instead of at
import: langchain-core 1.0 removed langchain_core.memory and nothing noticed. This script
installs nothing itself; CI installs one extra's requirements into a fresh environment and
then runs `check` on that extra's adapter.

  python_extras.py extras                         names of the extras with an adapter
  python_extras.py requirements EXTRA floor|latest   pip requirement lines for one extra
  python_extras.py requirements-file PATH floor|latest
  python_extras.py check FILE...                  import every third-party name the files use
"""

import ast
import importlib
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTEGRATIONS = ROOT / "python" / "veld" / "integrations"
LOCAL_PACKAGES = {"veld"}


def optional_dependencies() -> dict[str, list[str]]:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


def adapter_for(extra: str) -> Path:
    return INTEGRATIONS / f"{extra.replace('-', '_')}.py"


def to_floor(spec: str) -> str:
    match = re.match(r"^\s*([A-Za-z0-9_.\-\[\]]+)\s*>=\s*([^,;\s]+)", spec)
    if not match:
        raise SystemExit(f"no >= floor to pin in requirement {spec!r}")
    return f"{match.group(1)}=={match.group(2)}"


def emit(specs: list[str], mode: str) -> None:
    if mode not in ("floor", "latest"):
        raise SystemExit(f"mode must be floor or latest, not {mode!r}")
    for spec in specs:
        print(to_floor(spec) if mode == "floor" else spec)


def requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def imported_names(path: Path) -> list[tuple[str, str | None]]:
    """(module, attribute) pairs for every absolute import in the file, at any depth."""
    pairs: list[tuple[str, str | None]] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            pairs.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            pairs.extend((node.module, alias.name) for alias in node.names)
    return pairs


def is_third_party(module: str, source: Path) -> bool:
    top = module.split(".")[0]
    sibling = (source.parent / f"{top}.py").exists() or (source.parent / top).is_dir()
    return (
        top not in sys.stdlib_module_names
        and top not in LOCAL_PACKAGES
        and top != "__future__"
        and not sibling
    )


def check(paths: list[Path]) -> int:
    failures = []
    checked = 0
    for path in paths:
        for module, attribute in imported_names(path):
            if not is_third_party(module, path):
                continue
            checked += 1
            label = f"{module}.{attribute}" if attribute else module
            try:
                loaded = importlib.import_module(module)
                if attribute and attribute != "*" and not hasattr(loaded, attribute):
                    importlib.import_module(f"{module}.{attribute}")
            except Exception as error:  # noqa: BLE001 - any failure to resolve the name is the finding
                failures.append(f"{path.relative_to(ROOT)}: {label}: {type(error).__name__}: {error}")
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"{checked} third-party imports checked, {len(failures)} failed")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    command, args = argv[1], argv[2:]
    if command == "extras":
        for extra in optional_dependencies():
            if adapter_for(extra).exists():
                print(extra)
        return 0
    if command == "requirements":
        extra, mode = args
        emit(optional_dependencies()[extra], mode)
        return 0
    if command == "requirements-file":
        path, mode = args
        emit(requirement_lines(ROOT / path), mode)
        return 0
    if command == "check":
        return check([ROOT / p if not Path(p).is_absolute() else Path(p) for p in args])
    raise SystemExit(f"unknown command {command!r}\n{__doc__}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
