from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MAX_FUNCTION_LINES = 50
MAX_FILE_LINES = 300
MAX_NESTING_DEPTH = 3
MAX_POSITIONAL_PARAMS = 3
MAX_COMPLEXITY = 10
BASELINE_PATH = Path("backend/docs/agents_quality_baseline.json")
SCAN_DIRS = (Path("app"), Path("scripts"))
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".tmp", "runtime", "reports", "models"}
PY_EXTENSIONS = {".py"}
FRONTEND_EXTENSIONS = {".js", ".jsx", ".css"}
EMPTY_EXCEPT_PATTERN = re.compile(r"except\s+[^:\n]*:\s*(?:\n\s*)?pass\b")


@dataclass(frozen=True)
class Violation:
    code: str
    path: str
    line: int
    detail: str

    @property
    def key(self) -> str:
        return f"{self.code}|{self.path}|{self.line}|{self.detail}"


def main() -> int:
    args = _parse_args()
    root = Path(args.root).resolve()
    violations = scan_project(root, include_frontend=args.include_frontend)
    if args.update_baseline:
        write_baseline(root / args.baseline, violations)
        print(f"AGENTS quality baseline updated: {len(violations)} violations")
        return 0
    baseline = read_baseline(root / args.baseline)
    new_violations = [item for item in violations if item.key not in baseline]
    print_report(violations, new_violations)
    return 1 if new_violations else 0


def scan_project(root: Path, *, include_frontend: bool = False) -> list[Violation]:
    paths = list(_python_paths(root / "backend"))
    if include_frontend:
        paths.extend(_frontend_paths(root / "frontend" / "src"))
    violations: list[Violation] = []
    for path in paths:
        violations.extend(scan_file(path, root))
    return sorted(violations, key=lambda item: item.key)


def scan_file(path: Path, root: Path) -> list[Violation]:
    rel = _relative_path(path, root)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    violations = [_file_size_violation(rel, len(lines))]
    if path.suffix in PY_EXTENSIONS:
        violations.extend(scan_python(path, rel, text))
    violations.extend(_empty_except_violations(rel, lines))
    return [item for item in violations if item is not None]


def scan_python(path: Path, rel: str, text: str) -> list[Violation]:
    tree = ast.parse(text, filename=str(path))
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_function_violations(rel, node))
        elif isinstance(node, ast.ExceptHandler) and _handler_passes_silently(node):
            violations.append(Violation("EMPTY_EXCEPT_PASS", rel, int(node.lineno), "except block passes silently"))
    return violations


def _function_violations(rel: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Violation]:
    length = int(getattr(node, "end_lineno", node.lineno)) - int(node.lineno) + 1
    params = _positional_param_count(node)
    complexity = _complexity(node)
    nesting = _nesting_depth(node)
    checks = (
        ("FUNCTION_LENGTH", length, MAX_FUNCTION_LINES, f"{node.name} has {length} lines"),
        ("POSITIONAL_PARAMS", params, MAX_POSITIONAL_PARAMS, f"{node.name} has {params} positional params"),
        ("COMPLEXITY", complexity, MAX_COMPLEXITY, f"{node.name} complexity {complexity}"),
        ("NESTING_DEPTH", nesting, MAX_NESTING_DEPTH, f"{node.name} nesting depth {nesting}"),
    )
    return [
        Violation(code, rel, int(node.lineno), detail)
        for code, actual, limit, detail in checks
        if actual > limit
    ]


def _file_size_violation(rel: str, line_count: int) -> Violation | None:
    if line_count <= MAX_FILE_LINES:
        return None
    return Violation("FILE_SIZE", rel, 1, f"file has {line_count} lines")


def _empty_except_violations(rel: str, lines: list[str]) -> list[Violation]:
    return [
        Violation("EMPTY_EXCEPT_PASS", rel, index, "except block passes silently")
        for index, line in enumerate(lines, start=1)
        if EMPTY_EXCEPT_PATTERN.search(line)
    ]


def _handler_passes_silently(node: ast.ExceptHandler) -> bool:
    return bool(node.body) and all(isinstance(child, ast.Pass) for child in node.body)


def _positional_param_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    names = [arg.arg for arg in node.args.posonlyargs + node.args.args]
    return len([name for name in names if name not in {"self", "cls"}])


def _complexity(node: ast.AST) -> int:
    count = 1
    branch_nodes = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ExceptHandler,
        ast.IfExp,
        ast.Match,
        ast.Assert,
    )
    for child in ast.walk(node):
        if isinstance(child, branch_nodes):
            count += 1
        elif isinstance(child, ast.BoolOp):
            count += max(len(child.values) - 1, 0)
        elif isinstance(child, ast.comprehension):
            count += 1 + len(child.ifs)
    return count


def _nesting_depth(node: ast.AST) -> int:
    return max((_nested_depth(child, 0) for child in ast.iter_child_nodes(node)), default=0)


def _nested_depth(node: ast.AST, depth: int) -> int:
    next_depth = depth + 1 if isinstance(node, _nesting_nodes()) else depth
    return max([next_depth, *(_nested_depth(child, next_depth) for child in ast.iter_child_nodes(node))])


def _nesting_nodes() -> tuple[type[ast.AST], ...]:
    return (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)


def read_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("violations") or {}
    if isinstance(records, list):
        return set(str(item) for item in records)
    return set(str(key) for key in records.keys())


def write_baseline(path: Path, violations: Iterable[Violation]) -> None:
    payload = {
        "version": 1,
        "description": "Existing AGENTS quality debt baseline. The quality gate fails on new or worsened violations.",
        "violations": {item.key: item.detail for item in sorted(violations, key=lambda row: row.key)},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_report(violations: list[Violation], new_violations: list[Violation]) -> None:
    print(f"AGENTS quality gate: total={len(violations)} new={len(new_violations)}")
    for item in new_violations[:50]:
        print(f"NEW {item.code} {item.path}:{item.line} {item.detail}")
    if len(new_violations) > 50:
        print(f"... {len(new_violations) - 50} more new violations")


def _python_paths(backend_root: Path) -> Iterable[Path]:
    for directory in SCAN_DIRS:
        yield from _paths(backend_root / directory, PY_EXTENSIONS)


def _frontend_paths(frontend_src: Path) -> Iterable[Path]:
    yield from _paths(frontend_src, FRONTEND_EXTENSIONS)


def _paths(root: Path, extensions: set[str]) -> Iterable[Path]:
    if not root.exists():
        return []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in extensions and not _is_skipped(path, root):
            yield path


def _is_skipped(path: Path, root: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(root).parts)


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGENTS quality gate")
    parser.add_argument("--root", default="..", help="project root from backend working directory")
    parser.add_argument("--baseline", default=str(BASELINE_PATH), help="baseline path relative to backend")
    parser.add_argument("--include-frontend", action="store_true", help="also scan frontend/src")
    parser.add_argument("--update-baseline", action="store_true", help="write current violations as explicit baseline")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
