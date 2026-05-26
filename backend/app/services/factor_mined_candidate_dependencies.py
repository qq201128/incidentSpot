from __future__ import annotations

from typing import Any


def dependency_rows(target_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(row.get("factorName")): row for row in source_rows}
    selected: dict[str, dict[str, Any]] = {}
    for row in target_rows:
        collect_dependencies(row, by_name, selected, set())
    return list(selected.values())


def target_and_dependency_rows(
    target_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = {str(row.get("factorName")): row for row in dependency_rows(target_rows, source_rows)}
    for row in target_rows:
        selected[str(row.get("factorName"))] = row
    return list(selected.values())


def collect_dependencies(
    row: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    visiting: set[str],
) -> None:
    for member in members(row):
        name = str(member["name"])
        dependency = by_name.get(name)
        if dependency is None or name in selected:
            continue
        if name in visiting:
            raise ValueError(f"cycle in mined factor library: {name}")
        visiting.add(name)
        collect_dependencies(dependency, by_name, selected, visiting)
        visiting.remove(name)
        selected[name] = dependency


def members(row: dict[str, Any]) -> list[dict[str, Any]]:
    source = row.get("members")
    if not isinstance(source, list) or not source:
        raise ValueError(f"mined factor missing members: {row.get('factorName')}")
    return [dict(member) for member in source]
