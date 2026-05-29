# Global Agent Rules

## Language

Default to Chinese in user-facing replies unless the user explicitly requests another language.

## Response Style

Do not propose follow-up tasks or enhancements at the end of the final answer.

## Debug-First Policy

- Do not introduce silent fallbacks, mock success paths, hidden caps, or swallowed errors just to make the system run.
- Failures must surface through explicit errors, logs, tests, API payloads, or UI state.
- If a boundary rule or fallback is truly necessary, it must be explicit, documented, easy to disable, and agreed before implementation.
- In this project, real trading must stay disabled unless the user explicitly asks to enable it.

## Engineering Quality Baseline

- Follow SOLID, DRY, separation of concerns, and YAGNI.
- Prefer clear naming and small, testable units.
- Remove dead code and obsolete compatibility paths when changing behavior, unless compatibility is explicitly required.
- Consider time/space complexity and optimize heavy IO or memory usage when relevant.
- Handle edge cases explicitly; do not hide failures.

## Code Metrics

- Function length: max 50 nonblank lines.
- File size: max 300 lines.
- Nesting depth: max 3.
- Positional parameters: max 3, excluding `self` and `cls`.
- Cyclomatic complexity: max 10.
- Magic numbers should be extracted to named constants.

## Decoupling & Immutability

- Business logic should receive concrete dependencies through parameters or small interfaces.
- Prefer immutable inputs and return new values instead of mutating parameters or global state.

## Security Baseline

- Never hardcode secrets, API keys, or credentials in source code.
- Use parameterized queries for database access.
- Validate and sanitize external input at system boundaries.
- If an API key is only shared in conversation for normal setup/debugging, do not warn about a code leak unless it is written into source files.

## Validation

- Backend tests must run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Backend test commands should use a 60 second timeout for targeted runs.
- The project quality gate is `backend/scripts/agents_quality_gate.py`.
- `backend/check_backend.ps1` runs the AGENTS quality gate before pytest.

## Skills

- Skills live in `~/.codex/skills/` and optional project-shared `.codex/skills/`.
- Before starting a task, scan available skills, read matching `SKILL.md`, announce used skills, and follow them.
- Prefer `taskmaster` for tasks with 3+ ordered steps that produce file changes.
