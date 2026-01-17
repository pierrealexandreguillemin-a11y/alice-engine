#!/usr/bin/env python3
"""Pipeline Gate - Enforces ISO pipeline order via filesystem state.

Document ID: ALICE-HOOK-GATE-001
Version: 1.0.0
ISO: 5055, 42001 (tracabilite)

This hook checks ACTUAL filesystem state, not command patterns.
Cannot be bypassed by renaming commands.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

STATE_FILE = Path(".claude/hooks/pipeline_state.json")


def _check_exists(cwd: Path, pattern: str) -> bool:
    """Check if file/pattern exists."""
    if "*" in pattern:
        parts = pattern.split("/")
        search_path = cwd
        for part in parts[:-1]:
            if "*" in part:
                matches = list(search_path.glob(part))
                if not matches:
                    return False
                search_path = matches[0]
            else:
                search_path = search_path / part
        return bool(list(search_path.glob(parts[-1])))
    return (cwd / pattern).exists()


def _check_json_condition(cwd: Path, check: str) -> bool:
    """Check JSON file condition like 'file.json EXISTS AND field >= value'."""
    if " AND " not in check:
        pattern = check.replace(" EXISTS", "").strip()
        return _check_exists(cwd, pattern)

    parts = check.split(" AND ")
    file_check = parts[0].replace(" EXISTS", "").strip()
    if not _check_exists(cwd, file_check):
        return False

    # Parse condition like "field >= 0.95"
    condition = parts[1].strip()
    try:
        file_path = cwd / file_check
        data = json.loads(file_path.read_text())
        # Simple parser for "field >= value"
        for op in [">=", "<=", ">", "<", "=="]:
            if op in condition:
                field, value = condition.split(op)
                field = field.strip()
                value = float(value.strip())
                actual = data.get(field, 0)
                if op == ">=":
                    return actual >= value
                elif op == "<=":
                    return actual <= value
                elif op == ">":
                    return actual > value
                elif op == "<":
                    return actual < value
                elif op == "==":
                    return actual == value
    except (json.JSONDecodeError, KeyError, ValueError):
        return False
    return True


def _load_state(cwd: Path) -> dict:
    """Load pipeline state."""
    state_path = cwd / STATE_FILE
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"steps": {}}


def _save_state(cwd: Path, state: dict) -> None:
    """Save pipeline state."""
    state_path = cwd / STATE_FILE
    state_path.write_text(json.dumps(state, indent=2))


def _update_state(cwd: Path) -> tuple[list[str], list[str]]:
    """Update state based on actual filesystem. Return (completed, blocked)."""
    state = _load_state(cwd)
    completed = []
    blocked = []

    for step_id, step in state.get("steps", {}).items():
        check = step.get("check", "")
        is_done = _check_json_condition(cwd, check)

        # Check prerequisites
        prereqs_met = True
        for req in step.get("requires", []):
            req_step = state["steps"].get(req, {})
            if not req_step.get("completed", False):
                req_check = req_step.get("check", "")
                if not _check_json_condition(cwd, req_check):
                    prereqs_met = False
                    blocked.append(f"{step['name']} blocked by {req_step.get('name', req)}")

        if is_done and prereqs_met:
            if not step.get("completed"):
                step["completed"] = True
                step["timestamp"] = datetime.now().isoformat()
                completed.append(step["name"])
        elif is_done and not prereqs_met:
            blocked.append(f"{step['name']}: output exists but prerequisites missing!")

    _save_state(cwd, state)
    return completed, blocked


def main() -> None:
    """Gate check - verify pipeline state before/after commands."""
    data = json.load(sys.stdin)
    cwd = Path(data.get("cwd", "."))

    completed, blocked = _update_state(cwd)

    if blocked:
        output = {
            "hookSpecificOutput": {
                "hookEventName": data.get("hook_event_name", "Unknown"),
                "additionalContext": (
                    "PIPELINE STATE:\n"
                    + (f"Completed: {', '.join(completed)}\n" if completed else "")
                    + "Blocked:\n"
                    + "\n".join(f"  - {b}" for b in blocked)
                ),
            }
        }
        print(json.dumps(output))
    elif completed:
        output = {
            "hookSpecificOutput": {
                "hookEventName": data.get("hook_event_name", "Unknown"),
                "additionalContext": f"STEP COMPLETED: {', '.join(completed)}",
            }
        }
        print(json.dumps(output))

    sys.exit(0)


if __name__ == "__main__":
    main()
