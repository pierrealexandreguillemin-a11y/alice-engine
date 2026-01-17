#!/usr/bin/env python3
"""Inject pipeline status into Claude's context.

Document ID: ALICE-HOOK-INJECT-001
Version: 1.0.0
ISO: 42001 (tracabilite)

This hook outputs pipeline status that gets injected into Claude's context.
Claude CANNOT ignore this because it's in the system prompt.
"""

from __future__ import annotations

import json
import sys
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


def _get_status(cwd: Path) -> str:
    """Get pipeline status string."""
    state_path = cwd / STATE_FILE
    if not state_path.exists():
        return "PIPELINE STATE: Not initialized"

    state = json.loads(state_path.read_text())
    lines = ["## ISO PIPELINE STATUS (MANDATORY)"]

    pending = []
    done = []
    next_step = None

    for step_id, step in state.get("steps", {}).items():
        check = step.get("check", "").split(" AND ")[0].replace(" EXISTS", "").strip()
        is_done = _check_exists(cwd, check) if check else False

        if is_done:
            done.append(f"[x] {step['name']} ({step['iso']})")
        else:
            pending.append(f"[ ] {step['name']} ({step['iso']})")
            if next_step is None:
                # Check if prerequisites are met
                prereqs_met = True
                for req in step.get("requires", []):
                    req_step = state["steps"].get(req, {})
                    req_check = req_step.get("check", "").split(" AND ")[0].replace(" EXISTS", "").strip()
                    if not _check_exists(cwd, req_check):
                        prereqs_met = False
                        break
                if prereqs_met:
                    next_step = step["name"]

    lines.extend(done)
    lines.extend(pending)

    if next_step:
        lines.append(f"\n**NEXT REQUIRED STEP: {next_step}**")
        lines.append("You MUST complete this step before proceeding.")
    elif pending:
        lines.append("\n**BLOCKED: Prerequisites not met for remaining steps.**")
    else:
        lines.append("\n**PIPELINE COMPLETE: All ISO steps validated.**")

    return "\n".join(lines)


def main() -> None:
    """Output pipeline status for injection into Claude's context."""
    data = json.load(sys.stdin)
    cwd = Path(data.get("cwd", "."))

    status = _get_status(cwd)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": status,
        }
    }
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
