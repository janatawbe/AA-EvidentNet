"""Git repository introspection for experiment provenance.

All functions degrade gracefully (returning None / a status string) when run
outside a git repository, since the project may be exercised before version
control is initialized.
"""

import subprocess
from pathlib import Path
from typing import Optional


def _run_git(args, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    """Return the current commit SHA, or None if not in a git repo."""
    return _run_git(["rev-parse", "HEAD"], cwd=cwd)


def get_git_status_summary(cwd: Optional[Path] = None) -> str:
    """Return a short human-readable git status summary.

    Returns "not_a_git_repository" if cwd is not inside a git repo,
    "clean" if there are no uncommitted changes, or "dirty" otherwise.
    """
    commit = get_git_commit(cwd=cwd)
    if commit is None:
        return "not_a_git_repository"

    status = _run_git(["status", "--porcelain"], cwd=cwd)
    if status is None:
        return "unknown"
    return "clean" if status == "" else "dirty"
