"""Thin wrapper over the system `git` CLI, scoped to a project folder --
backs the Toolbar's Git panel (status/commit/push/pull/remote) so a project
folder (see project.ensure_project_scaffold) can be versioned and pushed to
GitHub without the app needing its own git implementation or GitHub
credentials of its own: pushing/pulling uses whatever git already has
configured on the machine (SSH keys, credential helper, ...).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class GitError(RuntimeError):
    """A git operation failed -- message is git's own stderr/stdout."""


def _run(args: list[str], cwd: Path, check: bool = True) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError as e:
        raise GitError("git is not installed, or not on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise GitError(f"git {' '.join(args)} timed out") from e
    if check and result.returncode != 0:
        raise GitError(result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def is_repo(project_dir: Path) -> bool:
    return (project_dir / ".git").is_dir()


def init(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    if not is_repo(project_dir):
        _run(["init"], cwd=project_dir)


def current_branch(project_dir: Path) -> str | None:
    out = _run(["rev-parse", "--abbrev-ref", "HEAD"], cwd=project_dir, check=False).strip()
    return out or None


def remote_url(project_dir: Path, name: str = "origin") -> str | None:
    return _run(["remote", "get-url", name], cwd=project_dir, check=False).strip() or None


def set_remote(project_dir: Path, url: str, name: str = "origin") -> None:
    if remote_url(project_dir, name):
        _run(["remote", "set-url", name, url], cwd=project_dir)
    else:
        _run(["remote", "add", name, url], cwd=project_dir)


def status(project_dir: Path) -> dict[str, Any]:
    """A snapshot of the project folder's git state for the Git panel: the
    current branch, its remote (if any), and every uncommitted change
    (porcelain status), plus how far the local branch is ahead/behind its
    upstream (0/0 when there is none to compare against)."""
    if not is_repo(project_dir):
        return {"is_repo": False, "branch": None, "remote": None, "changes": [], "ahead": 0, "behind": 0}

    branch = current_branch(project_dir)
    porcelain = _run(["status", "--porcelain=v1"], cwd=project_dir, check=False)
    changes = [{"status": line[:2].strip(), "path": line[3:]} for line in porcelain.splitlines() if line]

    ahead = behind = 0
    if branch:
        counts = _run(
            ["rev-list", "--left-right", "--count", f"{branch}...{branch}@{{u}}"], cwd=project_dir, check=False
        ).strip()
        parts = counts.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            ahead, behind = int(parts[0]), int(parts[1])

    return {
        "is_repo": True,
        "branch": branch,
        "remote": remote_url(project_dir),
        "changes": changes,
        "ahead": ahead,
        "behind": behind,
    }


def commit(project_dir: Path, message: str) -> None:
    _run(["add", "-A"], cwd=project_dir)
    _run(["commit", "-m", message], cwd=project_dir)


def push(project_dir: Path) -> None:
    branch = current_branch(project_dir) or "HEAD"
    _run(["push", "-u", "origin", branch], cwd=project_dir)


def pull(project_dir: Path) -> None:
    _run(["pull"], cwd=project_dir)
