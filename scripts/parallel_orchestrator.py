#!/usr/bin/env python3
"""
Parallel Orchestrator v2.3

Orchestrates parallel Claude Code agent runs in isolated git worktrees.
Reads a YAML/JSON configuration file and executes each task in its own
worktree, with an optional reviewer pass after the implementer.

Usage:
    python scripts/parallel_orchestrator.py config.json
    python scripts/parallel_orchestrator.py config.yaml
    python scripts/parallel_orchestrator.py --help

Safety guarantees (hard constraints, no opt-out):
    - Never merges branches.
    - Never deletes worktrees.
    - Never commits changes.
    - Never pushes to remote.
    - Never resolves conflicts.
    - Only touches git worktrees, never the main working tree.
    - Validates clean git status on the main repo before starting.

Configuration (JSON):
    {
      "run_id": "v2.3-exp-001",
      "base_branch": "main",
      "max_budget_usd": 5.0,
      "review_after_implement": true,
      "tasks": [
        {
          "name": "add_auth",
          "branch": "feature/auth",
          "worktree": "../worktrees/auth",
          "agent": "implementer",
          "prompt": "Add user authentication to the app",
          "review_prompt": "Review the authentication implementation"
        }
      ]
    }

    - run_id:               Unique identifier for this orchestrator run (used for
                            log directory naming).
    - base_branch:          Git branch to base all worktrees on.
    - max_budget_usd:       Maximum USD budget per claude agent invocation.
    - review_after_implement: Whether to run reviewer agent after implementer.
    - tasks:                List of task definitions (see below).

    Per-task fields:
    - name:                 Short, unique task name (used in log filenames).
    - branch:               Git branch name to create for this task.
    - worktree:             Filesystem path for the worktree (absolute, or
                            relative to the repository root).
    - agent:                Claude Code agent name (typically "implementer").
    - prompt:               The prompt passed to the implementer agent.
    - review_prompt:        The prompt passed to the reviewer agent. Only used
                            when review_after_implement is true. Optional;
                            defaults to a generic review message.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional YAML support  (JSON is always supported)
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """Single task definition from the config file."""

    name: str
    branch: str
    worktree: str
    agent: str
    prompt: str
    review_prompt: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "TaskConfig":
        return cls(
            name=d["name"],
            branch=d["branch"],
            worktree=d["worktree"],
            agent=d.get("agent", "implementer"),
            prompt=d["prompt"],
            review_prompt=d.get("review_prompt", ""),
        )


@dataclass
class OrchestratorConfig:
    """Top-level orchestrator configuration."""

    run_id: str
    base_branch: str
    max_budget_usd: float
    review_after_implement: bool
    tasks: list[TaskConfig]

    @classmethod
    def from_dict(cls, d: dict) -> "OrchestratorConfig":
        # Validate required top-level fields before anything else.
        missing = []
        if "run_id" not in d:
            missing.append("run_id")
        if "tasks" not in d:
            missing.append("tasks")
        if missing:
            print(
                f"[ERROR] Config is missing required field(s): {', '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)
        tasks = [TaskConfig.from_dict(t) for t in d["tasks"]]
        return cls(
            run_id=d["run_id"],
            base_branch=d.get("base_branch", "main"),
            max_budget_usd=float(d.get("max_budget_usd", 10.0)),
            review_after_implement=bool(d.get("review_after_implement", False)),
            tasks=tasks,
        )


@dataclass
class TaskResult:
    """Result of executing a single task."""

    task_name: str
    worktree: str
    implementer_rc: Optional[int] = None
    reviewer_rc: Optional[int] = None
    implementer_log: Optional[str] = None
    reviewer_log: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True if the task completed without error and implementer returned 0."""
        return self.error is None and self.implementer_rc == 0


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> OrchestratorConfig:
    """Load configuration from a JSON or YAML file.

    JSON is always supported.  YAML requires PyYAML; if missing, a clear
    error message is printed and the process exits.
    """
    path = Path(config_path)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        if not HAS_YAML:
            print("[ERROR] YAML configuration requires PyYAML.", file=sys.stderr)
            print("[INFO]  Install with:  pip install pyyaml", file=sys.stderr)
            print("[INFO]  Or convert your config to JSON format.", file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as fh:
            raw = _yaml.safe_load(fh)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        print(f"[ERROR] Unsupported config format: {suffix}", file=sys.stderr)
        print("[INFO]  Supported formats: .json, .yaml, .yml", file=sys.stderr)
        sys.exit(1)

    return OrchestratorConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# Low-level git helpers  (all safe: read-only or worktree creation only)
# ---------------------------------------------------------------------------

GIT = "git"


def _run_git(
    args: list[str], cwd: Path, timeout: int = 120
) -> subprocess.CompletedProcess:
    """Run a git command, capturing stdout and stderr.  No shell=True."""
    return subprocess.run(
        [GIT] + args,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=timeout,
    )


def _git_root(cwd: Path) -> Path:
    """Return the absolute path to the repository root."""
    proc = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if proc.returncode != 0:
        print("[ERROR] Not inside a git repository.", file=sys.stderr)
        sys.exit(1)
    return Path(proc.stdout.strip())


def _is_worktree(path: Path) -> bool:
    """Check whether *path* is a valid git worktree (or main tree)."""
    return _run_git(["rev-parse", "--git-dir"], cwd=path).returncode == 0


def _is_clean(path: Path) -> bool:
    """Check whether the git working tree at *path* has no uncommitted changes."""
    proc = _run_git(["status", "--porcelain"], cwd=path)
    return proc.stdout.strip() == ""


def _create_worktree(
    repo_root: Path, branch: str, worktree_path: str, base_branch: str
) -> subprocess.CompletedProcess:
    """Create a new git worktree with a new branch based on *base_branch*."""
    return _run_git(
        ["worktree", "add", "-b", branch, worktree_path, base_branch],
        cwd=repo_root,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# Claude agent execution  (the "parallel" part is orchestrated above this)
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string (compact)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_claude_agent(
    *,
    agent: str,
    prompt: str,
    max_budget_usd: float,
    worktree_path: Path,
    log_path: Path,
) -> int:
    """Run `claude -p --agent <agent> ...` inside *worktree_path*.

    Stdout and stderr are merged and written to *log_path*.  Returns the
    process exit code.
    """
    # Ensure the log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "claude",
        "-p",
        "--agent",
        agent,
        "--no-session-persistence",
        "--max-budget-usd",
        str(max_budget_usd),
        prompt,
    ]

    with open(str(log_path), "w", encoding="utf-8") as fh:
        # --- header ---
        fh.write(f"{'=' * 60}\n")
        fh.write(f"Agent:       {agent}\n")
        fh.write(f"Worktree:    {worktree_path}\n")
        fh.write(f"Max budget:  ${max_budget_usd:.2f} USD\n")
        fh.write(f"Started:     {_utc_now_iso()}\n")
        fh.write(f"Command:     {' '.join(cmd)}\n")
        fh.write(f"{'=' * 60}\n\n")
        fh.flush()

        # --- launch ---
        # subprocess.Popen is used (rather than run) so the caller can manage
        # multiple processes in parallel across threads.
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(worktree_path),
            text=True,
            # No shell=True — avoids injection risk on Windows.
        )
        proc.wait()

        # --- footer ---
        fh.write(f"\n{'=' * 60}\n")
        fh.write(f"Exit code:   {proc.returncode}\n")
        fh.write(f"Finished:    {_utc_now_iso()}\n")
        fh.write(f"{'=' * 60}\n")

    return proc.returncode


# ---------------------------------------------------------------------------
# Single-task execution  (runs in a thread; see execute_all)
# ---------------------------------------------------------------------------


def _execute_one_task(
    task: TaskConfig,
    config: OrchestratorConfig,
    repo_root: Path,
    logs_dir: Path,
    dry_run: bool = False,
) -> TaskResult:
    """Execute a single task: prepare worktree → implementer → (opt) reviewer.

    This function is designed to be called from a thread so multiple tasks
    can run in parallel.  It never merges, commits, pushes, or deletes
    anything — it only creates (or reuses) worktrees and runs agents inside
    them.

    When *dry_run* is True, only validate worktree paths — no worktrees are
    created and no agents are executed.
    """
    result = TaskResult(task_name=task.name, worktree=task.worktree)

    # Normalize the worktree path: absolute paths are used as-is; relative
    # paths are resolved against the repository root.
    wt = Path(task.worktree)
    if not wt.is_absolute():
        wt = repo_root / task.worktree
    wt = wt.resolve()
    result.worktree = str(wt)

    # ---- 1. Prepare the worktree -----------------------------------------
    if wt.exists():
        if not _is_worktree(wt):
            result.error = f"Path exists but is not a valid git worktree: {wt}"
            return result
        if not _is_clean(wt):
            result.error = f"Existing worktree is not clean (has uncommitted changes): {wt}"
            return result
        print(f"[{task.name}]  Reusing existing worktree: {wt}")
    else:
        print(f"[{task.name}]  Creating worktree: {wt}")
        if not dry_run:
            proc = _create_worktree(repo_root, task.branch, str(wt), config.base_branch)
            if proc.returncode != 0:
                result.error = (
                    f"Failed to create worktree.\n"
                    f"  stdout: {proc.stdout.strip()}\n"
                    f"  stderr: {proc.stderr.strip()}"
                )
                return result

    # Set planned log paths (used by both real and dry-run modes).
    impl_log = logs_dir / f"{task.name}_implementer.log"
    result.implementer_log = str(impl_log)

    # ---- dry-run: stop here, report plan, do not create files or run agents ---
    if dry_run:
        planned_log = str(impl_log)
        if config.review_after_implement:
            planned_log += f"  +  {logs_dir / f'{task.name}_reviewer.log'}"
        print(f"[{task.name}]  [DRY-RUN]  Would run agent={task.agent}, logs → {planned_log}")
        return result

    # ---- 2. Run implementer ----------------------------------------------
    print(f"[{task.name}]  Implementer starting  (agent={task.agent}) ...")
    rc = _run_claude_agent(
        agent=task.agent,
        prompt=task.prompt,
        max_budget_usd=config.max_budget_usd,
        worktree_path=wt,
        log_path=impl_log,
    )
    result.implementer_rc = rc
    print(f"[{task.name}]  Implementer finished  (rc={rc})")

    # If the implementer failed, skip the reviewer even when requested.
    if rc != 0 and config.review_after_implement:
        print(
            f"[{task.name}]  Implementer failed (rc={rc}) — skipping reviewer."
        )
        return result

    # ---- 3. Run reviewer (optional) --------------------------------------
    if config.review_after_implement:
        review_log = logs_dir / f"{task.name}_reviewer.log"
        result.reviewer_log = str(review_log)

        review_prompt = task.review_prompt or (
            f"Review the changes made by the implementer. "
            f"Original task: {task.prompt}"
        )

        print(f"[{task.name}]  Reviewer starting ...")
        rc = _run_claude_agent(
            agent="reviewer",
            prompt=review_prompt,
            max_budget_usd=config.max_budget_usd,
            worktree_path=wt,
            log_path=review_log,
        )
        result.reviewer_rc = rc
        print(f"[{task.name}]  Reviewer finished  (rc={rc})")

    return result


# ---------------------------------------------------------------------------
# Parallel execution orchestration
# ---------------------------------------------------------------------------


def execute_all(
    config: OrchestratorConfig,
    repo_root: Path,
    logs_dir: Path,
    max_parallel: Optional[int] = None,
    dry_run: bool = False,
) -> list[TaskResult]:
    """Run all tasks in parallel using a thread pool.

    Each task is submitted to a ``ThreadPoolExecutor``.  Inside each thread
    the implementer (and optionally reviewer) are launched via
    ``subprocess.Popen``, so tasks make progress concurrently even though
    individual agent runs are blocking within their thread.
    """
    workers = max_parallel if max_parallel else len(config.tasks)

    results: list[TaskResult] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(
                _execute_one_task, task, config, repo_root, logs_dir, dry_run
            ): task
            for task in config.tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                r = future.result()
            except Exception as exc:
                r = TaskResult(
                    task_name=task.name,
                    worktree=task.worktree,
                    error=f"Unhandled exception in task thread: {exc}",
                )
            results.append(r)

    # Sort results to match the original task order in the config.
    order = {t.name: i for i, t in enumerate(config.tasks)}
    results.sort(key=lambda r: order.get(r.task_name, 9999))
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(
    results: list[TaskResult],
    run_id: str,
    elapsed_s: float,
    dry_run: bool = False,
) -> None:
    """Print a human-readable summary of every task result."""

    def _rc_str(rc: Optional[int]) -> str:
        if rc is None:
            return "-"
        return str(rc)

    # In dry-run mode a task with implementer_rc=None is a pass, not a failure.
    if dry_run:
        any_failure = any(r.error is not None for r in results)
    else:
        any_failure = any(not r.ok for r in results)
    dry_tag = " [DRY-RUN]" if dry_run else ""

    print()
    print("=" * 72)
    print(f"  PARALLEL ORCHESTRATOR  v2.3  —  SUMMARY{dry_tag}")
    print("=" * 72)
    print(f"  Run ID:      {run_id}")
    print(f"  Tasks:       {len(results)}")
    print(f"  Elapsed:     {elapsed_s:.1f} s")
    print(f"  Final state: {'SOME TASKS FAILED' if any_failure else 'ALL TASKS OK'}")
    print("-" * 72)

    for r in results:
        print()
        print(f"  Task:              {r.task_name}")
        print(f"  Worktree:          {r.worktree}")
        if r.error:
            # Indent multi-line errors for readability.
            for line in r.error.splitlines():
                print(f"  ERROR              {line}")
        else:
            print(f"  Implementer rc:    {_rc_str(r.implementer_rc)}")
            print(f"  Implementer log:   {r.implementer_log}")
            if r.reviewer_rc is not None:
                print(f"  Reviewer rc:       {_rc_str(r.reviewer_rc)}")
                print(f"  Reviewer log:      {r.reviewer_log}")

    print()
    if dry_run:
        print(
            "[DRY-RUN]  No worktrees were created and no agents were "
            "executed.  Re-run without --dry-run to execute for real."
        )
    elif any_failure:
        print(
            "[WARNING]  Some tasks failed.  No integration or merge has been "
            "performed."
        )
        print(
            "[WARNING]  Review the per-task logs above, fix issues, then "
            "re-run or merge manually."
        )
    else:
        print(
            "[OK]  All tasks completed.  Review the worktrees manually, then "
            "merge when ready."
        )
    print()


def _write_summary_json(
    config: OrchestratorConfig,
    results: list[TaskResult],
    logs_dir: Path,
) -> None:
    """Write a structured JSON summary of the orchestrator run to summary.json.

    Called only after a real (non-dry-run) execution completes.
    The summary includes run-level metadata and per-task results, including
    branch names matched from the original task configuration.
    """
    # Build a lookup so we can attach the branch name to each result.
    task_branch_map: dict[str, str] = {t.name: t.branch for t in config.tasks}

    any_failure = any(not r.ok for r in results)
    overall_status = "SOME_TASKS_FAILED" if any_failure else "ALL_TASKS_OK"

    tasks_summary: list[dict] = []
    for r in results:
        branch = task_branch_map.get(r.task_name, "")
        tasks_summary.append(
            {
                "name": r.task_name,
                "branch": branch,
                "worktree": r.worktree,
                "implementer_rc": r.implementer_rc,
                "reviewer_rc": r.reviewer_rc,
                "implementer_log": r.implementer_log,
                "reviewer_log": r.reviewer_log,
                "error": r.error,
            }
        )

    summary = {
        "run_id": config.run_id,
        "base_branch": config.base_branch,
        "max_budget_usd": config.max_budget_usd,
        "review_after_implement": config.review_after_implement,
        "overall_status": overall_status,
        "tasks": tasks_summary,
    }

    summary_path = logs_dir / "summary.json"
    with open(str(summary_path), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"[INFO]  Summary written: {summary_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="parallel_orchestrator.py",
        description=(
            "Parallel Orchestrator v2.3 — run Claude Code agents in parallel "
            "across isolated git worktrees."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python scripts/parallel_orchestrator.py tasks.json
  python scripts/parallel_orchestrator.py tasks.yaml
  python scripts/parallel_orchestrator.py tasks.json --max-parallel 2

Configuration file (JSON)
-------------------------
  {
    "run_id": "my-run-001",
    "base_branch": "main",
    "max_budget_usd": 5.0,
    "review_after_implement": true,
    "tasks": [
      {
        "name": "add_feature_x",
        "branch": "feat/add-x",
        "worktree": "../worktrees/add_x",
        "agent": "implementer",
        "prompt": "Add feature X to the codebase",
        "review_prompt": "Review the feature X implementation"
      },
      {
        "name": "fix_bug_y",
        "branch": "fix/bug-y",
        "worktree": "../worktrees/fix_y",
        "agent": "implementer",
        "prompt": "Fix bug Y in the authentication flow",
        "review_prompt": "Review the bug Y fix"
      }
    ]
  }

The same structure works for YAML (requires PyYAML).
""",
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to configuration file (.json or .yaml/.yml).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help=(
            "Maximum number of tasks to run concurrently "
            "(default: run all tasks in parallel)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate configuration and worktree paths, then print what would "
            "happen without creating worktrees, running agents, or writing logs."
        ),
    )
    args = parser.parse_args()

    # No config → print help and exit cleanly.
    if not args.config:
        parser.print_help()
        sys.exit(0)

    # ---- 0. Locate repo root ----------------------------------------------
    repo_root = _git_root(Path.cwd())
    print(f"[INFO]  Repository root:  {repo_root}")

    # ---- 1. Main working tree MUST be clean -------------------------------
    if not _is_clean(repo_root):
        print(
            "[ERROR]  Main working tree is not clean (uncommitted changes).",
            file=sys.stderr,
        )
        print(
            "[ERROR]  Please commit, stash, or discard changes before "
            "running the orchestrator.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[INFO]  Main working tree is clean.")

    # ---- 2. Load configuration --------------------------------------------
    config = load_config(args.config)
    print(f"[INFO]  Run ID:              {config.run_id}")
    print(f"[INFO]  Base branch:         {config.base_branch}")
    print(f"[INFO]  Max budget / task:   ${config.max_budget_usd:.2f} USD")
    print(f"[INFO]  Review after impl:   {config.review_after_implement}")
    print(f"[INFO]  Number of tasks:     {len(config.tasks)}")
    for t in config.tasks:
        print(f"         - {t.name}  |  branch: {t.branch}  |  agent: {t.agent}")

    # ---- 3. Prepare logs directory ----------------------------------------
    logs_dir = repo_root / "logs" / config.run_id
    if args.dry_run:
        print(f"[INFO]  Logs directory (planned): {logs_dir}")
    else:
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO]  Logs directory:      {logs_dir}")

    # ---- 4. Execute all tasks in parallel ---------------------------------
    t0 = time.monotonic()
    results = execute_all(
        config=config,
        repo_root=repo_root,
        logs_dir=logs_dir,
        max_parallel=args.max_parallel,
        dry_run=args.dry_run,
    )
    elapsed = time.monotonic() - t0

    # ---- 5. Print summary ------------------------------------------------
    _print_summary(results, config.run_id, elapsed, dry_run=args.dry_run)

    # ---- 6. Write structured JSON summary (not in dry-run) ---------------
    if not args.dry_run:
        _write_summary_json(config, results, logs_dir)

    # ---- 7. Exit code ----------------------------------------------------
    if args.dry_run:
        any_failure = any(r.error is not None for r in results)
    else:
        any_failure = any(not r.ok for r in results)
    sys.exit(1 if any_failure else 0)


if __name__ == "__main__":
    main()
