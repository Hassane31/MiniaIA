"""Runs commands. Knows nothing about the LLM.

Flow for every command:  security.assess -> (approval) -> run -> log.
No shell is ever used: pipelines are built with subprocess pipes, argv lists only.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

from .config import Config
from .command_lint import lint
from .error_hints import diagnose
from .security import Assessment, TargetPolicy, assess, redact

Approver = Callable[[str, Assessment], bool]

# Children get a minimal, fixed environment: no API keys/tokens inherited, and a
# PATH without user-writable directories (prevents PATH hijacking).
FIXED_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_PASSTHROUGH_ENV = ("HOME", "USER", "LOGNAME", "TERM", "TZ", "LANG")


def build_env() -> dict:
    env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
    env["PATH"] = FIXED_PATH
    env["LC_ALL"] = "C.UTF-8"  # stable, English output that is easy for the model to read
    return env


@dataclass
class ExecResult:
    command: str
    status: str  # ok | failed | timeout | denied | blocked | invalid | error
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    assessment: Optional[Assessment] = None
    truncated: bool = False

    @property
    def ran(self) -> bool:
        return self.status in ("ok", "failed", "timeout")

    def observation(self, limit: int = 6000) -> str:
        """Text fed back to the model: redacted and size-limited."""
        head = f"OBSERVATION [{self.status}" + (f", exit code {self.exit_code}" if self.exit_code is not None else "") + "]"
        if self.status == "blocked":
            return (f"{head}: blocked by the safety policy ({self.stderr}). Do NOT retry this command; "
                    "choose a safer alternative or explain the situation to the user.")
        if self.status == "denied":
            return (f"{head}: the user refused permission to run this command. Do NOT retry it; "
                    "propose an alternative or explain what the user could do manually.")
        if self.status == "invalid":
            return (f"{head}: NOT RUN because it is malformed: {self.stderr}\n"
                    "Fix the command and propose ONE corrected command, or use a built-in tool / answer instead.")
        parts = []
        if self.status == "timeout":
            parts.append("The command timed out and was killed.")
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.rstrip())
        if not self.stdout.strip() and not self.stderr.strip():
            parts.append("(no output)")
        body = redact("\n".join(parts))
        if len(body) > limit:
            body = body[:limit] + "\n...[output truncated]"
        elif self.truncated:
            body += "\n...[output truncated]"
        text = f"{head}:\n{body}"
        hints = diagnose(self.status, self.stderr)
        if self.status != "ok":
            text += ("\nThe command did not succeed. Briefly explain the cause, then propose ONE corrected command "
                     "(it will be validated again and may need approval again) or answer the user. "
                     "Stay on the user's original request; do not call unrelated tools.")
            if hints:
                text += "\nHINTS: " + " ".join(hints)
        return text


class CommandLogger:
    """Append-only JSON-lines log (0600), secrets redacted, size-capped."""

    MAX_BYTES = 5 * 1024 * 1024

    def __init__(self, path: Path):
        self.path = Path(path)
        self._warned = False

    def log(self, result: ExecResult) -> None:
        a = result.assessment
        entry = {
            "type": "command",
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": redact(result.command),
            "status": result.status,
            "exit_code": result.exit_code,
            "level": a.level.value if a else None,
            "categories": list(a.categories) if a else [],
            "duration_s": round(result.duration, 3),
            "stdout": redact(result.stdout)[:2000],
            "stderr": redact(result.stderr)[:2000],
        }
        self._write(entry)

    def log_tool(self, name: str, args: dict, status: str, duration: float, summary: str, commands=()) -> None:
        try:
            safe_args = {str(k): redact(v) if isinstance(v, str) else v for k, v in args.items()}
            self._write({"type": "tool", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": name,
                         "args": safe_args, "status": status, "duration_s": round(duration, 3),
                         "commands": [redact(c) for c in commands], "summary": redact(summary)[:500]})
        except Exception:      # logging must never break a tool call
            pass

    def _write(self, entry: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.path.exists() and self.path.stat().st_size > self.MAX_BYTES:
                self.path.replace(self.path.with_suffix(".log.1"))
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            if not self._warned:
                self._warned = True
                print(f"miniaIA: warning: cannot write log {self.path}: {exc}", file=sys.stderr)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except OSError:
            pass


def _read_limited(fh, limit_chars: int) -> tuple:
    fh.seek(0)
    data = fh.read(limit_chars * 4 + 1)
    text = data.decode("utf-8", errors="replace")
    more = len(data) > limit_chars * 4 or len(text) > limit_chars
    return text[:limit_chars], more


class CommandExecutor:
    def __init__(self, config: Config, approver: Optional[Approver] = None,
                 logger: Optional[CommandLogger] = None, cwd: Optional[str] = None,
                 targets: Optional[TargetPolicy] = None):
        self.config = config
        self.approver: Approver = approver or (lambda cmd, assessment: False)  # default: deny
        self.logger = logger
        self.targets = targets
        # UI hook: activity(label) -> context manager shown while a command runs (set by main)
        self.activity: Callable[[str], ContextManager] = lambda label: nullcontext()
        self.cwd = os.path.realpath(cwd or os.getcwd())
        self._env = build_env()

    # ------------------------------------------------------------------ public
    def run(self, command: str) -> ExecResult:
        command = command.strip()
        assessment = assess(command, self.cwd, self.targets)

        if assessment.blocked:
            return self._finish(command, "blocked", assessment, stderr=assessment.summary)

        problems = lint(assessment.segments)
        if problems:  # malformed: do not bother the human with an approval prompt for it
            return self._finish(command, "invalid", assessment, stderr="; ".join(problems))

        if assessment.needs_approval:
            try:
                approved = bool(self.approver(command, assessment))
            except (KeyboardInterrupt, EOFError):
                approved = False
            if not approved:
                return self._finish(command, "denied", assessment)

        try:
            if self._needs_sudo(assessment.segments) and not self._prime_sudo():
                return self._finish(command, "error", assessment, stderr="sudo authentication failed")
            with self.activity(f"Executing: {command}"):
                return self._execute(command, assessment)
        except Exception as exc:  # never crash the agent loop
            return self._finish(command, "error", assessment, stderr=f"internal error: {exc}")

    # ---------------------------------------------------------------- internals
    def _finish(self, command: str, status: str, assessment: Assessment, **kw) -> ExecResult:
        result = ExecResult(command=command, status=status, assessment=assessment, **kw)
        if self.logger:
            self.logger.log(result)
        return result

    def _builtin_cd(self, command: str, segments: tuple, assessment: Assessment) -> Optional[ExecResult]:
        if len(segments) != 1 or segments[0][0] != "cd":
            return None
        args = list(segments[0][1:])
        if len(args) > 1:
            return self._finish(command, "failed", assessment, exit_code=1, stderr="cd: too many arguments")
        target = os.path.realpath(os.path.join(self.cwd, os.path.expanduser(args[0] if args else "~")))
        if not os.path.isdir(target):
            return self._finish(command, "failed", assessment, exit_code=1, stderr=f"cd: no such directory: {target}")
        self.cwd = target
        return self._finish(command, "ok", assessment, exit_code=0, stdout=target + "\n")

    def _needs_sudo(self, segments: tuple) -> bool:
        return any(os.path.basename(s[0]) == "sudo" for s in segments)

    def _prime_sudo(self) -> bool:
        """Ask sudo for the password directly on the user's terminal (never through the model)."""
        if not (self.config.interactive_sudo and sys.stdin.isatty()):
            return True
        return subprocess.run(["sudo", "-v"]).returncode == 0

    def _execute(self, command: str, assessment: Assessment) -> ExecResult:
        segments = assessment.segments
        cd_result = self._builtin_cd(command, segments, assessment)
        if cd_result:
            return cd_result

        limit = self.config.max_output_chars
        start = time.monotonic()
        deadline = start + self.config.command_timeout
        procs: list = []
        err_files: list = []
        out_file = tempfile.TemporaryFile()
        prev_stdout = None
        try:
            for idx, argv in enumerate(segments):
                last = idx == len(segments) - 1
                err = tempfile.TemporaryFile()
                err_files.append(err)
                try:
                    proc = subprocess.Popen(
                        list(argv), shell=False, cwd=self.cwd, env=self._env,
                        stdin=prev_stdout if prev_stdout else subprocess.DEVNULL,
                        stdout=out_file if last else subprocess.PIPE,
                        stderr=err, start_new_session=True,
                    )
                except FileNotFoundError:
                    return self._finish(command, "error", assessment, stderr=f"command not found: {argv[0]}",
                                        duration=time.monotonic() - start)
                except PermissionError:
                    return self._finish(command, "error", assessment, stderr=f"permission denied: {argv[0]}",
                                        duration=time.monotonic() - start)
                if prev_stdout:
                    prev_stdout.close()
                prev_stdout = proc.stdout
                procs.append(proc)

            timed_out = False
            for proc in procs:
                try:
                    proc.wait(timeout=max(0.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    break
            if timed_out:
                for proc in procs:
                    _kill_group(proc)
                    proc.wait()

            duration = time.monotonic() - start
            stdout, cut_out = _read_limited(out_file, limit)
            errs, cut_err = [], False
            for f in err_files:
                text, cut = _read_limited(f, limit)
                cut_err = cut_err or cut
                if text.strip():
                    errs.append(text.rstrip())
            stderr = "\n".join(errs)[:limit]
            code = procs[-1].returncode
            if timed_out:
                status = "timeout"
                stderr = (stderr + f"\ntimed out after {self.config.command_timeout:.0f}s").strip()
            else:
                status = "ok" if code == 0 else "failed"
            return self._finish(command, status, assessment, exit_code=code, stdout=stdout, stderr=stderr,
                                duration=duration, truncated=cut_out or cut_err)
        finally:
            for proc in procs:
                if proc.poll() is None:  # e.g. Ctrl+C
                    _kill_group(proc)
                    proc.wait()
                if proc.stdout:
                    proc.stdout.close()
            out_file.close()
            for f in err_files:
                f.close()
