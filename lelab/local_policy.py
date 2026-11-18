# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model runs on this machine through LeLab's own control loop.

A plain local run is a `lerobot-rollout` subprocess whose task is fixed on its
command line. Two kinds of run need more than that, and get the remote-policy
loop (lelab/remote_policy.py) instead, against lelab/policy_server.py started
here on 127.0.0.1:
- an interactive session, which changes the task with every command and keeps
  the arm connected in between;
- a run with chunk blending, which lerobot-rollout's local inference doesn't do.
Both record the same video as a remote run, too. The server runs with
--watch-stdin: its stdin is a pipe only LeLab holds, so it exits with LeLab
even after a crash.

Loading the model takes a while (tens of seconds, more on a CPU), so the start
request returns at once and a `LocalRunSetup` stands in for the run in
rollout's state: the page shows "setting up" and Stop works. Once the server
serves and the arm is connected, it hands over to the RemoteRun.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .policy_server import DEFAULT_PORT
from .remote_ssh import RemoteServerError
from .utils import config

logger = logging.getLogger(__name__)

# Models come from disk (Hub checkpoints are downloaded before this starts), but a big VLA on a CPU loads slowly.
LOAD_TIMEOUT_S = 900.0
# How long the server gets to exit after its stdin closes, then after SIGTERM, before it is killed.
CLOSE_GRACE_S = 3.0
TERMINATE_GRACE_S = 5.0
# `=== lelab_policy_server <version>: serving on <node IP>:<port> ===`; the port can be past DEFAULT_PORT.
_SERVING_RE = re.compile(r"serving on \S+:(\d+)")


class SetupStoppedError(Exception):
    """Stop was pressed while the session was setting up."""


def new_log_path() -> Path:
    log_dir = Path(config.INFERENCE_LOGS_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{int(time.time())}.log"


class LocalPolicyServer:
    """lelab/policy_server.py serving one checkpoint on 127.0.0.1, its output going to `log_path`."""

    def __init__(self, policy_path: str, log_path: Path) -> None:
        self.policy_path = policy_path
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self._serving = threading.Event()
        self._output_done = threading.Event()

    def command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "lelab.policy_server",
            "--policy",
            self.policy_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(DEFAULT_PORT),
            "--watch-stdin",
        ]

    def start(self) -> None:
        log = self.log_path.open("w", buffering=1)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                self.command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except BaseException:
            log.close()
            raise
        threading.Thread(target=self._pump, args=(log,), name="local-policy-server-log", daemon=True).start()
        logger.info("Local policy server started: pid=%s policy=%s", self.proc.pid, self.policy_path)

    def _pump(self, log: Any) -> None:
        """Copy the server's output to the log, watching for the line with its port."""
        try:
            assert self.proc is not None and self.proc.stdout is not None
            for raw in iter(self.proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                with contextlib.suppress(Exception):
                    log.write(line)
                if self.port is None and (match := _SERVING_RE.search(line)):
                    self.port = int(match.group(1))
                    self._serving.set()
        except Exception:
            logger.exception("Reading the local policy server's output failed")
        finally:
            with contextlib.suppress(Exception):
                log.close()
            self._output_done.set()

    def wait_ready(self, stop: threading.Event, timeout_s: float = LOAD_TIMEOUT_S) -> int:
        """The port once the server serves. Raises SetupStoppedError on `stop`, RemoteServerError if it fails."""
        deadline = time.monotonic() + timeout_s
        while self.port is None:
            if stop.is_set():
                raise SetupStoppedError()
            if self.proc is not None and self.proc.poll() is not None:
                self._output_done.wait(2.0)  # the last lines, with the error, reach the log
                if self.port is not None:
                    break
                raise RemoteServerError(self._failure(self.proc.returncode), 500)
            if time.monotonic() > deadline:
                raise RemoteServerError(
                    f"The model did not load within {timeout_s / 60:.0f} minutes. See {self.log_path}.", 504
                )
            self._serving.wait(0.2)
        return self.port

    def _failure(self, returncode: int | None) -> str:
        from .rollout import _extract_error_from_log

        error = _extract_error_from_log(str(self.log_path))
        head = f"The model could not be loaded (exit code {returncode})"
        return f"{head}: {error}" if error else f"{head}. See {self.log_path}."

    def close(self) -> None:
        """Stop the server: closing its stdin ends it (--watch-stdin); SIGTERM, then SIGKILL, if it lingers."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(Exception):
            assert proc.stdin is not None
            proc.stdin.close()
        try:
            proc.wait(CLOSE_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.terminate()
        try:
            proc.wait(TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            logger.warning("The local policy server did not exit; killing it")
            proc.kill()
            proc.wait()


class LocalRunSetup:
    """Stands in for a local run (an interactive session, or a run with chunk blending) while its model loads
    and the arm connects.

    It has what rollout and stop_run use of a run (finished, status(), result(), request_stop(), join(),
    submit_command(), end_command(), update_settings()). Once ready it puts the RemoteRun in its place in
    rollout's state.
    """

    def __init__(self, request: Any, policy_path: str) -> None:
        from .remote_policy import RemoteInferenceRequest, video_layout

        self.request = request
        # Raises pydantic's ValidationError for values the loop doesn't take (a duration over an hour, say).
        self.remote_request = RemoteInferenceRequest(
            follower_port=request.follower_port,
            follower_config=request.follower_config,
            connection="direct",
            policy_host="127.0.0.1",
            policy_port=DEFAULT_PORT,  # replaced by the port the server reports
            cameras=request.cameras,
            task=request.task,
            duration_s=request.duration_s,
            interactive=request.interactive,
            command_time_s=request.command_time_s,
            return_between_commands=request.return_between_commands,
            auto_complete=request.auto_complete,
            chunk_blending=request.chunk_blending,
            blend_steps=request.blend_steps,
            record_video=request.record_video,
            video_camera=request.video_camera,
            video_corner_camera=request.video_corner_camera,
        )
        if self.remote_request.record_video:
            video_layout(self.remote_request)  # a bad camera choice fails the start request, not the setup
        self.interactive = self.remote_request.interactive
        self.log_path = new_log_path()
        self.server = LocalPolicyServer(policy_path, self.log_path)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started_at = time.time()
        self.phase = "loading the model"
        self.finished = False
        self.error: str | None = None
        self.run: Any = None  # the RemoteRun, once handed over
        self._lock = threading.Lock()

    def start(self) -> None:
        self.server.start()
        self.thread = threading.Thread(target=self._setup, name="local-run-setup", daemon=True)
        self.thread.start()

    def _setup(self) -> None:
        from . import rollout
        from .remote_policy import connect_run

        run = None
        try:
            port = self.server.wait_ready(self.stop_event)
            self.phase = "connecting the arm and cameras"
            request = self.remote_request.model_copy(update={"policy_port": port})
            run = connect_run(request, server=self.server, label=self.request.policy_ref)
            with self._lock:
                handed_over = not self.stop_event.is_set() and rollout._replace_remote_run(self, run)
                if handed_over:
                    self.run = run
                    if self.interactive:  # blending switched while the model loaded
                        run.update_settings(
                            self.remote_request.chunk_blending, self.remote_request.blend_steps
                        )
            if handed_over:
                run.start()
                logger.info(
                    "Local %s started: %s",
                    "interactive session" if self.interactive else "run",
                    self.request.policy_ref,
                )
                return
        except SetupStoppedError:
            pass
        except RemoteServerError as exc:
            self.error = exc.message
        except Exception as exc:
            logger.exception("Starting the local run failed")
            self.error = f"{type(exc).__name__}: {exc}"
        if run is not None:
            run.discard()  # connected but stopped before it started; also closes the server
        self.server.close()
        self.phase = "stopped"
        self.finished = True
        rollout._finish_remote_run(self)

    def request_stop(self) -> None:
        with self._lock:
            self.stop_event.set()
            run = self.run
        if run is not None:
            run.request_stop()

    def join(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        if self.thread is not None:
            self.thread.join(timeout_s)
            if self.thread.is_alive():
                return False
        run = self.run
        return run.join(max(0.0, deadline - time.monotonic())) if run is not None else True

    def _refusal(self, message: str) -> dict[str, Any]:
        return {"success": False, "status_code": 409, "message": message}

    def submit_command(self, text: str) -> dict[str, Any]:
        if not self.interactive:
            return self._refusal("This run isn't interactive. Start an interactive session to send commands.")
        return self._refusal("The session is still starting. Send the command once it's ready.")

    def end_command(self, completed: bool) -> dict[str, Any]:
        return self._refusal("No command is running.")

    def update_settings(self, chunk_blending: bool, blend_steps: int | None = None) -> dict[str, Any]:
        """While the model loads: kept, and applied when the session starts."""
        if not self.interactive:
            return self._refusal(
                "Chunk blending of a run is set when it starts. Interactive sessions can change it between commands."
            )
        with self._lock:
            if self.run is not None:
                run = self.run
            else:
                run = None
                update: dict[str, Any] = {"chunk_blending": chunk_blending}
                if blend_steps is not None:
                    update["blend_steps"] = blend_steps
                self.remote_request = self.remote_request.model_copy(update=update)
        if run is not None:
            return run.update_settings(chunk_blending, blend_steps)
        return {
            "success": True,
            "message": "Chunk blending on" if chunk_blending else "Chunk blending off",
            "chunk_blending": chunk_blending,
            "blend_steps": self.remote_request.blend_steps,
        }

    def status(self) -> dict[str, Any]:
        now = time.time()
        request = self.remote_request
        return {
            "mode": "local",
            "started_at": self.started_at,
            "rollout_started_at": None,
            "elapsed_s": now - self.started_at,
            "rollout_elapsed_s": 0,
            "duration_s": None if self.interactive else request.duration_s,
            "policy_ref": self.request.policy_ref,
            "log_path": str(self.log_path),
            "video": None,
            "arm_link_issues": [],
            "interactive": {
                "state": "setting up",
                "phase": self.phase,
                "command_time_s": request.command_time_s,
                "return_between_commands": request.return_between_commands,
                "auto_complete": request.auto_complete,
                "chunk_blending": request.chunk_blending,
                "blend_steps": request.blend_steps,
                "commands": [],
            }
            if self.interactive
            else None,
        }

    def result(self) -> dict[str, Any]:
        from .rollout import _friendly_hint

        failed = self.error is not None
        return {
            **self.status(),
            "inference_active": False,
            "exited": True,
            "exit_code": 1 if failed else 0,
            "outcome": "failed" if failed else "ok",
            "error": self.error,
            "hint": _friendly_hint(self.error) if failed else None,
            "returned_to_start": None,
        }
