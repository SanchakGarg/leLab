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

"""Interactive model sessions: the commands typed during one session, and when a command is done.

The session (lelab/remote_policy.py's `RemoteRun` with `interactive=True`) waits with the arm still until a
command arrives, runs the policy with that command as its task for up to `command_time_s`, and ends it early
when it is done: the user says so, or the arm moved and then stayed still (`SettleDetector`). Policies can't
report success themselves, so "the arm stopped moving" is the best automatic guess; a policy that pauses
mid-task fools it, which is why the Done button exists too.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

# The arm counts as settled once every joint stays within SETTLE_DEG for SETTLE_S, after it has first moved
# MOVED_DEG away from where the command found it and at least MIN_RUN_S into the command. Joints are in
# degrees, the gripper on its 0-100 scale.
SETTLE_S = 3.0
SETTLE_DEG = 2.0
MOVED_DEG = 8.0
MIN_RUN_S = 4.0
# Readings further apart than this restart the still window: the loop was waiting for the model's next
# chunk, and an arm holding still for the model is not an arm that has finished.
MAX_GAP_S = 0.5
# The video shows this many of the latest commands.
CAPTION_COMMANDS = 3

Status = Literal["pending", "running", "completed", "not_completed", "stopped", "failed"]
FINAL: frozenset[str] = frozenset({"completed", "not_completed", "stopped", "failed"})


class SettleDetector:
    """Says when the arm has done something and then stopped moving."""

    def __init__(
        self,
        start_state: np.ndarray,
        settle_s: float | None = None,
        settle_deg: float | None = None,
        moved_deg: float | None = None,
        min_run_s: float | None = None,
    ) -> None:
        self.start = np.asarray(start_state, np.float64)
        self.settle_s = SETTLE_S if settle_s is None else settle_s
        self.settle_deg = SETTLE_DEG if settle_deg is None else settle_deg
        self.moved_deg = MOVED_DEG if moved_deg is None else moved_deg
        self.min_run_s = MIN_RUN_S if min_run_s is None else min_run_s
        self.moved = False
        self.t0: float | None = None
        self.window: deque[tuple[float, np.ndarray]] = deque()

    def clear(self) -> None:
        """Forget the recent positions (the bus is glitching: the readings are stand-ins)."""
        self.window.clear()

    def update(self, t: float, state: np.ndarray) -> bool:
        """Add the joint positions read at time `t` (seconds); True once the arm has settled."""
        state = np.asarray(state, np.float64)
        if self.t0 is None:
            self.t0 = t
        if self.window and t - self.window[-1][0] > MAX_GAP_S:
            self.window.clear()
        if not self.moved and float(np.max(np.abs(state - self.start))) >= self.moved_deg:
            self.moved = True
        self.window.append((t, state))
        while len(self.window) > 1 and t - self.window[1][0] >= self.settle_s:
            self.window.popleft()
        if not self.moved or t - self.t0 < self.min_run_s:
            return False
        if t - self.window[0][0] < self.settle_s:
            return False
        positions = np.stack([s for _, s in self.window])
        return float(np.max(positions.max(axis=0) - positions.min(axis=0))) <= self.settle_deg


@dataclass
class Command:
    id: int
    text: str
    submitted_at: float
    status: Status = "pending"
    # How it ended: "arm settled", "marked done", "time limit", "stopped", "session ended" or the error.
    how: str | None = None
    started_at: float | None = None  # the command's first action
    ended_at: float | None = None
    # Set by the Done (True) and Stop command (False) buttons; the control loop acts on it.
    end_request: bool | None = None

    def elapsed_s(self, now: float) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, (self.ended_at or now) - self.started_at)

    def to_dict(self, now: float) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "how": self.how,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_s": round(self.elapsed_s(now), 1),
        }


def _clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


class CommandLog:
    """Every command of one session, and the requests from the page for the control loop.

    `wake` is set whenever something the control loop waits for happens (a command arrives, Done or Stop
    command is pressed, the session is stopped), so it can stop sleeping or waiting for a chunk at once.
    """

    def __init__(self, command_time_s: float) -> None:
        self.command_time_s = command_time_s
        self.commands: list[Command] = []
        self.returning = False
        self.lock = threading.Lock()
        self.wake = threading.Event()

    def current(self) -> Command | None:
        """The command pending or running, if any."""
        with self.lock:
            return self._current()

    def _current(self) -> Command | None:
        last = self.commands[-1] if self.commands else None
        return last if last is not None and last.status not in FINAL else None

    def submit(self, text: str) -> Command:
        """Queue a command. ValueError (with the reason to show) when it can't run now."""
        text = " ".join(text.split())
        if not text:
            raise ValueError("Type a command first.")
        with self.lock:
            if self._current() is not None:
                raise ValueError(
                    "A command is still running. Wait for it to end, or press Done or Stop command."
                )
            if self.returning:
                raise ValueError("The arm is going back to its start pose. Send the command when it's there.")
            cmd = Command(id=len(self.commands) + 1, text=text, submitted_at=time.time())
            self.commands.append(cmd)
        self.wake.set()
        return cmd

    def take_pending(self) -> Command | None:
        with self.lock:
            cmd = self._current()
            return cmd if cmd is not None and cmd.status == "pending" else None

    def request_end(self, completed: bool) -> Command:
        """Done (True) or Stop command (False) for the current command. ValueError when none is running."""
        with self.lock:
            cmd = self._current()
            if cmd is None:
                raise ValueError("No command is running.")
            cmd.end_request = completed
        self.wake.set()
        return cmd

    def end_requested(self, cmd: Command) -> bool | None:
        with self.lock:
            return cmd.end_request

    def mark_started(self, cmd: Command, at: float) -> None:
        with self.lock:
            cmd.started_at = at
            cmd.status = "running"

    def finish(self, cmd: Command, status: Status, how: str) -> None:
        with self.lock:
            if cmd.status in FINAL:
                return
            cmd.status = status
            cmd.how = how
            cmd.ended_at = time.time()

    def set_returning(self, returning: bool) -> None:
        with self.lock:
            self.returning = returning

    def close(self, how: str = "session ended") -> None:
        """The session ended: a command still pending or running counts as stopped."""
        with self.lock:
            cmd = self._current()
        if cmd is not None:
            self.finish(cmd, "stopped", how)

    def state(self) -> str:
        with self.lock:
            if self.returning:
                return "returning"
            return "running" if self._current() is not None else "ready"

    def to_list(self, now: float) -> list[dict[str, Any]]:
        with self.lock:
            return [c.to_dict(now) for c in self.commands]

    def captions(self, now: float, limit: int = CAPTION_COMMANDS) -> tuple[tuple[str, str, str], ...]:
        """The latest `limit` commands for the video, oldest first: (status, text, time).

        Whole seconds only, so the video's caption strip changes (and is redrawn) at most once a second.
        """
        with self.lock:
            latest = self.commands[-limit:] if limit > 0 else []
            rows = []
            for cmd in latest:
                if cmd.status == "pending":
                    right = "starting"
                elif cmd.status == "running":
                    right = f"{_clock(cmd.elapsed_s(now))} / {_clock(self.command_time_s)}"
                elif cmd.status == "not_completed":
                    right = f"{_clock(cmd.elapsed_s(now))} time limit"
                else:
                    right = _clock(cmd.elapsed_s(now))
                rows.append((cmd.status, cmd.text, right))
            return tuple(rows)

    def summary(self) -> dict[str, int]:
        with self.lock:
            counts: dict[str, int] = {}
            for cmd in self.commands:
                counts[cmd.status] = counts.get(cmd.status, 0) + 1
            return counts
