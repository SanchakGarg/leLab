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

"""Inference on the real follower with the policy running on a remote server.

The model runs in lelab/policy_server.py on the user's GPU server; this module
runs the control loop here, next to the arm. Each tick reads the joints and
cameras, plays one action from a local queue of the last predicted chunk and,
when `prefetch` actions or fewer are left, sends the current observation for
the next chunk in a background thread. The answer replaces the queue minus the
actions already played since that observation, so it lines up with the
present. The loop waits only when the queue is empty; the motors then hold
their last target. With chunk blending on, the first `blend_steps` actions of
the new chunk are crossfaded from the old chunk's actions not yet played (the
new chunk's weight ramping up), so the switch from one chunk to the next
doesn't jerk the arm where the two disagree.

The connection to the server goes through the saved SSH session (a
direct-tcpip channel, like `ssh -N -L`) or straight over TCP on the same
network.

Safety: lerobot's `max_relative_target` caps every step; a chunk must have the
right shape and finite values; an error reply, a dropped connection, a chunk
that doesn't arrive in time or a camera that stalls stops the run. Then the
arm returns to where it started (unless a motor command itself failed) and is
released with the torque off, like a local LeLab run. Stop is checked every
tick and while waiting for a chunk.

Interactive sessions (`interactive=True`) keep the arm and the connection
between commands: the arm holds still until the page sends a command, the
policy runs with that command as its task until it is done or its time is up
(lelab/interactive.py decides "done"), and the arm optionally goes back to the
start pose before the next one. Local interactive sessions use the same loop
with lelab/policy_server.py running on this machine (lelab/local_policy.py).
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from . import remote_ssh
from .camera_config import validate_camera_rotations
from .interactive import CAPTION_COMMANDS, CommandLog, SettleDetector
from .policy_server import encode_jpeg, recv_msg, send_msg
from .remote_ssh import RemoteServerError
from .run_video import RunVideo, default_layout, video_file_name
from .utils import config
from .utils.bus_guard import GUARD_ATTR, ArmCommunicationLostError, BusGuard, notable_outages
from .utils.devices import safe_disconnect_device

logger = logging.getLogger(__name__)

HELLO_TIMEOUT_S = 10.0
# Give the browser's camera previews time to let go of the devices (as recording does).
CAMERA_RELEASE_S = 2.0
RETURN_DURATION_S = 3.0
RETURN_FPS = 50
# Between commands of an interactive session nothing is sent, and the policy server drops a client that is
# silent for 120 s (IDLE_TIMEOUT_S); a reset this often keeps the connection, and works with any server.
KEEPALIVE_S = 30.0


class StopRequestedError(Exception):
    """The user pressed Stop while the loop was waiting for a chunk."""


class RemotePolicyError(Exception):
    """The policy server or the connection to it failed."""


class _RobotIOError(Exception):
    def __init__(self, exc: BaseException, motors_ok: bool):
        super().__init__(str(exc))
        self.exc = exc
        self.motors_ok = motors_ok


class PolicyConnection(BaseModel):
    # Saved server whose SSH session carries the tunnel ("tunnel"), or None for "direct".
    server: str | None = None
    connection: Literal["tunnel", "direct"] = "tunnel"
    # Where the policy server listens, as seen from the server for "tunnel" (the GPU
    # node's IP the job log prints, or 127.0.0.1 when it runs on the login host),
    # or from this machine for "direct".
    policy_host: str = "127.0.0.1"
    policy_port: int = Field(5555, ge=1, le=65535)


class RemoteInferenceRequest(PolicyConnection):
    follower_port: str
    follower_config: str
    task: str = ""
    # Keyed by the policy's camera names (the hello's `cameras`).
    cameras: dict[str, dict[str, Any]] = {}
    duration_s: int = Field(60, ge=1, le=3600)
    # Control rate. None: the rate the server announces.
    fps: int | None = Field(None, ge=1, le=120)
    # Play only the first N actions of each chunk (replan more often). None: the server's.
    n_action_steps: int | None = Field(None, ge=1)
    # Ask for the next chunk while this many actions are still queued.
    prefetch: int = Field(10, ge=0, le=200)
    jpeg_quality: int = Field(90, ge=30, le=100)
    # Largest move per step (degrees; gripper 0-100), enforced by lerobot. None: no cap.
    max_relative_target: float | None = Field(20.0, gt=0)
    return_to_start: bool = True
    # How long to wait for a chunk once the run is going, and for the first one (warm-up, queues).
    chunk_timeout_s: float = Field(5.0, gt=0, le=120)
    first_chunk_timeout_s: float = Field(60.0, gt=0, le=600)
    # Record a video of the run: `video_camera` full frame and `video_corner_camera` inset in
    # a corner (policy camera names; None: no inset). Without `video_camera`, a wrist camera
    # goes in the corner when there is one.
    record_video: bool = False
    video_camera: str | None = None
    video_corner_camera: str | None = None
    # Interactive session: no task or duration up front; the page sends commands one at a time
    # (/inference-command), each running for at most `command_time_s` from its first action.
    interactive: bool = False
    command_time_s: int = Field(60, ge=5, le=600)
    # Back to the start pose after each command (the session's end always goes back with return_to_start).
    return_between_commands: bool = True
    # End a command as done once the arm has moved and then stays still (interactive.SettleDetector).
    auto_complete: bool = True
    # Crossfade into each new chunk over its first `blend_steps` actions (as far as the old chunk's
    # unplayed actions reach: blending needs prefetch > 0). An interactive session can switch it between
    # commands (/inference-settings).
    chunk_blending: bool = False
    blend_steps: int = Field(10, ge=1, le=200)

    @field_validator("cameras")
    @classmethod
    def validate_rotations(cls, cameras: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return validate_camera_rotations(cameras)


def _refusal(message: str, status_code: int = 409) -> dict[str, Any]:
    return {"success": False, "status_code": status_code, "message": message}


# ------------------------------------------------------------------------------------------------- connection
def _tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


def open_policy_socket(conn: PolicyConnection) -> tuple[Any, Any, str]:
    """(socket-like, SSH lease or None, description). The caller closes both."""
    host, port = conn.policy_host.strip(), conn.policy_port
    if not host:
        raise RemoteServerError("Enter the policy server's host.")
    if conn.connection == "tunnel":
        if not conn.server:
            raise RemoteServerError("Choose the server to tunnel through.")
        lease = remote_ssh.lease(conn.server)
        session = lease.__enter__()
        try:
            channel = remote_ssh.open_policy_channel(session, host, port, timeout=HELLO_TIMEOUT_S)
        except BaseException:
            lease.__exit__(None, None, None)
            raise
        return channel, lease, f"{host}:{port} via {conn.server}"
    try:
        sock = socket.create_connection((host, port), timeout=HELLO_TIMEOUT_S)
    except OSError as exc:
        raise RemoteServerError(
            f"Can't connect to {host}:{port} ({exc}). Is the policy server running, and is this machine on the "
            "same network? Otherwise connect through the server (tunnel).",
            502,
        ) from None
    _tune(sock)
    return sock, None, f"{host}:{port}"


def read_hello(sock: Any) -> dict[str, Any]:
    sock.settimeout(HELLO_TIMEOUT_S)
    try:
        hello, _ = recv_msg(sock)
    except TimeoutError:
        raise RemoteServerError(
            f"Connected, but the policy server said nothing within {HELLO_TIMEOUT_S:.0f} s. It is still loading "
            "the model, or it is an older server busy with another client.",
            504,
        ) from None
    except (ConnectionError, OSError, ValueError) as exc:
        raise RemoteServerError(
            f"Connected, but the other side hung up before the policy server's hello ({exc}). Nothing is "
            "listening on that port there: wait for 'serving on' in the server's log and use the IP and port "
            "it prints.",
            502,
        ) from None
    if hello.get("type") == "error":
        raise RemoteServerError(f"The policy server refused the connection: {hello.get('message')}", 409)
    if hello.get("type") != "hello" or "cameras" not in hello or "chunk_size" not in hello:
        raise RemoteServerError("That port answered, but not like a LeLab policy server.", 502)
    sock.settimeout(None)
    return hello


def _close(sock: Any, lease: Any) -> None:
    if sock is not None:
        with contextlib.suppress(Exception):
            sock.close()
    if lease is not None:
        lease.__exit__(None, None, None)


def probe(conn: PolicyConnection) -> dict[str, Any]:
    """Connect, read the policy server's hello and hang up."""
    sock, lease, address = open_policy_socket(conn)
    try:
        hello = read_hello(sock)
    finally:
        _close(sock, lease)
    return {**hello, "address": address}


# ------------------------------------------------------------------------------------------------------ client
class RemotePolicyClient:
    """Plays actions from the remote policy's chunks, fetching the next chunk ahead of time.

    A port of the SO-101 sim project's RemoteAgent, with timeouts, stop checks and chunk validation.
    """

    def __init__(
        self,
        sock: Any,
        hello: dict[str, Any],
        n_action_steps: int | None = None,
        prefetch: int = 10,
        jpeg_quality: int = 90,
        action_dim: int = 6,
        blend_steps: int = 0,
    ):
        self.sock = sock
        self.cameras = tuple(hello["cameras"])
        self.chunk_size = int(hello["chunk_size"])
        self.n_use = max(
            1, min(n_action_steps or hello.get("n_action_steps") or self.chunk_size, self.chunk_size)
        )
        self.prefetch = max(0, min(prefetch, self.n_use - 1))
        self.quality = jpeg_quality
        self.action_dim = action_dim
        self.lock = threading.Lock()
        self.next_id = 0
        self.queue: deque[np.ndarray] = deque()
        self.step = 0  # actions played since the last reset
        self.pending: tuple[threading.Thread, dict[str, Any]] | None = None
        self.chunks = 0
        self.rtt_ms: list[float] = []
        self.infer_ms: list[float] = []
        self.wait_s = 0.0
        # Crossfade into a new chunk over this many actions (0: replace the queue outright).
        self.blend_steps = blend_steps
        self.blended = 0  # actions blended so far

    def _check_chunk(self, header: dict[str, Any], body: bytes) -> np.ndarray:
        if header.get("type") == "error":
            raise RemotePolicyError(f"the policy server failed: {header.get('message')}")
        if header.get("type") != "chunk":
            raise RemotePolicyError(f"unexpected reply {header.get('type')!r} from the policy server")
        shape = header.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or shape[0] < 1 or shape[1] != self.action_dim:
            raise RemotePolicyError(
                f"the policy returned actions of shape {shape}, expected [T, {self.action_dim}]"
            )
        actions = np.frombuffer(body, np.float32)
        if actions.size != shape[0] * shape[1]:
            raise RemotePolicyError("the policy server sent a truncated action chunk")
        actions = actions.reshape(shape)
        if not np.isfinite(actions).all():
            raise RemotePolicyError("the policy returned NaN or infinite actions")
        return actions

    def _request(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> None:
        self.next_id += 1
        request_id = self.next_id
        result: dict[str, Any] = {"sent_step": self.step}

        def run() -> None:
            t = time.perf_counter()
            try:
                parts, payload, offset = {}, bytearray(), 0
                for cam in self.cameras:
                    jpg = encode_jpeg(images[cam], self.quality)
                    parts[cam] = [offset, len(jpg)]
                    payload += jpg
                    offset += len(jpg)
                header = {
                    "type": "obs",
                    "id": request_id,
                    "task": task,
                    "state": [float(x) for x in state],
                    "images": parts,
                }
                with self.lock:
                    send_msg(self.sock, header, bytes(payload))
                    reply, body = recv_msg(self.sock)
                result["rtt_ms"] = 1000 * (time.perf_counter() - t)
                result["infer_ms"] = float(reply.get("infer_ms", 0.0) or 0.0)
                result["actions"] = self._check_chunk(reply, body)
            except Exception as exc:  # reported in the control loop
                result["error"] = exc

        thread = threading.Thread(target=run, name="remote-policy-request", daemon=True)
        thread.start()
        self.pending = (thread, result)

    def _collect(self, wait: bool, stop: threading.Event | None = None, timeout_s: float = 5.0) -> None:
        if self.pending is None:
            return
        thread, result = self.pending
        if thread.is_alive():
            if not wait:
                return
            start = time.perf_counter()
            while thread.is_alive():
                if stop is not None and stop.is_set():
                    raise StopRequestedError()
                if time.perf_counter() - start > timeout_s:
                    raise RemotePolicyError(
                        f"no action chunk from the policy server within {timeout_s:.0f} s"
                    )
                thread.join(0.05)
            self.wait_s += time.perf_counter() - start
        self.pending = None
        if "error" in result:
            exc = result["error"]
            if isinstance(exc, RemotePolicyError):
                raise exc
            raise RemotePolicyError(f"lost the connection to the policy server ({exc})") from exc
        self.chunks += 1
        self.rtt_ms.append(result["rtt_ms"])
        self.infer_ms.append(result["infer_ms"])
        skip = self.step - result["sent_step"]  # actions played while the request was out
        fresh = result["actions"][skip : self.n_use]
        if len(fresh):
            self.queue = deque(self._blend(fresh))

    def _blend(self, fresh: np.ndarray) -> np.ndarray:
        """`fresh` with its first actions crossfaded from the old chunk's actions not played yet.

        Over n = min(blend_steps, overlap) actions the new chunk's weight ramps 1/(n+1), ..., n/(n+1); the
        action after the window is the new chunk's own. Nothing to blend with on a command's first chunk or
        when the queue had run empty.
        """
        n = min(self.blend_steps, len(self.queue), len(fresh))
        if n <= 0:
            return fresh
        old = np.array([self.queue[i] for i in range(n)], np.float64)
        weight = (np.arange(1, n + 1, dtype=np.float64) / (n + 1))[:, None]
        out = np.array(fresh, np.float32)  # a writable copy: the received chunk is read-only
        out[:n] = (1 - weight) * old + weight * fresh[:n]
        self.blended += n
        return out

    def next_action(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        task: str,
        stop: threading.Event | None = None,
        timeout_s: float = 5.0,
    ) -> np.ndarray:
        """The action to play now. Waits (checking `stop`) only when the queue is empty."""
        self._collect(wait=False)
        while not self.queue:
            if self.pending is None:
                self._request(images, state, task)
            self._collect(wait=True, stop=stop, timeout_s=timeout_s)
        if self.prefetch and self.pending is None and len(self.queue) <= self.prefetch:
            self._request(images, state, task)
        self.step += 1
        return self.queue.popleft()

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "rtt_ms_median": round(float(np.median(self.rtt_ms)), 1) if self.rtt_ms else None,
            "infer_ms_median": round(float(np.median(self.infer_ms)), 1) if self.infer_ms else None,
            "wait_s": round(self.wait_s, 2),
            "actions_per_chunk": self.n_use,
            "blend_steps": self.blend_steps,
            "blended": self.blended,
        }

    def drop(self) -> None:
        """Forget the queued actions and a chunk still on its way (waiting a little for its answer)."""
        if self.pending is not None:
            self.pending[0].join(5.0)
            self.pending = None
        self.queue.clear()
        self.step = 0

    def reset(self) -> None:
        """New episode: drop queued actions and clear the policy's state on the server."""
        self.drop()
        with self.lock:
            send_msg(self.sock, {"type": "reset"})

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.sock.close()


# --------------------------------------------------------------------------------------------------- the run
class RemoteRun:
    """One remote-policy run on the follower: its worker thread, progress and result."""

    def __init__(
        self,
        request: RemoteInferenceRequest,
        robot: Any,
        client: RemotePolicyClient,
        hello: dict,
        address: str,
        lease: Any,
        server: Any = None,
        label: str | None = None,
    ):
        self.request = request
        self.robot = robot
        self.client = client
        self.hello = hello
        self.address = address
        self.lease = lease
        # A local_policy.LocalPolicyServer this run owns (local interactive sessions): closed at the end.
        self.server = server
        self.label = label
        self.motors = list(robot.bus.motors)
        self.fps = int(request.fps or hello.get("fps") or 30)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started_at = time.time()
        self.rollout_started_at: float | None = None
        self.phase = "waiting for the first chunk"
        self.steps = 0
        self.finished = False
        self.error: str | None = None
        self.hint: str | None = None
        self.returned_to_start: bool | None = None
        self.initial: dict[str, float] | None = None
        self.commands = CommandLog(request.command_time_s) if request.interactive else None
        client.blend_steps = request.blend_steps if request.chunk_blending else 0
        self.video: RunVideo | None = None
        if request.record_video:
            main, corner = video_layout(request)
            path = Path(config.RUN_VIDEOS_PATH) / video_file_name(hello.get("name"))
            self.video = RunVideo(
                path, self.fps, main, corner, caption_lines=CAPTION_COMMANDS if request.interactive else 0
            )
        self._last_camera_frame = 0.0
        # Glitch tolerance on the follower's bus, from start() to the release.
        self.guard: BusGuard | None = None

    @property
    def policy_ref(self) -> str:
        return self.label or f"{self.hello.get('name', 'remote policy')} @ {self.address}"

    def _observe(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        try:
            obs = self.robot.get_observation()
        except ArmCommunicationLostError as exc:
            raise _RobotIOError(exc, motors_ok=False) from exc
        except Exception as exc:
            raise _RobotIOError(exc, motors_ok=_motors_respond(self.robot)) from exc
        state = np.array([obs[f"{m}.pos"] for m in self.motors], np.float32)
        return state, {cam: obs[cam] for cam in self.client.cameras}

    def _send(self, action: np.ndarray) -> None:
        try:
            self.robot.send_action({f"{m}.pos": float(v) for m, v in zip(self.motors, action, strict=True)})
        except Exception as exc:
            raise _RobotIOError(exc, motors_ok=False) from exc

    def _record(self, images: dict[str, np.ndarray]) -> None:
        if self.video is not None:
            captions = self.commands.captions(time.time()) if self.commands is not None else ()
            self.video.add(images, captions)

    def _record_cameras(self) -> None:
        """Keep the video going while the arm moves back to the start pose (no observations then)."""
        if self.video is None:
            return
        now = time.monotonic()
        if now - self._last_camera_frame < 1.0 / self.fps:
            return
        self._last_camera_frame = now
        cameras = getattr(self.robot, "cameras", None) or {}
        images = {}
        for name in (self.video.main, self.video.corner):
            if name and name in cameras:
                with contextlib.suppress(Exception):
                    images[name] = cameras[name].read_latest()
        if images:
            self._record(images)

    def _loop(self) -> None:
        if self.commands is not None:
            self._interactive_loop()
            return
        from lerobot.utils.robot_utils import precise_sleep

        period = 1.0 / self.fps
        req = self.request
        while not self.stop_event.is_set():
            tick = time.perf_counter()
            if (
                self.rollout_started_at is not None
                and time.time() - self.rollout_started_at >= req.duration_s
            ):
                return
            state, images = self._observe()
            first = self.rollout_started_at is None
            timeout = req.first_chunk_timeout_s if first else req.chunk_timeout_s
            action = self.client.next_action(images, state, req.task, self.stop_event, timeout)
            if first:
                self.rollout_started_at = time.time()
                self.phase = "running"
            self._send(action)
            self.steps += 1
            self._record(images)
            precise_sleep(period - (time.perf_counter() - tick))

    # ---------------------------------------------------------------------------------- interactive session
    def _interactive_loop(self) -> None:
        self.rollout_started_at = time.time()
        while True:
            self.phase = "ready"
            cmd = self._idle()
            if cmd is None:
                return
            self._run_command(cmd)
            if self.stop_event.is_set():
                return
            self._between_commands()

    def _keepalive(self) -> None:
        try:
            self.client.reset()
        except Exception as exc:
            raise RemotePolicyError(f"lost the connection to the policy server ({exc})") from exc

    def _idle(self) -> Any:
        """Hold still (watching the arm and filming) until a command comes; None once the session is stopped."""
        assert self.commands is not None
        period = 1.0 / self.fps
        wake = self.commands.wake
        last_ping = time.monotonic()
        while not self.stop_event.is_set():
            tick = time.perf_counter()
            wake.clear()
            cmd = self.commands.take_pending()
            if cmd is not None:
                return cmd
            _, images = self._observe()
            self._record(images)
            if time.monotonic() - last_ping >= KEEPALIVE_S:
                self._keepalive()
                last_ping = time.monotonic()
            wake.wait(max(0.0, period - (time.perf_counter() - tick)))
        return None

    def _run_command(self, cmd: Any) -> None:
        """Run the policy with `cmd` as its task until it is done, its time is up or it is stopped."""
        from lerobot.utils.robot_utils import precise_sleep

        log = self.commands
        assert log is not None
        req = self.request
        period = 1.0 / self.fps
        log.wake.clear()
        self.phase = "waiting for the first chunk"
        started: float | None = None
        detector: SettleDetector | None = None
        try:
            self._keepalive()  # a new episode for the policy: clears its state and our queue
            while True:
                if self.stop_event.is_set():
                    log.finish(cmd, "stopped", "session ended")
                    return
                end = log.end_requested(cmd)
                if end is not None:
                    log.finish(cmd, "completed" if end else "stopped", "marked done" if end else "stopped")
                    return
                tick = time.perf_counter()
                if started is not None and time.time() - started >= req.command_time_s:
                    log.finish(cmd, "not_completed", "time limit")
                    return
                state, images = self._observe()
                if detector is not None and req.auto_complete:
                    if self.guard is not None and self.guard.in_outage:
                        detector.clear()  # the guard answers with the last positions: not real stillness
                    elif detector.update(time.monotonic(), state):
                        log.finish(cmd, "completed", "arm settled")
                        return
                first = started is None
                timeout = req.first_chunk_timeout_s if first else req.chunk_timeout_s
                try:
                    action = self.client.next_action(images, state, cmd.text, log.wake, timeout)
                except StopRequestedError:
                    if not self.stop_event.is_set() and log.end_requested(cmd) is None:
                        log.wake.clear()  # woken by something this command doesn't act on
                    continue
                if first:
                    started = time.time()
                    log.mark_started(cmd, started)
                    detector = SettleDetector(state)
                    self.phase = "running"
                self._send(action)
                self.steps += 1
                self._record(images)
                precise_sleep(period - (time.perf_counter() - tick))
        except Exception as exc:
            log.finish(cmd, "failed", str(exc) or type(exc).__name__)
            raise

    def _between_commands(self) -> None:
        assert self.commands is not None
        self.client.drop()  # a chunk still on its way belongs to the command that just ended
        if not (self.request.return_between_commands and self.initial):
            return
        self.phase = "returning to the start pose"
        self.commands.set_returning(True)
        try:
            return_to(self.robot, self.initial, on_step=self._record_cameras)
        finally:
            self.commands.set_returning(False)

    def submit_command(self, text: str) -> dict[str, Any]:
        if self.commands is None:
            return _refusal("This run isn't interactive. Start an interactive session to send commands.")
        if self.finished or self.stop_event.is_set():
            return _refusal("The session is ending.")
        try:
            cmd = self.commands.submit(text)
        except ValueError as exc:
            return _refusal(str(exc))
        logger.info("Interactive session: command %d: %s", cmd.id, cmd.text)
        return {"success": True, "message": "Command sent", "command": cmd.to_dict(time.time())}

    def update_settings(self, chunk_blending: bool, blend_steps: int | None = None) -> dict[str, Any]:
        """Switch chunk blending of an interactive session, between commands."""
        if self.commands is None:
            return _refusal(
                "Chunk blending of a run is set when it starts. Interactive sessions can change it between commands."
            )
        if self.finished or self.stop_event.is_set():
            return _refusal("The session is ending.")
        if self.commands.current() is not None:
            return _refusal("A command is running. Change chunk blending between commands.")
        if blend_steps is not None:
            self.request.blend_steps = blend_steps
        self.request.chunk_blending = chunk_blending
        self.client.blend_steps = self.request.blend_steps if chunk_blending else 0
        logger.info(
            "Interactive session: chunk blending %s",
            f"on, over {self.request.blend_steps} actions" if chunk_blending else "off",
        )
        return {
            "success": True,
            "message": "Chunk blending on" if chunk_blending else "Chunk blending off",
            "chunk_blending": chunk_blending,
            "blend_steps": self.request.blend_steps,
        }

    def end_command(self, completed: bool) -> dict[str, Any]:
        if self.commands is None:
            return _refusal("This run isn't interactive.")
        try:
            self.commands.request_end(completed)
        except ValueError as exc:
            return _refusal(str(exc))
        return {"success": True, "message": "Command marked done" if completed else "Stopping the command"}

    # ------------------------------------------------------------------------------------------ the worker
    def run(self, initial: dict[str, float] | None) -> None:
        """Worker body: the loop, then the way back and the release, whatever happened."""
        self.initial = initial
        motors_ok = True
        try:
            self._loop()
        except StopRequestedError:
            pass
        except _RobotIOError as exc:
            motors_ok = exc.motors_ok
            lost = isinstance(exc.exc, ArmCommunicationLostError)
            self.error = str(exc.exc) if lost else f"{type(exc.exc).__name__}: {exc.exc}"
            logger.error("Remote inference: robot I/O failed: %s", self.error)
        except RemotePolicyError as exc:
            self.error = str(exc)
            logger.error("Remote inference stopped: %s", exc)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Remote inference failed")
        finally:
            if self.commands is not None:
                self.commands.close()
            self.client.close()  # also unblocks a request still waiting for its answer
            _close(None, self.lease)
            if self.request.return_to_start and initial and motors_ok:
                self.phase = "returning to the start pose"
                self.returned_to_start = return_to(self.robot, initial, on_step=self._record_cameras)
            if self.guard is not None:
                self.guard.uninstall()
            safe_disconnect_device(self.robot, logger)
            if self.server is not None:
                self.server.close()
            if self.video is not None:
                self.phase = "saving the video"
                self.video.close()
            self.phase = "stopped"
            self.finished = True
            from . import rollout

            rollout._finish_remote_run(self)
            logger.info("Remote inference ended after %d steps (%s)", self.steps, self.error or "ok")

    def start(self) -> None:
        # A failed read or write of the follower is skipped (the last positions
        # stand in) until it stops answering for MAX_OUTAGE_S. It stays on for
        # the way back, so a glitch there doesn't drop the arm.
        self.guard = BusGuard(self.robot.bus, "follower arm").install()
        initial = None
        try:
            positions = self.robot.bus.sync_read("Present_Position")
            initial = {f"{m}.pos": float(v) for m, v in positions.items()}
        except Exception as exc:
            logger.warning("Could not read the start pose; the arm will not return to it: %s", exc)
        self.thread = threading.Thread(target=self.run, args=(initial,), name="remote-inference", daemon=True)
        self.thread.start()

    def discard(self) -> None:
        """Release what connect_run opened, for a run that was never started."""
        self.client.close()
        _close(None, self.lease)
        safe_disconnect_device(self.robot, logger)
        if self.server is not None:
            self.server.close()
        if self.video is not None:
            self.video.close()
        self.finished = True

    def request_stop(self) -> None:
        self.stop_event.set()
        if self.commands is not None:
            self.commands.wake.set()

    def join(self, timeout_s: float) -> bool:
        """Wait for the worker to release the arm; False if it hasn't within `timeout_s`."""
        if self.thread is not None:
            self.thread.join(timeout_s)
            return not self.thread.is_alive()
        return True

    def status(self) -> dict[str, Any]:
        now = time.time()
        interactive = None
        if self.commands is not None:
            interactive = {
                "state": self.commands.state() if self.rollout_started_at else "setting up",
                "command_time_s": self.request.command_time_s,
                "return_between_commands": self.request.return_between_commands,
                "auto_complete": self.request.auto_complete,
                "chunk_blending": self.client.blend_steps > 0,
                "blend_steps": self.request.blend_steps,
                "commands": self.commands.to_list(now),
            }
        return {
            "mode": "local" if self.server is not None else "remote",
            "started_at": self.started_at,
            "rollout_started_at": self.rollout_started_at,
            "elapsed_s": now - self.started_at,
            "rollout_elapsed_s": (now - self.rollout_started_at) if self.rollout_started_at else 0,
            # A session has no overall duration (each command has its own limit).
            "duration_s": None if self.commands is not None else self.request.duration_s,
            "policy_ref": self.policy_ref,
            "log_path": str(self.server.log_path) if self.server is not None else None,
            "remote": {
                "address": self.address,
                "policy": self.hello.get("name"),
                "phase": self.phase,
                "steps": self.steps,
                "fps": self.fps,
                **self.client.stats(),
                "arm_skipped": self.guard.skipped if self.guard else 0,
            },
            "video": self.video.info() if self.video is not None else None,
            "arm_link_issues": notable_outages([self.guard], since=time.time() - 5) if self.guard else [],
            "interactive": interactive,
        }

    def result(self) -> dict[str, Any]:
        failed = self.error is not None
        return {
            **self.status(),
            "inference_active": False,
            "exited": True,
            "exit_code": 1 if failed else 0,
            "outcome": "failed" if failed else "ok",
            "error": self.error,
            "hint": self.hint or (_hint(self.error) if failed else None),
            "returned_to_start": self.returned_to_start,
        }


def video_layout(request: RemoteInferenceRequest) -> tuple[str, str | None]:
    """(full-frame camera, corner camera) for the run's video, checked against the run's cameras."""
    names = list(request.cameras)
    if not names:
        raise RemoteServerError("This model uses no camera, so there is no video to record.", 400)
    if request.video_camera:
        main, corner = request.video_camera, request.video_corner_camera
    else:
        main, corner = default_layout(names)
    for name in (main, corner):
        if name is not None and name not in names:
            raise RemoteServerError(
                f"There is no camera {name!r} in this run to record (cameras: {', '.join(names)}).", 400
            )
    return main, (None if corner == main else corner)


def _motors_respond(robot: Any) -> bool:
    guard = getattr(robot.bus, GUARD_ATTR, None)
    try:
        # Past the guard, which would answer with the last positions.
        (guard.raw_read if guard else robot.bus.sync_read)("Present_Position")
        return True
    except Exception:
        return False


def return_to(
    robot: Any,
    target: dict[str, float],
    duration_s: float | None = None,
    fps: int | None = None,
    on_step: Any = None,
) -> bool:
    """Move smoothly from the present joint positions to `target` (what lerobot rollout does at the end).

    `on_step()` runs after every step (an interactive session films the way back with it); its errors are
    ignored.
    """
    from lerobot.utils.robot_utils import precise_sleep

    duration_s = RETURN_DURATION_S if duration_s is None else duration_s
    fps = fps or RETURN_FPS
    try:
        present = robot.bus.sync_read("Present_Position")
        current = {f"{m}.pos": float(v) for m, v in present.items() if f"{m}.pos" in target}
        if not all(np.isfinite(list(current.values()))) or not all(np.isfinite(list(target.values()))):
            raise ValueError("non-finite joint positions")
        steps = max(int(duration_s * fps), 1)
        for step in range(1, steps + 1):
            t = step / steps
            robot.send_action({k: current[k] * (1 - t) + target[k] * t for k in current})
            if on_step is not None:
                with contextlib.suppress(Exception):
                    on_step()
            precise_sleep(1 / fps)
        return True
    except Exception as exc:
        logger.warning("Could not return to the start pose: %s", exc)
        return False


def _hint(error: str | None) -> str | None:
    low = (error or "").lower()
    if "no action chunk" in low:
        return (
            "The policy server stopped answering in time. Check its log on the server (a crash, or the job "
            "ended), and the network; raise the chunk timeout if inference is just slow."
        )
    if "lost the connection" in low:
        return "The connection to the policy server dropped: the job may have ended, or the SSH session or network went down."
    if "frame is too old" in low or "no frame" in low or "has not captured" in low:
        return (
            "A camera can't keep up — frames are arriving too slowly. Lower its resolution/FPS, set "
            "FOURCC=MJPG, and close other heavy apps, then try again."
        )
    if "the policy server failed" in low:
        return "The model raised an error on the server; its log has the details."
    from .rollout import _friendly_hint

    return _friendly_hint(error)


def connect_run(request: RemoteInferenceRequest, server: Any = None, label: str | None = None) -> RemoteRun:
    """Open the policy connection, check its hello against the request and connect the follower.

    Returns the run, not started yet. Raises (RemoteServerError with the message to show, or anything the
    arm raised) after closing whatever it opened. `server` and `label`: see RemoteRun.
    """
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    from . import record as _record
    from .utils.config import setup_follower_calibration_file

    sock = lease = robot = None
    try:
        sock, lease, address = open_policy_socket(request)
        hello = read_hello(sock)
        policy_cameras = list(hello.get("cameras") or [])
        if sorted(request.cameras) != sorted(policy_cameras):
            raise RemoteServerError(
                f"The policy expects the cameras {', '.join(policy_cameras) or '(none)'}; bind one camera to "
                f"each (got {', '.join(sorted(request.cameras)) or 'none'}).",
                400,
            )
        if request.record_video:
            video_layout(request)
        for key in ("state_dim", "action_dim"):
            if hello.get(key) not in (None, 0, 6):
                raise RemoteServerError(
                    f"This policy has {key.split('_')[0]} size {hello[key]}, but the SO-101 follower has 6 joints.",
                    400,
                )
        client = RemotePolicyClient(
            sock, hello, request.n_action_steps, request.prefetch, request.jpeg_quality
        )

        follower_id = setup_follower_calibration_file(request.follower_config)
        robot = SO101Follower(
            SO101FollowerConfig(
                port=request.follower_port,
                id=follower_id,
                cameras=_record._build_camera_configs(request.cameras, _record._platform_backend()),
                max_relative_target=request.max_relative_target,
            )
        )
        try:
            robot.bus.connect()
        except Exception as exc:
            raise RemoteServerError(
                f"Could not connect to the follower arm on {request.follower_port}. Make sure it's plugged in "
                "and powered on, then try again.",
                500,
            ) from exc
        # Feetech motors only store calibration registers with the torque off.
        robot.bus.disable_torque()
        robot.bus.write_calibration(robot.calibration)
        if robot.cameras:
            time.sleep(CAMERA_RELEASE_S)
            for cam in robot.cameras.values():
                cam.connect()
        robot.configure()
        return RemoteRun(request, robot, client, hello, address, lease, server=server, label=label)
    except Exception:
        _close(sock, lease)
        safe_disconnect_device(robot, logger)
        raise


def handle_start_remote_inference(request: RemoteInferenceRequest) -> dict[str, Any]:
    """Connect to the policy server and the arm synchronously, so failures reach the caller,
    then run the control loop in the background."""
    from . import rollout

    refused = rollout._claim_inference_slot()
    if refused is not None:
        return refused

    try:
        run = connect_run(request)
    except Exception as exc:
        rollout._release_inference_slot()
        if isinstance(exc, RemoteServerError):
            return {"success": False, "status_code": exc.status_code, "message": exc.message}
        logger.exception("Failed to start remote inference")
        return {"success": False, "status_code": 500, "message": f"Failed to start remote inference: {exc}"}

    rollout._begin_remote_run(run)
    run.start()
    logger.info(
        "Remote inference started: %s%s", run.policy_ref, " (interactive)" if request.interactive else ""
    )
    return {"success": True, "message": "Remote inference started", "policy_ref": run.policy_ref}


def stop_run(run: Any, timeout_s: float = 10.0) -> dict[str, Any]:
    """Ask the run (a RemoteRun, or a local session still setting up) to stop and wait for it to release
    the arm (after the way back to the start pose)."""
    run.request_stop()
    if not run.join(timeout_s):
        return {"success": False, "status_code": 409, "message": "The run has not released the arm yet"}
    return {"success": True, "message": "Inference stopped"}
