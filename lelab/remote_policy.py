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
their last target.

The connection to the server goes through the saved SSH session (a
direct-tcpip channel, like `ssh -N -L`) or straight over TCP on the same
network.

Safety: lerobot's `max_relative_target` caps every step; a chunk must have the
right shape and finite values; an error reply, a dropped connection, a chunk
that doesn't arrive in time or a camera that stalls stops the run. Then the
arm returns to where it started (unless a motor command itself failed) and is
released with the torque off, like a local LeLab run. Stop is checked every
tick and while waiting for a chunk.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time
from collections import deque
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from . import remote_ssh
from .camera_config import validate_camera_rotations
from .policy_server import encode_jpeg, recv_msg, send_msg
from .remote_ssh import RemoteServerError
from .utils.devices import safe_disconnect_device

logger = logging.getLogger(__name__)

HELLO_TIMEOUT_S = 10.0
# Give the browser's camera previews time to let go of the devices (as recording does).
CAMERA_RELEASE_S = 2.0
RETURN_DURATION_S = 3.0
RETURN_FPS = 50


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

    @field_validator("cameras")
    @classmethod
    def validate_rotations(cls, cameras: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return validate_camera_rotations(cameras)


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
            self.queue = deque(fresh)

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
        }

    def reset(self) -> None:
        """New episode: drop queued actions and clear the policy's state on the server."""
        if self.pending is not None:
            self.pending[0].join(5.0)
            self.pending = None
        self.queue.clear()
        self.step = 0
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
    ):
        self.request = request
        self.robot = robot
        self.client = client
        self.hello = hello
        self.address = address
        self.lease = lease
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

    @property
    def policy_ref(self) -> str:
        return f"{self.hello.get('name', 'remote policy')} @ {self.address}"

    def _observe(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        try:
            obs = self.robot.get_observation()
        except Exception as exc:
            raise _RobotIOError(exc, motors_ok=_motors_respond(self.robot)) from exc
        state = np.array([obs[f"{m}.pos"] for m in self.motors], np.float32)
        return state, {cam: obs[cam] for cam in self.client.cameras}

    def _send(self, action: np.ndarray) -> None:
        try:
            self.robot.send_action({f"{m}.pos": float(v) for m, v in zip(self.motors, action, strict=True)})
        except Exception as exc:
            raise _RobotIOError(exc, motors_ok=False) from exc

    def _loop(self) -> None:
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
            precise_sleep(period - (time.perf_counter() - tick))

    def run(self, initial: dict[str, float] | None) -> None:
        """Worker body: the loop, then the way back and the release, whatever happened."""
        motors_ok = True
        try:
            self._loop()
        except StopRequestedError:
            pass
        except _RobotIOError as exc:
            motors_ok = exc.motors_ok
            self.error = f"{type(exc.exc).__name__}: {exc.exc}"
            logger.error("Remote inference: robot I/O failed: %s", self.error)
        except RemotePolicyError as exc:
            self.error = str(exc)
            logger.error("Remote inference stopped: %s", exc)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Remote inference failed")
        finally:
            self.client.close()  # also unblocks a request still waiting for its answer
            _close(None, self.lease)
            if self.request.return_to_start and initial and motors_ok:
                self.phase = "returning to the start pose"
                self.returned_to_start = return_to(self.robot, initial)
            safe_disconnect_device(self.robot, logger)
            self.phase = "stopped"
            self.finished = True
            from . import rollout

            rollout._finish_remote_run(self)
            logger.info("Remote inference ended after %d steps (%s)", self.steps, self.error or "ok")

    def start(self) -> None:
        initial = None
        try:
            positions = self.robot.bus.sync_read("Present_Position")
            initial = {f"{m}.pos": float(v) for m, v in positions.items()}
        except Exception as exc:
            logger.warning("Could not read the start pose; the arm will not return to it: %s", exc)
        self.thread = threading.Thread(target=self.run, args=(initial,), name="remote-inference", daemon=True)
        self.thread.start()

    def status(self) -> dict[str, Any]:
        now = time.time()
        return {
            "mode": "remote",
            "started_at": self.started_at,
            "rollout_started_at": self.rollout_started_at,
            "elapsed_s": now - self.started_at,
            "rollout_elapsed_s": (now - self.rollout_started_at) if self.rollout_started_at else 0,
            "duration_s": self.request.duration_s,
            "policy_ref": self.policy_ref,
            "log_path": None,
            "remote": {
                "address": self.address,
                "policy": self.hello.get("name"),
                "phase": self.phase,
                "steps": self.steps,
                "fps": self.fps,
                **self.client.stats(),
            },
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


def _motors_respond(robot: Any) -> bool:
    try:
        robot.bus.sync_read("Present_Position")
        return True
    except Exception:
        return False


def return_to(
    robot: Any, target: dict[str, float], duration_s: float | None = None, fps: int | None = None
) -> bool:
    """Move smoothly from the present joint positions to `target` (what lerobot rollout does at the end)."""
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


def handle_start_remote_inference(request: RemoteInferenceRequest) -> dict[str, Any]:
    """Connect to the policy server and the arm synchronously, so failures reach the caller,
    then run the control loop in the background."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    from . import record as _record, rollout
    from .utils.config import setup_follower_calibration_file

    refused = rollout._claim_inference_slot()
    if refused is not None:
        return refused

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
    except Exception as exc:
        _close(sock, lease)
        safe_disconnect_device(robot, logger)
        rollout._release_inference_slot()
        if isinstance(exc, RemoteServerError):
            return {"success": False, "status_code": exc.status_code, "message": exc.message}
        logger.exception("Failed to start remote inference")
        return {"success": False, "status_code": 500, "message": f"Failed to start remote inference: {exc}"}

    run = RemoteRun(request, robot, client, hello, address, lease)
    rollout._begin_remote_run(run)
    run.start()
    logger.info("Remote inference started: %s", run.policy_ref)
    return {"success": True, "message": "Remote inference started", "policy_ref": run.policy_ref}


def stop_run(run: RemoteRun, timeout_s: float = 10.0) -> dict[str, Any]:
    """Ask the run to stop and wait for it to release the arm (after the way back to the start pose)."""
    run.stop_event.set()
    if run.thread is not None:
        run.thread.join(timeout_s)
        if run.thread.is_alive():
            return {"success": False, "status_code": 409, "message": "The run has not released the arm yet"}
    return {"success": True, "message": "Inference stopped"}
