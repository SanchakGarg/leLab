"""Tests for lelab.remote_policy: the chunk-queue client, probing a policy
server, and remote runs on a fake follower (stop, faults, the way back to the
start pose, the shared inference slot). Policy servers here are threads on
127.0.0.1 running lelab/policy_server.py's own serve loop with a fake model."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@pytest.fixture(autouse=True)
def _reset_rollout_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab import remote_policy, rollout

    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_proc", None)
    monkeypatch.setattr(rollout, "_inference_mode", "local")
    monkeypatch.setattr(rollout, "_remote_run", None)
    monkeypatch.setattr(remote_policy, "RETURN_DURATION_S", 0.1)
    monkeypatch.setattr(remote_policy, "CAMERA_RELEASE_S", 0.0)


class FakeModel:
    """What lelab/policy_server.py serves: action = state + step index, or an error on request."""

    name = "fake act"
    cameras = ("front",)
    image_shapes = {"front": [48, 64, 3]}
    state_dim = 6
    action_dim = 6
    chunk_size = 10
    n_action_steps = 10

    def __init__(self, fail_after: int | None = None, delay_s: float = 0.0) -> None:
        self.calls = 0
        self.fail_after = fail_after
        self.delay_s = delay_s

    def reset(self) -> None:
        pass

    def predict_chunk(self, images: dict, state: np.ndarray, task: str) -> np.ndarray:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("CUDA out of memory")
        time.sleep(self.delay_s)
        return np.tile(state, (self.chunk_size, 1)) + np.arange(self.chunk_size, dtype=np.float32)[:, None]


@pytest.fixture
def policy_server() -> Iterator[Any]:
    """Start fake policy servers on free local ports; yields a factory returning the port."""
    from lelab.policy_server import bind_first_free, make_hello, serve

    started = []

    def start(model: FakeModel | None = None) -> int:
        model = model or FakeModel()
        srv, port = bind_first_free("127.0.0.1", 48000, span=1000)
        stop = threading.Event()
        thread = threading.Thread(target=serve, args=(model, srv, make_hello(model), stop), daemon=True)
        thread.start()
        started.append((srv, stop, thread))
        return port

    yield start
    for srv, stop, thread in started:
        stop.set()
        thread.join(2)
        srv.close()


# ---------------------------------------------------------------------------------------------- client
HELLO = {"type": "hello", "cameras": ["front"], "chunk_size": 10, "n_action_steps": 10, "fps": 30}
IMAGES = {"front": np.zeros((48, 64, 3), np.uint8)}


class ScriptedServer:
    """The server end of a socketpair that answers each obs only when told to."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.requests: list[dict] = []
        self.release = threading.Semaphore(0)
        self.reply: Any = None  # callable(request) -> (header, payload)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        from lelab.policy_server import recv_msg, send_msg

        try:
            while True:
                header, _ = recv_msg(self.sock)
                if header["type"] != "obs":
                    continue
                self.requests.append(header)
                self.release.acquire()
                send_msg(self.sock, *self.reply(header))
        except (ConnectionError, OSError):
            pass


def _chunk(header: dict, value_base: float | None = None) -> tuple[dict, bytes]:
    base = 100.0 * header["id"] if value_base is None else value_base
    actions = np.repeat((base + np.arange(10, dtype=np.float32))[:, None], 6, axis=1)
    return {"type": "chunk", "id": header["id"], "shape": [10, 6], "infer_ms": 5.0}, actions.tobytes()


def _client(prefetch: int = 3) -> tuple[Any, ScriptedServer]:
    from lelab.remote_policy import RemotePolicyClient

    a, b = socket.socketpair()
    server = ScriptedServer(b)
    server.reply = _chunk
    return RemotePolicyClient(a, HELLO, prefetch=prefetch), server


def test_client_prefetches_and_skips_actions_already_played() -> None:
    client, server = _client(prefetch=3)
    server.release.release()
    played = [client.next_action(IMAGES, np.zeros(6), "t")[0] for _ in range(8)]
    assert played == [100, 101, 102, 103, 104, 105, 106, 107]
    # The 8th call saw 3 actions left and sent the observation of step 7 for the next chunk.
    deadline = time.time() + 2
    while len(server.requests) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert len(server.requests) == 2
    # One more action plays from the queue while the server thinks...
    assert client.next_action(IMAGES, np.zeros(6), "t")[0] == 108
    server.release.release()
    time.sleep(0.1)
    # ...so the new chunk (predicted for step 7) is entered 2 actions in, lined up with the present.
    assert client.next_action(IMAGES, np.zeros(6), "t")[0] == 202
    assert client.stats()["chunks"] == 2


def test_client_raises_on_an_error_reply() -> None:
    from lelab.remote_policy import RemotePolicyError

    client, server = _client()
    server.reply = lambda h: ({"type": "error", "id": h["id"], "message": "RuntimeError: boom"}, b"")
    server.release.release()
    with pytest.raises(RemotePolicyError, match="boom"):
        client.next_action(IMAGES, np.zeros(6), "t")


@pytest.mark.parametrize(
    ("header", "payload", "match"),
    [
        ({"type": "chunk", "id": 1, "shape": [10, 5]}, np.zeros((10, 5), np.float32).tobytes(), "shape"),
        ({"type": "chunk", "id": 1, "shape": [10, 6]}, np.zeros(10, np.float32).tobytes(), "truncated"),
        ({"type": "chunk", "id": 1, "shape": [2, 6]}, np.full((2, 6), np.nan, np.float32).tobytes(), "NaN"),
        ({"type": "hello"}, b"", "unexpected"),
    ],
)
def test_client_rejects_bad_chunks(header: dict, payload: bytes, match: str) -> None:
    from lelab.remote_policy import RemotePolicyError

    client, server = _client()
    server.reply = lambda h: (header, payload)
    server.release.release()
    with pytest.raises(RemotePolicyError, match=match):
        client.next_action(IMAGES, np.zeros(6), "t")


def test_client_times_out_when_no_chunk_comes() -> None:
    from lelab.remote_policy import RemotePolicyError

    client, _server = _client()
    start = time.perf_counter()
    with pytest.raises(RemotePolicyError, match="no action chunk"):
        client.next_action(IMAGES, np.zeros(6), "t", timeout_s=0.3)
    assert time.perf_counter() - start < 1.5
    client.close()


def test_stop_interrupts_the_wait() -> None:
    from lelab.remote_policy import StopRequestedError

    client, _server = _client()
    stop = threading.Event()
    threading.Timer(0.1, stop.set).start()
    start = time.perf_counter()
    with pytest.raises(StopRequestedError):
        client.next_action(IMAGES, np.zeros(6), "t", stop=stop, timeout_s=30)
    assert time.perf_counter() - start < 1.0
    client.close()


# ----------------------------------------------------------------------------------------------- probe
def test_probe_direct(policy_server) -> None:
    from lelab.remote_policy import PolicyConnection, probe

    port = policy_server()
    hello = probe(PolicyConnection(connection="direct", policy_host="127.0.0.1", policy_port=port))

    assert hello["cameras"] == ["front"]
    assert hello["image_shapes"] == {"front": [48, 64, 3]}
    assert hello["address"] == f"127.0.0.1:{port}"


def test_probe_reports_nothing_listening() -> None:
    from lelab.remote_policy import PolicyConnection, probe
    from lelab.remote_ssh import RemoteServerError

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(RemoteServerError) as exc:
        probe(PolicyConnection(connection="direct", policy_host="127.0.0.1", policy_port=port))
    assert exc.value.status_code == 502


def test_probe_reports_a_busy_server(policy_server) -> None:
    from lelab.remote_policy import PolicyConnection, probe
    from lelab.remote_ssh import RemoteServerError

    port = policy_server()
    first = socket.create_connection(("127.0.0.1", port))
    try:
        time.sleep(0.1)
        with pytest.raises(RemoteServerError) as exc:
            probe(PolicyConnection(connection="direct", policy_host="127.0.0.1", policy_port=port))
        assert exc.value.status_code == 409
        assert "busy" in exc.value.message
    finally:
        first.close()


def test_probe_through_the_tunnel_uses_the_ssh_session(
    policy_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The channel paramiko opens is socket-like; here a plain socket stands in for it."""
    from lelab import remote_ssh
    from lelab.remote_policy import PolicyConnection, probe

    port = policy_server()
    session = type("S", (), {"users": 0, "name": "cluster"})()
    opened = []

    def open_channel(sess, host, p, timeout):
        opened.append((host, p))
        return socket.create_connection(("127.0.0.1", port))

    monkeypatch.setattr(remote_ssh, "ensure_connected", lambda name: session)
    monkeypatch.setattr(remote_ssh, "open_policy_channel", open_channel)

    hello = probe(PolicyConnection(server="cluster", policy_host="10.2.0.7", policy_port=port))

    assert opened == [("10.2.0.7", port)]
    assert hello["address"] == f"10.2.0.7:{port} via cluster"
    assert session.users == 0


def test_probe_route(policy_server, client: TestClient) -> None:
    port = policy_server()
    response = client.post(
        "/remote-policy/probe", json={"connection": "direct", "policy_host": "127.0.0.1", "policy_port": port}
    )
    assert response.status_code == 200
    assert response.json()["chunk_size"] == 10


# ------------------------------------------------------------------------------------------------- runs
class FakeBus:
    def __init__(self, robot: FakeFollower) -> None:
        self.robot = robot
        self.motors = dict.fromkeys(MOTORS)

    def sync_read(self, register: str, num_retry: int = 0) -> dict[str, float]:
        if self.robot.motors_dead:
            raise ConnectionError("motor 6 not responding")
        return dict(zip(MOTORS, self.robot.position, strict=True))


class FakeFollower:
    def __init__(self) -> None:
        self.position = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        self.sent: list[dict[str, float]] = []
        self.disconnected = False
        self.motors_dead = False
        self.fail_send = False
        self.bus = FakeBus(self)

    def get_observation(self) -> dict[str, Any]:
        obs = dict(zip([f"{m}.pos" for m in MOTORS], self.position, strict=True))
        obs["front"] = np.zeros((48, 64, 3), np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if self.fail_send:
            raise ConnectionError("Overload error on motor 6")
        self.sent.append(action)
        self.position = [action[f"{m}.pos"] for m in MOTORS]
        return action

    def disconnect(self) -> None:
        self.disconnected = True


class BusFollower(FakeFollower):
    """A follower whose reads and writes go through its motor bus, which can glitch (as over a noisy USB link)."""

    def __init__(self) -> None:
        from tests.test_bus_guard import GlitchyBus

        super().__init__()
        self.bus = GlitchyBus()

    def get_observation(self) -> dict[str, Any]:
        obs = {f"{m}.pos": v for m, v in self.bus.sync_read("Present_Position").items()}
        obs["front"] = np.zeros((48, 64, 3), np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.sent.append(action)
        self.bus.sync_write("Goal_Position", {k.removesuffix(".pos"): v for k, v in action.items()})
        return action


def _start_run(port: int, robot: Any = None, **overrides: Any) -> tuple[Any, FakeFollower]:
    from lelab import rollout
    from lelab.remote_policy import (
        RemoteInferenceRequest,
        RemotePolicyClient,
        RemoteRun,
        open_policy_socket,
        read_hello,
    )

    request = RemoteInferenceRequest(
        follower_port="/dev/follower",
        follower_config="follower",
        connection="direct",
        policy_host="127.0.0.1",
        policy_port=port,
        cameras={"front": {"type": "opencv", "camera_index": 0}},
        task="pick the cube",
        **{"fps": 100, "duration_s": 30, **overrides},
    )
    sock, lease, address = open_policy_socket(request)
    hello = read_hello(sock)
    robot = robot or FakeFollower()
    assert rollout._claim_inference_slot() is None
    run = RemoteRun(request, robot, RemotePolicyClient(sock, hello, prefetch=3), hello, address, lease)
    rollout._begin_remote_run(run)
    run.start()
    return run, robot


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        time.sleep(0.01)


def test_run_plays_actions_in_motor_order_and_stops_cleanly(policy_server) -> None:
    from lelab import rollout

    port = policy_server()
    run, robot = _start_run(port)
    _wait_for(lambda: run.steps >= 15)

    status = rollout.handle_inference_status()
    assert status["inference_active"] is True
    assert status["mode"] == "remote"
    assert status["remote"]["chunks"] >= 1
    assert status["rollout_started_at"] is not None

    assert rollout.handle_stop_inference()["success"] is True
    assert robot.disconnected
    # Every command names the six motors, and the last moves end where the run began.
    assert list(robot.sent[0]) == [f"{m}.pos" for m in MOTORS]
    assert robot.sent[-1] == pytest.approx(
        dict(zip([f"{m}.pos" for m in MOTORS], [1, 2, 3, 4, 5, 6], strict=True))
    )
    assert rollout.inference_active is False

    result = rollout.handle_inference_status()
    assert result["exited"] is True
    assert result["outcome"] == "ok"
    assert result["returned_to_start"] is True
    # Reported once; then idle.
    assert rollout.handle_inference_status()["inference_active"] is False
    assert rollout.handle_stop_inference()["success"] is False


def test_run_ends_after_its_duration(policy_server) -> None:
    from lelab import rollout

    run, robot = _start_run(policy_server(), duration_s=1)
    _wait_for(lambda: run.finished, timeout=6)
    assert rollout.handle_inference_status()["outcome"] == "ok"
    assert robot.disconnected


def test_a_server_error_stops_the_run_and_returns_home(policy_server) -> None:
    from lelab import rollout

    run, robot = _start_run(policy_server(FakeModel(fail_after=2)))
    _wait_for(lambda: run.finished)

    result = rollout.handle_inference_status()
    assert result["outcome"] == "failed"
    assert result["exit_code"] == 1
    assert "CUDA out of memory" in result["error"]
    assert result["hint"]
    assert result["returned_to_start"] is True
    assert robot.disconnected


def test_a_failed_motor_command_skips_the_way_back(policy_server) -> None:
    from lelab import rollout

    run, robot = _start_run(policy_server())
    _wait_for(lambda: run.steps >= 3)
    robot.fail_send = True
    _wait_for(lambda: run.finished)

    result = rollout.handle_inference_status()
    assert result["outcome"] == "failed"
    assert "Overload" in result["error"]
    assert result["returned_to_start"] is None
    assert robot.disconnected


def test_a_few_failed_bus_transfers_do_not_stop_the_run(policy_server) -> None:
    from lelab import rollout

    run, robot = _start_run(policy_server(), robot=BusFollower())
    _wait_for(lambda: run.steps >= 3)
    robot.bus.fail_reads = 2
    robot.bus.fail_writes = 2
    _wait_for(lambda: run.guard.skipped >= 4)
    steps = run.steps
    _wait_for(lambda: run.steps >= steps + 5)
    assert rollout.handle_inference_status()["remote"]["arm_skipped"] >= 4

    assert rollout.handle_stop_inference()["success"] is True
    result = rollout.handle_inference_status()
    assert result["outcome"] == "ok"
    assert result["returned_to_start"] is True
    assert "sync_read" not in vars(robot.bus)


def test_a_silent_follower_stops_the_run_without_moving_it(policy_server, monkeypatch) -> None:
    from lelab import rollout
    from lelab.utils import bus_guard

    monkeypatch.setattr(bus_guard, "MAX_OUTAGE_S", 0.3)
    run, robot = _start_run(policy_server(), robot=BusFollower())
    _wait_for(lambda: run.steps >= 3)
    robot.bus.dead = True
    _wait_for(lambda: run.finished)

    result = rollout.handle_inference_status()
    assert result["outcome"] == "failed"
    assert result["error"].startswith("Lost contact with the follower arm")
    assert result["returned_to_start"] is None
    assert robot.disconnected


def test_a_run_can_record_a_video(policy_server, tmp_lerobot_home) -> None:
    from pathlib import Path

    from lelab import rollout

    run, robot = _start_run(policy_server(), record_video=True)
    _wait_for(lambda: run.steps >= 10)
    assert rollout.handle_inference_status()["video"]["recording"] is True

    assert rollout.handle_stop_inference()["success"] is True
    video = rollout.handle_inference_status()["video"]
    assert video["camera"] == "front"
    assert video["corner_camera"] is None
    assert video["error"] is None
    assert Path(video["path"]).parent == tmp_lerobot_home / "run_videos"
    assert Path(video["path"]).is_file()
    assert video["name"].endswith("_fake_act.mp4")


def test_a_slow_server_times_out(policy_server) -> None:
    from lelab import rollout

    run, robot = _start_run(policy_server(FakeModel(delay_s=1.0)), first_chunk_timeout_s=0.3)
    _wait_for(lambda: run.finished)
    result = rollout.handle_inference_status()
    assert "no action chunk" in result["error"]
    assert robot.disconnected


def test_other_modes_refuse_to_start_during_a_remote_run(policy_server) -> None:
    from lelab import rollout
    from lelab.replay import ReplayRequest, handle_start_replay

    run, _robot = _start_run(policy_server())
    try:
        result = handle_start_replay(
            ReplayRequest(
                follower_port="/dev/f", follower_config="f", dataset_repo_id="local/x", episode_index=0
            )
        )
        assert result["success"] is False
        assert "Inference" in result["message"]
        assert rollout._claim_inference_slot()["status_code"] == 409
    finally:
        rollout.handle_stop_inference()


# ------------------------------------------------------------------------------------------------ start
def _request(port: int, **overrides: Any) -> Any:
    from lelab.remote_policy import RemoteInferenceRequest

    body = {
        "follower_port": "/dev/follower",
        "follower_config": "follower",
        "connection": "direct",
        "policy_host": "127.0.0.1",
        "policy_port": port,
        "cameras": {"front": {"type": "opencv", "camera_index": 0}},
        **overrides,
    }
    return RemoteInferenceRequest(**body)


def test_start_refuses_while_teleoperating(policy_server, monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab import teleoperate
    from lelab.remote_policy import handle_start_remote_inference

    monkeypatch.setattr(teleoperate, "teleoperation_active", True)
    result = handle_start_remote_inference(_request(policy_server()))
    assert result["status_code"] == 409
    assert "Teleoperation" in result["message"]


def test_start_needs_the_policy_cameras(policy_server) -> None:
    from lelab import rollout
    from lelab.remote_policy import handle_start_remote_inference

    result = handle_start_remote_inference(_request(policy_server(), cameras={}))
    assert result["status_code"] == 400
    assert "front" in result["message"]
    assert rollout.inference_active is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"video_camera": "side"}, "no camera 'side'"),
        ({"video_camera": "front", "video_corner_camera": "wrist"}, "no camera 'wrist'"),
    ],
)
def test_start_checks_the_video_cameras(policy_server, overrides: dict, match: str) -> None:
    from lelab import rollout
    from lelab.remote_policy import handle_start_remote_inference

    result = handle_start_remote_inference(_request(policy_server(), record_video=True, **overrides))
    assert result["status_code"] == 400
    assert match in result["message"]
    assert rollout.inference_active is False


def test_video_layout_defaults_to_the_wrist_camera_in_the_corner() -> None:
    from lelab.remote_policy import video_layout

    cams = {"wrist": {"type": "opencv", "camera_index": 1}, "front": {"type": "opencv", "camera_index": 0}}
    assert video_layout(_request(5555, cameras=cams, record_video=True)) == ("front", "wrist")
    assert video_layout(
        _request(5555, cameras=cams, record_video=True, video_camera="wrist", video_corner_camera="wrist")
    ) == ("wrist", None)


def test_start_reports_an_unreachable_server() -> None:
    from lelab import rollout
    from lelab.remote_policy import handle_start_remote_inference

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    result = handle_start_remote_inference(_request(port))
    assert result["status_code"] == 502
    assert rollout.inference_active is False


def test_start_releases_the_slot_when_the_arm_is_missing(
    policy_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lerobot.robots.so_follower as so_follower
    from lelab import rollout
    from lelab.remote_policy import PolicyConnection, handle_start_remote_inference, probe

    class _Bus:
        def connect(self) -> None:
            raise RuntimeError("serial port unavailable")

    class _Follower:
        def __init__(self, config) -> None:
            self.config = config
            self.bus = _Bus()
            self.cameras = {}

        def disconnect(self) -> None:
            pass

    model = FakeModel()
    model.cameras, model.image_shapes = (), {}
    port = policy_server(model)
    monkeypatch.setattr(so_follower, "SO101Follower", _Follower)
    monkeypatch.setattr("lelab.utils.config.setup_follower_calibration_file", lambda config: "follower")

    result = handle_start_remote_inference(_request(port, cameras={}, max_relative_target=5))

    assert result["status_code"] == 500
    assert "/dev/follower" in result["message"]
    assert rollout.inference_active is False
    # The connection to the policy server was closed, so it takes the next client.
    time.sleep(0.2)
    assert (
        probe(PolicyConnection(connection="direct", policy_host="127.0.0.1", policy_port=port))["cameras"]
        == []
    )


def test_request_defaults_are_safe() -> None:
    req = _request(5555)
    assert req.max_relative_target == 20.0
    assert req.return_to_start is True
    assert req.prefetch == 10
    with pytest.raises(ValueError):
        _request(5555, cameras={"front": {"type": "opencv", "rotation": 45}})
