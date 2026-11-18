"""Tests for lelab.local_policy: interactive sessions with a model on this machine. The policy server is a
real subprocess (a tiny fake model served by lelab/policy_server.py's own loop); the follower is the fake
from test_remote_policy, put in through a patched connect_run."""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from tests.test_remote_policy import FakeFollower, _wait_for

FAKE_SERVER = textwrap.dedent(
    """
    import numpy as np
    from lelab.policy_server import bind_first_free, exit_when_stdin_closes, make_hello, serve

    class Model:
        name = "fake act (outputs/fake/checkpoints/last/pretrained_model)"
        cameras = ("front",)
        image_shapes = {"front": [48, 64, 3]}
        state_dim = action_dim = 6
        chunk_size = n_action_steps = 10

        def reset(self):
            pass

        def predict_chunk(self, images, state, task):
            return np.tile(state, (10, 1)) + np.arange(10, dtype=np.float32)[:, None]

    exit_when_stdin_closes()
    print("loading the model...", flush=True)
    srv, port = bind_first_free("127.0.0.1", 49000, 500)
    print(f"=== lelab_policy_server test: serving on 10.1.2.3:{port} ===", flush=True)
    serve(Model(), srv, make_hello(Model()))
    """
)
FAILING_SERVER = "print('Traceback (most recent call last):'); print('FileNotFoundError: config.json not found'); raise SystemExit(1)"
SLOW_SERVER = "import time; print('loading...', flush=True); time.sleep(60)"
CAMERAS = {"front": {"type": "opencv", "camera_index": 0}}


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home: Path) -> None:
    from lelab import local_policy, remote_policy, rollout

    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "_inference_proc", None)
    monkeypatch.setattr(rollout, "_inference_mode", "local")
    monkeypatch.setattr(rollout, "_remote_run", None)
    monkeypatch.setattr(remote_policy, "RETURN_DURATION_S", 0.1)
    monkeypatch.setattr(local_policy, "CLOSE_GRACE_S", 1.0)

    def connect_run(request: Any, server: Any = None, label: str | None = None) -> Any:
        sock, lease, address = remote_policy.open_policy_socket(request)
        hello = remote_policy.read_hello(sock)
        client = remote_policy.RemotePolicyClient(sock, hello, prefetch=3)
        return remote_policy.RemoteRun(
            request, FakeFollower(), client, hello, address, lease, server=server, label=label
        )

    monkeypatch.setattr(remote_policy, "connect_run", connect_run)


def _serve_with(monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    from lelab import local_policy

    monkeypatch.setattr(
        local_policy.LocalPolicyServer, "command", lambda self: [sys.executable, "-c", script]
    )


def _start(tmp_path: Path, **overrides: Any) -> dict:
    from lelab.rollout import InferenceRequest, handle_start_inference

    return handle_start_inference(
        InferenceRequest(
            follower_port="/dev/follower",
            follower_config="follower",
            policy_ref=str(tmp_path),
            cameras=CAMERAS,
            **{"interactive": True, **overrides},
        )
    )


def _status() -> dict:
    from lelab import rollout

    return rollout.handle_inference_status()


def test_the_server_runs_on_this_machine_and_exits_with_lelab() -> None:
    from lelab.local_policy import LocalPolicyServer

    cmd = LocalPolicyServer("/models/act", Path("x.log")).command()
    assert cmd[:3] == [sys.executable, "-m", "lelab.policy_server"]
    assert cmd[cmd.index("--policy") + 1] == "/models/act"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert "--watch-stdin" in cmd


def test_a_local_session_loads_the_model_then_takes_commands(monkeypatch, tmp_path) -> None:
    from lelab import rollout
    from lelab.rollout import InferenceCommandRequest, handle_inference_command

    _serve_with(monkeypatch, FAKE_SERVER)
    started = _start(tmp_path)
    assert started["success"] is True
    assert Path(started["log_path"]).parent == Path(rollout.config.INFERENCE_LOGS_PATH)

    _wait_for(lambda: (_status().get("interactive") or {}).get("state") == "ready", timeout=30)
    status = _status()
    assert status["mode"] == "local"
    assert status["policy_ref"] == str(tmp_path)
    assert status["duration_s"] is None
    run = rollout._remote_run
    server = run.server

    assert handle_inference_command(InferenceCommandRequest(text="pick the cube"))["success"] is True
    _wait_for(lambda: run.steps >= 5)
    assert rollout.handle_stop_inference()["success"] is True

    result = _status()
    assert result["outcome"] == "ok"
    assert result["mode"] == "local"
    assert result["interactive"]["commands"][0]["text"] == "pick the cube"
    assert server.proc.poll() is not None  # the policy server went with the session
    assert "serving on" in Path(result["log_path"]).read_text()
    assert rollout.inference_active is False


def test_commands_wait_until_the_model_is_loaded(monkeypatch, tmp_path) -> None:
    from lelab import rollout
    from lelab.rollout import InferenceCommandRequest, handle_inference_command

    _serve_with(monkeypatch, SLOW_SERVER)
    assert _start(tmp_path)["success"] is True
    status = _status()
    assert status["inference_active"] is True
    assert status["rollout_started_at"] is None
    assert status["interactive"]["state"] == "setting up"
    refused = handle_inference_command(InferenceCommandRequest(text="pick"))
    assert refused["status_code"] == 409 and "still starting" in refused["message"]

    server = rollout._remote_run.server
    began = time.time()
    assert rollout.handle_stop_inference()["success"] is True
    assert time.time() - began < 8
    assert server.proc.poll() is not None
    result = _status()
    assert result["exited"] is True and result["outcome"] == "ok"
    assert _status()["inference_active"] is False


def test_a_model_that_fails_to_load_reports_its_error_once(monkeypatch, tmp_path) -> None:
    from lelab import rollout

    _serve_with(monkeypatch, FAILING_SERVER)
    assert _start(tmp_path)["success"] is True
    _wait_for(lambda: rollout._remote_run is not None and rollout._remote_run.finished, timeout=15)

    result = _status()
    assert result["inference_active"] is False
    assert result["outcome"] == "failed"
    assert "config.json not found" in result["error"]
    assert _status()["inference_active"] is False  # reported once, then idle
    assert rollout.inference_active is False


def test_a_bad_video_camera_fails_the_start(monkeypatch, tmp_path) -> None:
    from lelab import rollout

    _serve_with(monkeypatch, FAKE_SERVER)
    result = _start(tmp_path, record_video=True, video_camera="side")
    assert result["status_code"] == 400
    assert "no camera 'side'" in result["message"]
    assert rollout.inference_active is False


def test_a_plain_run_with_chunk_blending_goes_through_the_local_policy_server(monkeypatch, tmp_path) -> None:
    from lelab import rollout

    _serve_with(monkeypatch, FAKE_SERVER)
    started = _start(
        tmp_path, interactive=False, chunk_blending=True, blend_steps=4, task="pick", duration_s=1
    )
    assert started["success"] is True and started["message"] == "Run with chunk blending starting"
    status = _status()
    assert status["interactive"] is None and status["duration_s"] == 1
    _wait_for(lambda: rollout._remote_run is not None and rollout._remote_run.finished, timeout=30)

    result = _status()
    assert result["outcome"] == "ok"
    assert result["mode"] == "local"
    assert result["interactive"] is None
    assert result["remote"]["blend_steps"] == 4
    assert result["remote"]["blended"] > 0


def test_blending_switched_while_the_model_loads_applies_to_the_session(monkeypatch, tmp_path) -> None:
    from lelab import rollout
    from lelab.rollout import InferenceSettingsRequest, handle_inference_settings

    _serve_with(monkeypatch, FAKE_SERVER)
    assert _start(tmp_path)["success"] is True
    assert (
        handle_inference_settings(InferenceSettingsRequest(chunk_blending=True, blend_steps=7))["success"]
        is True
    )
    _wait_for(lambda: (_status().get("interactive") or {}).get("state") == "ready", timeout=30)
    session = _status()["interactive"]
    assert (session["chunk_blending"], session["blend_steps"]) == (True, 7)
    assert rollout._remote_run.client.blend_steps == 7
    assert rollout.handle_stop_inference()["success"] is True


def test_settings_the_loop_does_not_take_are_refused(monkeypatch, tmp_path) -> None:
    from lelab import rollout

    _serve_with(monkeypatch, FAKE_SERVER)
    result = _start(tmp_path, interactive=False, chunk_blending=True, duration_s=7200)
    assert result["status_code"] == 400
    assert "duration_s" in result["message"]
    assert rollout.inference_active is False


def test_the_interactive_request_bounds_the_time_per_command() -> None:
    from pydantic import ValidationError

    from lelab.rollout import InferenceRequest

    base = {"follower_port": "p", "follower_config": "c", "policy_ref": "r"}
    request = InferenceRequest(**base)
    assert (request.interactive, request.command_time_s, request.return_between_commands) == (False, 60, True)
    assert request.auto_complete is True and request.record_video is False
    assert (request.chunk_blending, request.blend_steps) == (False, 10)
    with pytest.raises(ValidationError):
        InferenceRequest(**base, command_time_s=4)


def test_the_command_routes(monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from lelab.server import app

    client = TestClient(app)
    response = client.post("/inference-command", json={"text": "pick"})
    assert response.status_code == 409
    assert response.json()["detail"] == "No interactive session is running."
    assert client.post("/inference-command", json={"text": ""}).status_code == 422
    assert client.post("/inference-command/end", json={"completed": True}).status_code == 409
    assert client.post("/inference-settings", json={"chunk_blending": True}).status_code == 409
    assert (
        client.post("/inference-settings", json={"chunk_blending": True, "blend_steps": 0}).status_code == 422
    )
