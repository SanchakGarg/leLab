"""Tests for lelab/policy_server.py, the standalone script that serves a policy
on the GPU server: it must stay importable without LeLab, speak the protocol,
and keep serving after a bad request."""

from __future__ import annotations

import ast
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "lelab" / "policy_server.py"


def test_script_imports_nothing_from_lelab_or_torch_at_module_level() -> None:
    """It is copied to the server on its own, where neither LeLab nor (at import) torch is needed."""
    tree = ast.parse(SCRIPT.read_text())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports would break the copied script"
            top_level.add((node.module or "").split(".")[0])
    assert top_level <= {
        "__future__",
        "argparse",
        "contextlib",
        "io",
        "json",
        "socket",
        "struct",
        "sys",
        "threading",
        "time",
        "collections",
        "pathlib",
        "typing",
        "numpy",
    }
    all_modules = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [None])
    }
    assert not any(m.startswith("lelab") for m in all_modules)


def test_framing_round_trip() -> None:
    from lelab.policy_server import recv_msg, send_msg

    a, b = socket.socketpair()
    with a, b:
        send_msg(a, {"type": "obs", "id": 3}, b"\x00\x01payload")
        header, payload = recv_msg(b)
    assert header == {"type": "obs", "id": 3}
    assert payload == b"\x00\x01payload"


def test_oversized_messages_are_rejected() -> None:
    import struct

    from lelab.policy_server import recv_msg

    a, b = socket.socketpair()
    with a, b:
        a.sendall(struct.pack(">II", 10, 1 << 30))
        with pytest.raises(ValueError, match="too large"):
            recv_msg(b)


def test_jpeg_round_trip_and_resize() -> None:
    from lelab.policy_server import decode_jpeg, encode_jpeg, resize_image

    img = np.full((48, 64, 3), 128, np.uint8)
    decoded = decode_jpeg(encode_jpeg(img))
    assert decoded.shape == (48, 64, 3)
    assert abs(int(decoded.mean()) - 128) <= 2
    assert resize_image(img, 24, 32).shape == (24, 32, 3)
    assert resize_image(img, 48, 64) is img


def test_camera_names_undo_the_training_rename() -> None:
    from lelab.policy_server import camera_features, image_shapes

    features = {
        "observation.state": SimpleNamespace(shape=(6,)),
        "observation.images.camera1": SimpleNamespace(shape=(3, 480, 640)),
        "observation.images.camera2": SimpleNamespace(shape=(3, 240, 320)),
        "observation.images.camera3": SimpleNamespace(shape=(3, 480, 640)),
        "observation.images.side": SimpleNamespace(shape=(3, 360, 480)),
    }
    renames = [
        {
            "observation.images.front": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
        }
    ]

    cameras = camera_features(features, renames)

    assert cameras == {
        "front": "observation.images.camera1",
        "wrist": "observation.images.camera2",
        "side": "observation.images.side",
    }
    assert image_shapes(features, cameras) == {
        "front": [480, 640, 3],
        "wrist": [240, 320, 3],
        "side": [360, 480, 3],
    }


def test_an_unfilled_base_camera_slot_is_not_asked_for() -> None:
    from lelab.policy_server import camera_features

    features = {f"observation.images.camera{i}": SimpleNamespace(shape=(3, 256, 256)) for i in (1, 2, 3)}

    # No renames: the names are the dataset's own, so every camera is needed.
    assert set(camera_features(features)) == {"camera1", "camera2", "camera3"}


def test_bind_first_free_skips_a_taken_port() -> None:
    from lelab.policy_server import bind_first_free

    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen(1)
    port = taken.getsockname()[1]
    try:
        srv, got = bind_first_free("127.0.0.1", port, span=20)
        srv.close()
    finally:
        taken.close()
    assert got != port
    assert port < got < port + 20


class FakeAgent:
    name = "fake (test)"
    cameras = ("front",)
    image_shapes = {"front": [48, 64, 3]}
    state_dim = 6
    action_dim = 6
    chunk_size = 5
    n_action_steps = 5

    def __init__(self) -> None:
        self.resets = 0
        self.seen: list[tuple[dict, np.ndarray, str]] = []

    def reset(self) -> None:
        self.resets += 1

    def predict_chunk(self, images: dict, state: np.ndarray, task: str) -> np.ndarray:
        if task == "explode":
            raise RuntimeError("CUDA out of memory")
        self.seen.append((images, state, task))
        return np.tile(state, (self.chunk_size, 1)) + np.arange(self.chunk_size)[:, None]


def _obs(task: str = "pick the cube", obs_id: int = 1) -> tuple[dict, bytes]:
    from lelab.policy_server import encode_jpeg

    jpg = encode_jpeg(np.zeros((48, 64, 3), np.uint8))
    header = {
        "type": "obs",
        "id": obs_id,
        "task": task,
        "state": [1, 2, 3, 4, 5, 6],
        "images": {"front": [0, len(jpg)]},
    }
    return header, jpg


def test_serve_client_answers_resets_and_survives_errors() -> None:
    from lelab.policy_server import make_hello, recv_msg, send_msg, serve_client

    agent = FakeAgent()
    server_end, client = socket.socketpair()
    thread = threading.Thread(target=serve_client, args=(agent, server_end, make_hello(agent)), daemon=True)
    thread.start()
    with client:
        hello, _ = recv_msg(client)
        assert hello["type"] == "hello"
        assert hello["cameras"] == ["front"]
        assert hello["image_shapes"] == {"front": [48, 64, 3]}
        assert hello["chunk_size"] == 5
        assert hello["protocol"] == 1

        send_msg(client, *_obs())
        chunk, body = recv_msg(client)
        actions = np.frombuffer(body, np.float32).reshape(chunk["shape"])
        assert chunk["type"] == "chunk" and chunk["id"] == 1
        assert actions.shape == (5, 6)
        assert actions[0].tolist() == [1, 2, 3, 4, 5, 6]

        send_msg(client, *_obs("explode", 2))
        error, _ = recv_msg(client)
        assert error == {"type": "error", "id": 2, "message": "RuntimeError: CUDA out of memory"}

        send_msg(client, {"type": "reset"})
        send_msg(client, *_obs(obs_id=3))
        assert recv_msg(client)[0]["id"] == 3
    thread.join(2)
    assert not thread.is_alive()
    assert agent.resets == 2  # on connect, and the explicit reset
    assert agent.seen[0][0]["front"].shape == (48, 64, 3)


def test_a_second_client_is_told_the_server_is_busy() -> None:
    from lelab.policy_server import bind_first_free, make_hello, recv_msg, serve

    agent = FakeAgent()
    srv, port = bind_first_free("127.0.0.1", 47000, span=500)
    stop = threading.Event()
    thread = threading.Thread(target=serve, args=(agent, srv, make_hello(agent), stop), daemon=True)
    thread.start()
    try:
        first = socket.create_connection(("127.0.0.1", port), timeout=5)
        assert recv_msg(first)[0]["type"] == "hello"
        second = socket.create_connection(("127.0.0.1", port), timeout=5)
        message, _ = recv_msg(second)
        assert message["type"] == "error"
        assert message["message"].startswith("busy")
        second.close()
        first.close()
    finally:
        stop.set()
        thread.join(3)
        srv.close()
