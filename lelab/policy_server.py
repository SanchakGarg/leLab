#!/usr/bin/env python3
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

"""Serve a trained LeRobot policy over TCP, so LeLab can drive a real SO-101
from another machine while the model runs on a GPU server.

Run it where the GPU is (your server, or inside a Slurm job), with a Python
that has lerobot installed:

    python lelab_policy_server.py --policy outputs/act_pick/checkpoints/last/pretrained_model
    lelab-policy-server --policy ...        # the same, where LeLab itself is installed

It prints `serving on <node IP>:<port>`. In LeLab, open "Run model on server"
and enter that IP and port. LeLab reaches it through its SSH session to the
server (a tunnel, like `ssh -N -L port:<node IP>:port user@server`), or
directly when both are on the same network.

LeLab sends the arm's joint positions and camera frames; the server answers
with the whole predicted action chunk, which LeLab plays at the control rate
and refills before it runs out. Values pass through unchanged, so the policy
must have been trained on data in the same units LeLab records (SO-101 joints
in degrees, gripper 0-100).

This file is self-contained on purpose: only the standard library, numpy and
Pillow load at import, torch and lerobot only when a policy loads, so it can be
copied to the server on its own (LeLab's "Copy server script" does that).

Protocol (TCP, one client at a time). Every message is
    >II  header length, payload length;  JSON header;  payload bytes
server -> client  {"type": "hello", "protocol", "name", "cameras", "image_shapes", "state_dim", "action_dim",
                   "chunk_size", "n_action_steps", "fps"}  on connect
client -> server  {"type": "obs", "id", "task", "state": [...], "images": {camera: [offset, length]}}
                  + the JPEG frames, back to back
server -> client  {"type": "chunk", "id", "shape": [T, action_dim], "infer_ms"} + float32 actions
server -> client  {"type": "error", "id", "message"}  the request failed; the connection stays open
client -> server  {"type": "reset"}  new episode: clears the policy's internal state; no reply
A second client, while one is connected, gets an error message and is closed.
The messages match the SO-101 sim project's policy server, so either side works with the other.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import socket
import struct
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

# The version of LeLab's server-side files (this script and the Slurm kit). Bump it whenever any of them changes:
# `python lelab_policy_server.py --version` and the `# lelab-kit-version:` line in slurm/* show what a server has.
LELAB_KIT_VERSION = "2026.11.18-2"
PROTOCOL = 1
DEFAULT_PORT = 5555
FPS = 30
IMAGE_PREFIX = "observation.images."
# Camera slots of the pretrained bases the Slurm kit fine-tunes (smolvla_base, pi05_base). Training maps the
# dataset's cameras onto them; one left unmapped was never filled, and these policies run with it missing.
BASE_CAMERA_SLOTS = frozenset(
    {"camera1", "camera2", "camera3", "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
)
# A client that sends nothing for this long is dropped, so a crashed one can't hold the server.
IDLE_TIMEOUT_S = 120.0
_HEAD = struct.Struct(">II")
_MAX_HEADER = 1 << 20
_MAX_PAYLOAD = 64 << 20


# ---------------------------------------------------------------------------------------------------------- protocol
def send_msg(sock: Any, header: dict[str, Any], payload: bytes = b"") -> None:
    """Send one message. `sock` is a socket or anything with sendall (a paramiko Channel)."""
    h = json.dumps(header).encode()
    sock.sendall(_HEAD.pack(len(h), len(payload)) + h + payload)


def recv_exact(sock: Any, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: Any) -> tuple[dict[str, Any], bytes]:
    header_len, payload_len = _HEAD.unpack(recv_exact(sock, _HEAD.size))
    if header_len > _MAX_HEADER or payload_len > _MAX_PAYLOAD:
        raise ValueError(f"message too large ({header_len} + {payload_len} bytes)")
    header = json.loads(recv_exact(sock, header_len))
    if not isinstance(header, dict):
        raise ValueError("message header is not a JSON object")
    return header, (recv_exact(sock, payload_len) if payload_len else b"")


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(img).save(out, format="JPEG", quality=quality)
    return out.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))  # a writable copy (torch.from_numpy)


def resize_image(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """`img` at height x width (HWC uint8); unchanged if it already is."""
    if img.shape[0] == height and img.shape[1] == width:
        return img
    from PIL import Image

    return np.array(Image.fromarray(img).resize((width, height), Image.BILINEAR))


# ------------------------------------------------------------------------------------------------- checkpoint facts
def _inverse_renames(rename_maps: Iterable[Mapping[str, str]]) -> dict[str, str]:
    inverse: dict[str, str] = {}
    for rename_map in rename_maps:
        inverse.update({new: old for old, new in rename_map.items()})
    return inverse


def camera_features(
    input_features: Mapping[str, Any], rename_maps: Iterable[Mapping[str, str]] = ()
) -> dict[str, str]:
    """{camera name the client sends: policy input key}.

    Training may have renamed the cameras (smolvla_base wants camera1, camera2,
    ...); that rename is saved in the preprocessor, which applies it again, so
    clients keep sending the dataset's own camera names.
    """
    inverse = _inverse_renames(rename_maps)
    cameras: dict[str, str] = {}
    for key in input_features:
        if not key.startswith(IMAGE_PREFIX):
            continue
        if inverse and key not in inverse and key.removeprefix(IMAGE_PREFIX) in BASE_CAMERA_SLOTS:
            # A slot of the base model (e.g. smolvla_base's camera3) that no dataset camera was mapped onto:
            # training never filled it and the policy runs without it, so clients aren't asked for it.
            continue
        original = inverse.get(key, key)
        name = original.removeprefix(IMAGE_PREFIX) if original.startswith(IMAGE_PREFIX) else original
        cameras.setdefault(name, key)
    return cameras


def image_shapes(input_features: Mapping[str, Any], cameras: Mapping[str, str]) -> dict[str, list[int]]:
    """{camera: [height, width, 3]} the checkpoint was trained at (its features are channels-first).

    A camera mapped onto a base model's slot (smolvla_base's camera1, ...) is left out: that slot's shape is the
    base model's input size (256x256), not the camera's, and the policy resizes those images itself. Clients then
    capture at the camera's own resolution instead of asking a webcam for 256x256.
    """
    shapes = {}
    for name, key in cameras.items():
        slot = key.removeprefix(IMAGE_PREFIX)
        if slot in BASE_CAMERA_SLOTS and slot != name:
            continue
        shape = tuple(getattr(input_features[key], "shape", ()) or ())
        if len(shape) == 3:
            channels, height, width = shape
            shapes[name] = [int(height), int(width), int(channels)]
    return shapes


def pick_device(name: str = "auto") -> str:
    import torch

    if name != "auto":
        if name.startswith("cuda") and not torch.cuda.is_available():
            sys.exit(f"--device {name}, but this torch ({torch.__version__}) can't use CUDA")
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PolicyAgent:
    """A LeRobot checkpoint with its saved pre/post-processors (normalisation, renames)."""

    def __init__(self, path: str, device: str, n_action_steps: int | None = None):
        import torch

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device
        if n_action_steps:
            chunk = getattr(cfg, "chunk_size", None)
            if chunk and n_action_steps > chunk:
                sys.exit(f"--n-action-steps {n_action_steps} is more than this policy's chunk_size {chunk}")
            cfg.n_action_steps = n_action_steps
        if (Path(path) / "adapter_config.json").is_file():
            # A PEFT (LoRA) checkpoint holds only the adapter: load the base policy, apply it, merge it in.
            from peft import PeftConfig, PeftModel

            base = PeftConfig.from_pretrained(str(path)).base_model_name_or_path
            policy = get_policy_class(cfg.type).from_pretrained(base, config=cfg)
            self.policy = PeftModel.from_pretrained(policy, str(path)).merge_and_unload()
        else:
            self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}}
        )
        rename_maps = [getattr(step, "rename_map", None) or {} for step in getattr(self.pre, "steps", [])]
        features = getattr(cfg, "input_features", {}) or {}
        self.camera_keys = camera_features(features, rename_maps)
        self.cameras = tuple(self.camera_keys)
        self.image_shapes = image_shapes(features, self.camera_keys)
        state = features.get("observation.state")
        self.state_dim = int(state.shape[0]) if state is not None else 0
        action = (getattr(cfg, "output_features", {}) or {}).get("action")
        self.action_dim = int(action.shape[0]) if action is not None else 0
        self.n_action_steps = int(getattr(cfg, "n_action_steps", 0) or 1)
        self.chunk_size = int(getattr(cfg, "chunk_size", 0) or self.n_action_steps)
        self.name = f"{cfg.type} ({path})"

    def reset(self) -> None:
        self.policy.reset()

    def predict_chunk(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        """The whole predicted action chunk for one observation, (T, action_dim)."""
        torch = self.torch
        missing = [c for c in self.cameras if c not in images]
        if missing:
            raise ValueError(f"missing camera image(s): {', '.join(missing)}")
        obs = {}
        for name in self.cameras:
            img = images[name]
            if name in self.image_shapes:
                height, width, _ = self.image_shapes[name]
                img = resize_image(img, height, width)
            obs[f"{IMAGE_PREFIX}{name}"] = (
                torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
            )
        obs["observation.state"] = torch.from_numpy(np.asarray(state, np.float32)).unsqueeze(0)
        obs["task"] = [task]
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(obs))  # (1, T, D), normalised
        actions = self.post(chunk[0])  # the T steps go through as one batch
        return actions.detach().float().cpu().numpy()


# ------------------------------------------------------------------------------------------------------------- server
def make_hello(agent: Any, fps: int = FPS, n_action_steps: int | None = None) -> dict[str, Any]:
    return {
        "type": "hello",
        "protocol": PROTOCOL,
        "name": agent.name,
        "cameras": list(agent.cameras),
        "image_shapes": dict(agent.image_shapes),
        "state_dim": agent.state_dim,
        "action_dim": agent.action_dim,
        "chunk_size": agent.chunk_size,
        "n_action_steps": n_action_steps or agent.n_action_steps,
        "fps": fps,
    }


def _tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


def serve_client(agent: Any, conn: Any, hello: dict[str, Any], peer: str = "client") -> int:
    """Answer one client until it disconnects. Returns the number of chunks served."""
    served, times, t_log = 0, [], time.time()
    send_msg(conn, hello)
    agent.reset()
    while True:
        try:
            header, body = recv_msg(conn)
        except (ConnectionError, OSError, TimeoutError) as exc:
            print(f"{peer} disconnected ({exc}); {served} chunks served", flush=True)
            return served
        kind = header.get("type")
        if kind == "reset":
            agent.reset()
            continue
        if kind != "obs":
            send_msg(
                conn, {"type": "error", "id": header.get("id"), "message": f"unknown message type {kind!r}"}
            )
            continue
        try:
            images = {
                cam: decode_jpeg(body[off : off + length]) for cam, (off, length) in header["images"].items()
            }
            state = np.asarray(header["state"], np.float32)
            t = time.perf_counter()
            actions = agent.predict_chunk(images, state, str(header.get("task", "")))
            ms = 1000 * (time.perf_counter() - t)
            actions = np.ascontiguousarray(actions, np.float32)
        except Exception as exc:  # a bad request must not kill a server someone waited in a queue for
            print(f"{peer}: request failed: {type(exc).__name__}: {exc}", flush=True)
            send_msg(
                conn, {"type": "error", "id": header.get("id"), "message": f"{type(exc).__name__}: {exc}"}
            )
            continue
        send_msg(
            conn,
            {"type": "chunk", "id": header.get("id"), "shape": list(actions.shape), "infer_ms": round(ms, 1)},
            actions.tobytes(),
        )
        served += 1
        times.append(ms)
        if time.time() - t_log > 30:
            print(
                f"  {served} chunks served, inference median {np.median(times):.0f} ms, "
                f"max {max(times):.0f} ms (last 30 s)",
                flush=True,
            )
            times, t_log = [], time.time()


def bind_first_free(host: str, start: int, span: int = 200) -> tuple[socket.socket, int]:
    """A listening socket on the first free port from `start` up (GPU nodes are shared)."""
    for port in range(start, start + span):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            srv.bind((host, port))
        except OSError:
            srv.close()
            continue
        srv.listen(4)
        return srv, port
    raise RuntimeError(f"no free port in {start}-{start + span - 1}")


def node_ip() -> str:
    """This machine's address on its network (what LeLab should connect or tunnel to)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def serve(agent: Any, srv: socket.socket, hello: dict[str, Any], stop: threading.Event | None = None) -> None:
    """Accept clients on `srv`, one at a time; refuse others while one is connected."""
    busy = threading.Lock()
    srv.settimeout(0.5)

    def run(conn: socket.socket, peer: str) -> None:
        try:
            serve_client(agent, conn, hello, peer)
        except Exception as exc:
            print(f"{peer}: error {type(exc).__name__}: {exc}; dropping the connection", flush=True)
        finally:
            conn.close()
            busy.release()

    while stop is None or not stop.is_set():
        try:
            conn, addr = srv.accept()
        except TimeoutError:
            continue
        except OSError:
            if stop is not None and stop.is_set():
                break
            raise
        peer = f"{addr[0]}:{addr[1]}"
        _tune(conn)
        conn.settimeout(IDLE_TIMEOUT_S)
        if not busy.acquire(blocking=False):
            print(f"{peer} refused: another client is connected", flush=True)
            with contextlib.suppress(OSError):
                send_msg(
                    conn,
                    {"type": "error", "id": None, "message": "busy: another client is using this server"},
                )
            conn.close()
            continue
        print(f"{peer} connected", flush=True)
        threading.Thread(target=run, args=(conn, peer), name="policy-client", daemon=True).start()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"lelab_policy_server {LELAB_KIT_VERSION}")
    ap.add_argument(
        "--policy", required=True, help="pretrained_model folder from lerobot-train, or a Hub repo id"
    )
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on (default: all)")
    ap.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="first port to try; the next free one is used"
    )
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--fps", type=int, default=FPS, help="control rate the client should play actions at")
    ap.add_argument(
        "--n-action-steps",
        type=int,
        metavar="N",
        help="tell clients to play only the first N actions of each chunk (default: the checkpoint's)",
    )
    ap.add_argument("--no-warmup", action="store_true", help="skip the warm-up inferences")
    a = ap.parse_args(argv)

    device = pick_device(a.device)
    print(f"loading {a.policy} on {device}...", flush=True)
    agent = PolicyAgent(a.policy, device, a.n_action_steps)
    print(f"policy: {agent.name}", flush=True)
    cameras = ", ".join(
        f"{c} {'x'.join(map(str, agent.image_shapes[c][1::-1]))}" if c in agent.image_shapes else c
        for c in agent.cameras
    )
    print(
        f"cameras: {cameras or 'none'}; state {agent.state_dim}, action {agent.action_dim}, "
        f"chunk {agent.chunk_size}, plays {a.n_action_steps or agent.n_action_steps} per chunk",
        flush=True,
    )
    if not a.no_warmup:
        # The first inferences compile kernels and are much slower than the rest.
        blank = {
            c: np.zeros(tuple(agent.image_shapes.get(c, [480, 640, 3])), np.uint8) for c in agent.cameras
        }
        for i in range(3):
            t = time.perf_counter()
            out = agent.predict_chunk(blank, np.zeros(agent.state_dim, np.float32), "warm up")
            print(
                f"warm-up {i + 1}: chunk {tuple(out.shape)} in {1000 * (time.perf_counter() - t):.0f} ms",
                flush=True,
            )
        agent.reset()

    srv, port = bind_first_free(a.host, a.port)
    if port != a.port:
        print(f"port {a.port} is in use on this node; serving on {port} instead", flush=True)
    ip = node_ip()
    print(f"=== lelab_policy_server {LELAB_KIT_VERSION}: serving on {ip}:{port} ===", flush=True)
    print(f"In LeLab: Run model on server -> policy host {ip}, port {port}", flush=True)
    try:
        serve(agent, srv, make_hello(agent, a.fps, a.n_action_steps))
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        srv.close()


if __name__ == "__main__":
    main()
