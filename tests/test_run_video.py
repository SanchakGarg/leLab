"""Tests for lelab.run_video: which camera goes in the corner, the composed
frame, the mp4 itself, and a remote run that records one."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("cameras", "layout"),
    [
        ([], (None, None)),
        (["front"], ("front", None)),
        (["wrist"], ("wrist", None)),
        (["front", "wrist"], ("front", "wrist")),
        (["Wrist_cam", "top", "side"], ("top", "Wrist_cam")),
        (["top", "gripper"], ("top", "gripper")),
        (["left", "right"], ("left", None)),
    ],
)
def test_default_layout_puts_a_wrist_camera_in_the_corner(cameras, layout) -> None:
    from lelab.run_video import default_layout

    assert default_layout(cameras) == layout


def test_compose_crops_to_even_sides_and_insets_the_corner_bottom_right() -> None:
    from lelab.run_video import compose

    main = np.zeros((481, 641, 3), np.uint8)
    corner = np.full((240, 320, 3), 200, np.uint8)
    frame = compose(main, corner)

    assert frame.shape == (480, 640, 3)
    assert (frame[-20, -20] == 200).all()  # inside the inset
    assert (frame[10, 10] == 0).all()  # the main view elsewhere
    assert (frame[-20, 640 // 2] == 0).all()
    assert main.max() == 0  # the camera's frame is left alone


def test_a_portrait_corner_camera_stays_short() -> None:
    from lelab.run_video import CORNER_MAX_HEIGHT, compose

    frame = compose(np.zeros((480, 640, 3), np.uint8), np.full((640, 480, 3), 9, np.uint8))
    rows = np.where((frame[:, -20] == 9).all(axis=1))[0]
    assert 0 < len(rows) <= 480 * CORNER_MAX_HEIGHT


def _record(tmp_path: Path, frames: int, fps: int = 20, corner: bool = True):
    from lelab.run_video import RunVideo

    video = RunVideo(tmp_path / "run.mp4", fps, "front", "wrist" if corner else None)
    for i in range(frames):
        video.add(
            {"front": np.full((48, 64, 3), i * 10, np.uint8), "wrist": np.full((24, 32, 3), 250, np.uint8)}
        )
        time.sleep(1 / fps)
    return video, video.close()


def test_the_video_is_an_h264_mp4_of_the_full_frame_camera(tmp_path: Path) -> None:
    import av

    _, info = _record(tmp_path, frames=8)
    assert info["recording"] is False
    assert info["error"] is None
    assert info["name"] == "run.mp4"
    assert Path(info["path"]).is_file()
    assert not (tmp_path / "run.mp4.part").exists()
    assert info["frames"] >= 6

    with av.open(info["path"]) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        decoded = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    assert decoded[0].shape == (48, 64, 3)
    assert decoded[-1][-6, -6].min() > 200  # the wrist camera, bottom right


def test_a_run_without_frames_leaves_no_file(tmp_path: Path) -> None:
    video, info = _record(tmp_path, frames=0)
    assert info["name"] is None and info["frames"] == 0
    assert list(tmp_path.iterdir()) == []
    assert video.close() == info  # closing twice is harmless


def test_frames_inside_one_slot_are_kept_once(tmp_path: Path) -> None:
    from lelab.run_video import RunVideo

    video = RunVideo(tmp_path / "run.mp4", 1, "front")
    for _ in range(5):
        video.add({"front": np.zeros((8, 8, 3), np.uint8)})
    assert video.close()["frames"] == 1


@pytest.mark.parametrize(
    ("policy", "slug"),
    [
        ("act (outputs/act_pick_cube/checkpoints/last/pretrained_model)", "act_pick_cube"),
        ("smolvla (outputs/train/smolvla/checkpoints/020000/pretrained_model)", "smolvla"),
        ("pi05 (user/pi05 so101 v2)", "pi05_so101_v2"),
        ("act (pretrained_model)", "act"),
        (None, "model"),
    ],
)
def test_the_video_is_named_after_the_checkpoint(policy, slug) -> None:
    import re

    from lelab.run_video import video_file_name

    assert re.fullmatch(rf"\d{{8}}_\d{{6}}_{re.escape(slug)}\.mp4", video_file_name(policy))


@pytest.mark.parametrize("name", ["../x.mp4", "a/b.mp4", ".hidden.mp4", "x.mov", "missing.mp4"])
def test_resolve_video_refuses_other_files(tmp_path: Path, name: str) -> None:
    from lelab.run_video import resolve_video

    (tmp_path / "x.mov").write_bytes(b"")
    assert resolve_video(name, str(tmp_path)) is None


def test_the_route_serves_a_saved_video(tmp_lerobot_home: Path) -> None:
    from fastapi.testclient import TestClient

    from lelab.server import app
    from lelab.utils import config

    folder = Path(config.RUN_VIDEOS_PATH)
    folder.mkdir(parents=True)
    (folder / "20261117_120000_act.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

    client = TestClient(app)
    response = client.get("/run-videos/20261117_120000_act.mp4")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["content-disposition"].startswith("inline")
    assert client.get("/run-videos/nope.mp4").status_code == 404
    assert client.get("/run-videos/..%2Fremote_servers.mp4").status_code == 404
