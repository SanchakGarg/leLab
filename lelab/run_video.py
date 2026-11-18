"""A video of a model run on the arm: one camera full frame, another one (the
wrist camera, usually) inset in a corner. Interactive sessions add a strip
below the picture with the latest commands and how each one went.

The control loop hands each tick's camera frames to `RunVideo.add`, which only
queues them; a background thread composes and encodes the mp4 (H.264, so any
browser plays it), so recording never slows the arm down. When the encoder
falls behind, frames are dropped rather than queued without bound.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import queue
import re
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Camera names that mean "on the gripper": that camera goes in the corner by default.
WRIST_HINTS = ("wrist", "hand", "gripper")
# The inset's width as a share of the frame's, and its largest height share (for portrait cameras).
CORNER_WIDTH = 0.3
CORNER_MAX_HEIGHT = 0.45
QUEUE_FRAMES = 90
# Interactive sessions: (badge, badge colour) per command status in the caption strip. Words, not ✓/✗:
# Pillow's built-in font may not have those glyphs.
CAPTION_STYLES = {
    "pending": ("STARTING", (245, 158, 11)),
    "running": ("RUNNING", (245, 158, 11)),
    "completed": ("DONE", (34, 197, 94)),
    "not_completed": ("DIDN'T COMPLETE", (239, 68, 68)),
    "stopped": ("STOPPED", (148, 163, 184)),
    "failed": ("ERROR", (239, 68, 68)),
}
CAPTION_BACKGROUND = (17, 24, 39)
CAPTION_TEXT = (241, 245, 249)
CAPTION_MUTED = (148, 163, 184)
CAPTION_BADGE_TEXT = (15, 23, 42)
VIDEO_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.mp4$")


def is_wrist_camera(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in WRIST_HINTS)


def default_layout(cameras: list[str]) -> tuple[str | None, str | None]:
    """(full-frame camera, corner camera) for these names: a wrist camera goes in
    the corner when there is another camera to show full frame."""
    if not cameras:
        return None, None
    wrist = next((c for c in cameras if is_wrist_camera(c)), None)
    if wrist is None or len(cameras) == 1:
        return cameras[0], None
    return next(c for c in cameras if c != wrist), wrist


def _even(n: float) -> int:
    return max(2, int(n) // 2 * 2)


def _rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def compose(main: np.ndarray, corner: np.ndarray | None = None) -> np.ndarray:
    """The main frame (cropped to even sides, as H.264 needs) with `corner` inset at the bottom right."""
    import cv2

    main = _rgb(main)
    height, width = _even(main.shape[0]), _even(main.shape[1])
    frame = np.ascontiguousarray(main[:height, :width])
    if corner is None:
        return frame

    corner = _rgb(corner)
    ch, cw = corner.shape[:2]
    inset_w = width * CORNER_WIDTH
    inset_h = inset_w * ch / cw
    if inset_h > height * CORNER_MAX_HEIGHT:
        inset_h = height * CORNER_MAX_HEIGHT
        inset_w = inset_h * cw / ch
    inset_w, inset_h = _even(inset_w), _even(inset_h)
    small = cv2.resize(corner, (inset_w, inset_h), interpolation=cv2.INTER_AREA)

    margin = max(4, width // 50)
    border = 2
    x0, y0 = width - margin - inset_w, height - margin - inset_h
    frame = frame.copy()
    frame[max(y0 - border, 0) : y0 + inset_h + border, max(x0 - border, 0) : x0 + inset_w + border] = 255
    frame[y0 : y0 + inset_h, x0 : x0 + inset_w] = small
    return frame


# System fonts with accented letters (and, for Arial Unicode, most scripts), tried in order; Pillow's own
# font, which has only basic Latin, when none is there.
CAPTION_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


@functools.lru_cache(maxsize=8)
def _caption_font(size: int) -> Any:
    from PIL import ImageFont

    for path in CAPTION_FONTS:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: the fixed-size bitmap font
        return ImageFont.load_default()


def _caption_metrics(width: int, lines: int) -> tuple[int, int, int]:
    """(font size, row height, strip height) for a strip of `lines` rows under a frame `width` wide."""
    size = max(12, width // 40)
    row = int(size * 1.7)
    pad = max(4, size // 3)
    return size, row, _even(lines * row + 2 * pad)


def _fit(draw: Any, text: str, font: Any, max_width: float) -> str:
    """`text`, cut short with "..." to fit `max_width` pixels."""
    if max_width <= 0:
        return ""
    if draw.textlength(text, font=font) <= max_width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:  # the longest prefix that fits with the dots
        mid = (lo + hi + 1) // 2
        if draw.textlength(text[:mid].rstrip() + "...", font=font) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "..." if lo else ""


def render_caption_strip(width: int, captions: tuple[tuple[str, str, str], ...], lines: int) -> np.ndarray:
    """The strip under an interactive session's video: one row per command (status badge, the command,
    its time on the right), newest at the bottom, at most `lines` rows. Always `lines` rows tall, so the
    video keeps one size whatever the number of commands."""
    from PIL import Image, ImageDraw

    size, row, height = _caption_metrics(width, lines)
    image = Image.new("RGB", (width, height), CAPTION_BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = _caption_font(size)
    pad = max(4, size // 3)
    gap = max(6, size // 2)
    rows = list(captions)[-lines:] if lines > 0 else []
    top = height - pad - len(rows) * row  # bottom-aligned: the newest command sits on the last row

    def text_at(x: float, y: float, text: str, fill: tuple[int, int, int]) -> None:
        box = draw.textbbox((0, 0), text, font=font)
        draw.text((x - box[0], y + (row - (box[3] - box[1])) / 2 - box[1]), text, font=font, fill=fill)

    for i, (status, text, right) in enumerate(rows):
        y = top + i * row
        label, color = CAPTION_STYLES.get(status, (status.upper(), CAPTION_MUTED))
        badge_w = draw.textlength(label, font=font) + size
        inset = max(2, row // 8)
        draw.rounded_rectangle(
            [pad, y + inset, pad + badge_w, y + row - inset], radius=max(2, size // 3), fill=color
        )
        text_at(pad + size / 2, y, label, CAPTION_BADGE_TEXT)
        right_w = draw.textlength(right, font=font)
        text_at(width - pad - right_w, y, right, CAPTION_MUTED)
        x = pad + badge_w + gap
        text_at(x, y, _fit(draw, text, font, width - pad - right_w - gap - x), CAPTION_TEXT)
    return np.asarray(image)


def _codec() -> str:
    import av

    for name in ("libx264", "h264", "libopenh264", "mpeg4"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    raise RuntimeError("PyAV has no H.264 or MPEG-4 encoder")


# Path parts that say nothing about which model it is.
_GENERIC_PARTS = {
    "pretrained_model",
    "checkpoints",
    "checkpoint",
    "last",
    "outputs",
    "output",
    "train",
    "model",
}


def video_file_name(policy: str | None) -> str:
    """`<date>_<time>_<model>.mp4`, the model named after its checkpoint folder.

    The policy server names a model "act (outputs/act_pick_cube/checkpoints/last/pretrained_model)";
    that gives `act_pick_cube`.
    """
    policy = policy or "model"
    match = re.match(r"^\s*([^()]*?)\s*\((.*)\)\s*$", policy)
    kind, path = (match.group(1), match.group(2)) if match else (policy, "")
    parts = [
        p for p in re.split(r"[\\/]+", path) if p and p.lower() not in _GENERIC_PARTS and not p.isdigit()
    ]
    label = parts[-1] if parts else kind
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-")[:60] or "model"
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}.mp4"


class RunVideo:
    """Records one run. `add()` from the control loop, `close()` once at the end."""

    def __init__(
        self,
        path: str | os.PathLike,
        fps: int,
        main: str,
        corner: str | None = None,
        caption_lines: int = 0,
    ) -> None:
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")
        self.fps = int(fps)
        self.main = main
        self.corner = corner
        # Interactive sessions: a strip this many commands tall under every frame (0: none).
        self.caption_lines = caption_lines
        self._strip_key: tuple | None = None
        self._strip: np.ndarray | None = None
        self.frames = 0
        self.dropped = 0
        self.error: str | None = None
        self._t0: float | None = None
        self._last_pts = -1
        self._closed = False
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self._thread = threading.Thread(target=self._work, name="run-video", daemon=True)
        self._thread.start()

    def add(self, images: dict[str, Any], captions: tuple[tuple[str, str, str], ...] = ()) -> None:
        """Queue this tick's frames, stamped with the time since the first one (never blocks).

        `captions`: the commands to show under the picture, oldest first (see CommandLog.captions).
        """
        if self._closed or self.error is not None:
            return
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        pts = round((now - self._t0) * self.fps)
        if pts <= self._last_pts:  # two ticks inside one frame's slot
            return
        main = images.get(self.main)
        if main is None:
            return
        corner = images.get(self.corner) if self.corner else None
        try:
            self._queue.put_nowait((pts, main, corner, captions))
            self._last_pts = pts
        except queue.Full:
            self.dropped += 1

    def _work(self) -> None:
        import av

        container = stream = None
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                pts, main, corner, captions = item
                frame = self._with_captions(compose(main, corner), captions)
                if container is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    container = av.open(str(self.part), "w", format="mp4", options={"movflags": "faststart"})
                    codec = _codec()
                    stream = container.add_stream(codec, rate=self.fps)
                    stream.height, stream.width = frame.shape[:2]
                    stream.pix_fmt = "yuv420p"
                    stream.codec_context.time_base = Fraction(1, self.fps)
                    if codec == "libx264":
                        stream.options = {"crf": "23", "preset": "veryfast"}
                elif frame.shape[:2] != (stream.height, stream.width):
                    import cv2

                    frame = cv2.resize(frame, (stream.width, stream.height), interpolation=cv2.INTER_AREA)
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts = pts
                video_frame.time_base = Fraction(1, self.fps)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                self.frames += 1
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Recording the run's video failed")
            # Let close() through, and add() stop queueing.
            with contextlib.suppress(queue.Empty):
                while True:
                    self._queue.get_nowait()
        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()

    def _with_captions(self, frame: np.ndarray, captions: tuple) -> np.ndarray:
        if not self.caption_lines:
            return frame
        key = (frame.shape[1], captions)
        if key != self._strip_key:  # redrawn only when a command or its time changes
            self._strip = render_caption_strip(frame.shape[1], captions, self.caption_lines)
            self._strip_key = key
        return np.vstack([frame, self._strip])

    def close(self, timeout_s: float = 60.0) -> dict[str, Any]:
        """Finish the file. Returns what the run's result reports about the video."""
        if not self._closed:
            self._closed = True
            with contextlib.suppress(queue.Full):
                self._queue.put(None, timeout=timeout_s)
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                self.error = self.error or "the encoder did not finish in time"
            if self.frames and self.error is None:
                os.replace(self.part, self.path)
                logger.info(
                    "Run video saved: %s (%d frames, %d dropped)", self.path, self.frames, self.dropped
                )
            else:
                with contextlib.suppress(FileNotFoundError):
                    self.part.unlink()
        return self.info()

    def info(self) -> dict[str, Any]:
        saved = self._closed and self.frames > 0 and self.error is None
        return {
            "recording": not self._closed,
            "name": self.path.name if saved else None,
            "path": str(self.path) if saved else None,
            "camera": self.main,
            "corner_camera": self.corner,
            "frames": self.frames,
            "dropped": self.dropped,
            "error": self.error,
        }


def resolve_video(name: str, videos_dir: str) -> Path | None:
    """The saved video called `name`, or None (also for names that try to leave the folder)."""
    if not VIDEO_NAME.match(name):
        return None
    path = Path(videos_dir) / name
    return path if path.is_file() else None
