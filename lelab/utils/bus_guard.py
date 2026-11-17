"""Ride out brief failures on an arm's serial bus.

USB interference (a motor's stall current, a long cable, a busy hub) now and
then makes one read or write of the motors fail. LeRobot raises on the first
one, which used to end the whole session. A `BusGuard` wraps the bus's
`sync_read`/`sync_write` instead: a failed read hands back the last good
value, a failed write is skipped (the motors hold their last goal), and the
loop carries on. Only when the bus has not answered for `MAX_OUTAGE_S` in a
row does it raise `ArmCommunicationLostError`, which stops the session as
before.

Installed after the device is set up (calibration, configure), so setup still
fails loudly, and removed before the disconnect.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# How long an arm may stop answering before the session stops.
MAX_OUTAGE_S = 10.0
# An outage this long or longer is worth telling the user about (a stale stretch in a dataset).
NOTABLE_OUTAGE_S = 0.5

GUARD_ATTR = "_lelab_bus_guard"


class ArmCommunicationLostError(ConnectionError):
    """An arm's bus kept failing for longer than the outage limit."""


def _is_glitch(exc: BaseException) -> bool:
    """A failed transfer worth retrying on the next tick.

    `SerialMotorsBus` raises ConnectionError when a motor doesn't answer and
    RuntimeError when it answers with an error; a port that was never opened
    (DeviceNotConnectedError) is a programming error, not interference.
    """
    try:
        from lerobot.utils.errors import DeviceNotConnectedError
    except ImportError:  # pragma: no cover - lerobot is a dependency
        DeviceNotConnectedError = ()  # noqa: N806
    if DeviceNotConnectedError and isinstance(exc, DeviceNotConnectedError):
        return False
    return isinstance(exc, (ConnectionError, RuntimeError, OSError))


def _motors_key(motors: Any) -> Any:
    if motors is None or isinstance(motors, (str, int)):
        return motors
    return tuple(motors)


class BusGuard:
    """Glitch tolerance for one motor bus. `label` names the arm in messages ("leader arm")."""

    def __init__(
        self,
        bus: Any,
        label: str,
        max_outage_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bus = bus
        self.label = label
        self.max_outage_s = MAX_OUTAGE_S if max_outage_s is None else max_outage_s
        self._clock = clock
        self._lock = threading.Lock()
        self._last: dict[tuple, dict] = {}
        self._read: Callable[..., Any] | None = None
        self._write: Callable[..., Any] | None = None
        self._outage_started: float | None = None
        self._outage_skipped = 0
        self.skipped = 0
        self.outages = 0
        self.longest_outage_s = 0.0
        self.last_outage: dict[str, Any] | None = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------ install
    def install(self) -> BusGuard:
        """Wrap the bus, and read the joint positions once so a first failed read has a value to fall back on."""
        self._read = getattr(self.bus, "sync_read", None)
        self._write = getattr(self.bus, "sync_write", None)
        if self._read is not None:
            self.bus.sync_read = self.sync_read
        if self._write is not None:
            self.bus.sync_write = self.sync_write
        setattr(self.bus, GUARD_ATTR, self)
        if self._read is not None:
            try:
                self.sync_read("Present_Position")
            except Exception as exc:
                logger.warning("%s: first read failed: %s", self.label, exc)
        return self

    def uninstall(self) -> None:
        """Hand the bus back its own methods (idempotent)."""
        state = getattr(self.bus, "__dict__", {})
        for name, mine in (("sync_read", self.sync_read), ("sync_write", self.sync_write)):
            if state.get(name) == mine:
                del state[name]
        if state.get(GUARD_ATTR) is self:
            del state[GUARD_ATTR]

    # ------------------------------------------------------------------ the bus
    def sync_read(self, data_name: str, *args: Any, **kwargs: Any) -> Any:
        motors = args[0] if args else kwargs.get("motors")
        key = (str(data_name).lower(), _motors_key(motors), kwargs.get("normalize", True))
        try:
            value = self._read(data_name, *args, **kwargs)
        except Exception as exc:
            if not _is_glitch(exc):
                raise
            self._failed(exc, f"reading {data_name}")
            last = self._last.get(key)
            if last is None:
                raise
            return dict(last)
        self._answered()
        if isinstance(value, dict):
            self._last[key] = dict(value)
        return value

    def sync_write(self, data_name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            result = self._write(data_name, *args, **kwargs)
        except Exception as exc:
            if not _is_glitch(exc):
                raise
            self._failed(exc, f"writing {data_name}")
            return None
        self._answered()
        return result

    def raw_read(self, data_name: str, *args: Any, **kwargs: Any) -> Any:
        """A read that fails like the bus does, to ask whether the motors really answer."""
        read = self._read or self.bus.sync_read
        return read(data_name, *args, **kwargs)

    # ------------------------------------------------------------------ state
    @property
    def in_outage(self) -> bool:
        return self._outage_started is not None

    def outage_s(self) -> float:
        started = self._outage_started
        return 0.0 if started is None else self._clock() - started

    def _failed(self, exc: BaseException, what: str) -> None:
        now = self._clock()
        with self._lock:
            self.skipped += 1
            self._outage_skipped += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self._outage_started is None:
                self._outage_started = now
                self.outages += 1
                logger.warning(
                    "%s: %s failed (%s); keeping its last value and carrying on (stops after %.0f s)",
                    self.label,
                    what,
                    exc,
                    self.max_outage_s,
                )
            elapsed = now - self._outage_started
        if elapsed >= self.max_outage_s:
            raise ArmCommunicationLostError(
                f"Lost contact with the {self.label} for {elapsed:.0f} s ({what} kept failing: {exc}). "
                "Check its USB cable, hub and power, then start again."
            ) from exc

    def _answered(self) -> None:
        if self._outage_started is None:
            return
        now = self._clock()
        with self._lock:
            if self._outage_started is None:
                return
            duration = now - self._outage_started
            self.longest_outage_s = max(self.longest_outage_s, duration)
            self.last_outage = {
                "duration_s": round(duration, 2),
                "skipped": self._outage_skipped,
                "ended_at": time.time(),
            }
            self._outage_started = None
            self._outage_skipped = 0
        log = logger.warning if duration >= NOTABLE_OUTAGE_S else logger.info
        log(
            "%s: answering again after %.2f s (%d skipped)", self.label, duration, self.last_outage["skipped"]
        )

    def stats(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "in_outage": self.in_outage,
            "outage_s": round(self.outage_s(), 2),
            "skipped": self.skipped,
            "outages": self.outages,
            "longest_outage_s": round(self.longest_outage_s, 2),
            "last_outage": self.last_outage,
            "last_error": self.last_error,
        }


def guard_buses(*devices: tuple[Any, str]) -> list[BusGuard]:
    """Install a guard on each (device, label) whose device has a bus; skip the None devices."""
    guards = []
    for device, label in devices:
        bus = getattr(device, "bus", None) if device is not None else None
        if bus is not None:
            guards.append(BusGuard(bus, label).install())
    return guards


def unguard(guards: list[BusGuard]) -> None:
    for guard in guards:
        try:
            guard.uninstall()
        except Exception as exc:  # pragma: no cover - never let cleanup fail on this
            logger.warning("Could not remove the bus guard of the %s: %s", guard.label, exc)


def notable_outages(guards: list[BusGuard], since: float | None = None) -> list[dict[str, Any]]:
    """The outages worth showing: one in progress, or the last one if it ended after `since` and was long enough."""
    out = []
    for guard in guards:
        if guard.in_outage and guard.outage_s() >= NOTABLE_OUTAGE_S:
            out.append({"device": guard.label, "ongoing": True, "duration_s": round(guard.outage_s(), 1)})
            continue
        last = guard.last_outage
        if last and last["duration_s"] >= NOTABLE_OUTAGE_S and (since is None or last["ended_at"] >= since):
            out.append({"device": guard.label, "ongoing": False, **last})
    return out
