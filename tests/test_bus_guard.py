"""Tests for lelab.utils.bus_guard: brief failures of an arm's bus are ridden
out, a long silence stops the session, and teleoperation uses it end to end
with fake arms."""

from __future__ import annotations

import time
from typing import Any

import pytest

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class GlitchyBus:
    """A motor bus that fails the next `fail_reads`/`fail_writes` transfers, or every one while `dead`."""

    def __init__(self, error: type[Exception] = ConnectionError) -> None:
        self.motors = dict.fromkeys(MOTORS)
        self.position = dict(zip(MOTORS, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], strict=True))
        self.error = error
        self.fail_reads = 0
        self.fail_writes = 0
        self.dead = False
        self.reads = 0
        self.writes: list[dict[str, float]] = []

    def connect(self) -> None:
        pass

    def disable_torque(self) -> None:
        pass

    def write_calibration(self, calibration: Any) -> None:
        pass

    def sync_read(self, data_name: str, motors: Any = None, *, normalize: bool = True, num_retry: int = 0):
        self.reads += 1
        if self.dead or self.fail_reads > 0:
            self.fail_reads = max(self.fail_reads - 1, 0)
            raise self.error("Failed to sync read 'Present_Position' from motor IDs [1, 2, 3, 4, 5, 6]")
        return dict(self.position)

    def sync_write(self, data_name: str, values: dict[str, float], *, num_retry: int = 0) -> None:
        if self.dead or self.fail_writes > 0:
            self.fail_writes = max(self.fail_writes - 1, 0)
            raise self.error("Failed to sync write 'Goal_Position'")
        self.writes.append(dict(values))
        self.position.update(values)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _guard(bus: GlitchyBus, clock: Clock | None = None, max_outage_s: float = 10.0):
    from lelab.utils.bus_guard import BusGuard

    return BusGuard(bus, "leader arm", max_outage_s=max_outage_s, clock=clock or Clock()).install()


# ------------------------------------------------------------------------------------------------ the guard
def test_a_failed_read_returns_the_last_positions() -> None:
    bus = GlitchyBus()
    guard = _guard(bus)
    bus.position["gripper"] = 50.0
    assert bus.sync_read("Present_Position")["gripper"] == 50.0

    bus.position["gripper"] = 80.0
    bus.fail_reads = 2
    assert bus.sync_read("Present_Position")["gripper"] == 50.0
    assert bus.sync_read("Present_Position")["gripper"] == 50.0
    assert guard.skipped == 2
    assert guard.in_outage

    assert bus.sync_read("Present_Position")["gripper"] == 80.0
    assert not guard.in_outage
    assert guard.outages == 1
    assert guard.last_outage["skipped"] == 2


def test_a_failed_write_is_skipped() -> None:
    bus = GlitchyBus()
    guard = _guard(bus)
    bus.fail_writes = 1
    assert bus.sync_write("Goal_Position", {"gripper": 10.0}) is None
    bus.sync_write("Goal_Position", {"gripper": 20.0})
    assert bus.writes == [{"gripper": 20.0}]
    assert guard.skipped == 1 and not guard.in_outage


def test_a_refused_read_counts_as_a_glitch_too() -> None:
    bus = GlitchyBus(error=RuntimeError)
    guard = _guard(bus)
    bus.fail_reads = 1
    assert bus.sync_read("Present_Position") == bus.position
    assert guard.skipped == 1


def test_a_long_silence_stops_with_a_clear_error() -> None:
    from lelab.utils.bus_guard import ArmCommunicationLostError

    clock = Clock()
    bus = GlitchyBus()
    _guard(bus, clock, max_outage_s=10.0)
    bus.dead = True
    bus.sync_read("Present_Position")
    clock.now += 9.9
    bus.sync_write("Goal_Position", {"gripper": 1.0})
    clock.now += 0.2
    with pytest.raises(ArmCommunicationLostError, match="Lost contact with the leader arm for 10 s"):
        bus.sync_read("Present_Position")


def test_each_answer_restarts_the_outage_clock() -> None:
    clock = Clock()
    bus = GlitchyBus()
    guard = _guard(bus, clock, max_outage_s=1.0)
    for _ in range(5):  # fails often, but never for a whole second
        bus.fail_reads = 1
        bus.sync_read("Present_Position")
        clock.now += 0.8
        bus.sync_read("Present_Position")
    assert guard.skipped == 5
    assert not guard.in_outage


def test_a_port_that_is_not_open_is_not_a_glitch() -> None:
    from lerobot.utils.errors import DeviceNotConnectedError

    bus = GlitchyBus(error=DeviceNotConnectedError)
    _guard(bus)
    bus.fail_reads = 1
    with pytest.raises(DeviceNotConnectedError):
        bus.sync_read("Present_Position")


def test_a_failed_read_with_nothing_to_fall_back_on_raises() -> None:
    bus = GlitchyBus()
    bus.fail_reads = 1  # the first read, at install, fails
    _guard(bus)
    bus.fail_reads = 1
    with pytest.raises(ConnectionError):
        bus.sync_read("Present_Position", ["gripper"])


def test_uninstall_gives_the_bus_its_own_methods_back() -> None:
    from lelab.utils.bus_guard import GUARD_ATTR

    bus = GlitchyBus()
    guard = _guard(bus)
    assert getattr(bus, GUARD_ATTR) is guard
    guard.uninstall()
    guard.uninstall()
    assert not hasattr(bus, GUARD_ATTR)
    bus.fail_reads = 1
    with pytest.raises(ConnectionError):
        bus.sync_read("Present_Position")


def test_raw_read_bypasses_the_guard() -> None:
    bus = GlitchyBus()
    guard = _guard(bus)
    bus.dead = True
    with pytest.raises(ConnectionError):
        guard.raw_read("Present_Position")


def test_notable_outages_reports_the_ongoing_and_the_recent_ones() -> None:
    from lelab.utils.bus_guard import notable_outages

    clock = Clock()
    bus = GlitchyBus()
    guard = _guard(bus, clock)
    bus.dead = True
    bus.sync_read("Present_Position")
    clock.now += 0.1
    bus.sync_read("Present_Position")
    assert notable_outages([guard]) == []  # too short to mention
    clock.now += 2.0
    assert notable_outages([guard]) == [{"device": "leader arm", "ongoing": True, "duration_s": 2.1}]

    bus.dead = False
    bus.sync_read("Present_Position")
    [past] = notable_outages([guard], since=time.time() - 5)
    assert past["ongoing"] is False and past["duration_s"] == 2.1
    assert notable_outages([guard], since=time.time() + 5) == []


# ------------------------------------------------------------------------------------------ teleoperation
class _Arm:
    """An SO-101 arm whose reads and writes go through its (glitchy) bus, like lerobot's."""

    def __init__(self, config: Any) -> None:
        self.bus = GlitchyBus()
        self.calibration: dict = {}
        self.cameras: dict = {}
        self.disconnected = False

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        return {f"{m}.pos": v for m, v in self.bus.sync_read("Present_Position").items()}

    def get_observation(self) -> dict[str, float]:
        return {f"{m}.pos": v for m, v in self.bus.sync_read("Present_Position").items()}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.bus.sync_write("Goal_Position", {k.removesuffix(".pos"): v for k, v in action.items()})
        return action

    def disconnect(self) -> None:
        self.disconnected = True


@pytest.fixture
def teleop(monkeypatch: pytest.MonkeyPatch):
    import lelab.teleoperate as teleop

    arms: dict[str, _Arm] = {}

    def make(role: str):
        def factory(config: Any) -> _Arm:
            arms[role] = _Arm(config)
            return arms[role]

        return factory

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_error", None)
    monkeypatch.setattr(teleop, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))
    monkeypatch.setattr(teleop, "SO101Follower", make("follower"))
    monkeypatch.setattr(teleop, "SO101Leader", make("leader"))
    yield teleop, arms
    if teleop.teleoperation_active:
        teleop.handle_stop_teleoperation()


def _start(teleop) -> dict:
    from lelab.teleoperate import TeleoperateRequest

    return teleop.handle_start_teleoperation(
        TeleoperateRequest(
            leader_port="/dev/leader",
            follower_port="/dev/follower",
            leader_config="leader",
            follower_config="follower",
        )
    )


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        time.sleep(0.01)


def test_teleoperation_rides_out_a_few_failed_transfers(teleop) -> None:
    teleop, arms = teleop
    assert _start(teleop)["success"] is True
    leader, follower = arms["leader"], arms["follower"]

    leader.bus.fail_reads = 2
    follower.bus.fail_writes = 2
    leader.bus.position["gripper"] = 42.0
    _wait_for(lambda: follower.bus.position["gripper"] == 42.0)

    assert teleop.teleoperation_active is True
    assert teleop.handle_teleoperation_status()["error"] is None
    assert teleop.handle_stop_teleoperation()["success"] is True
    assert leader.disconnected and follower.disconnected
    assert "sync_read" not in vars(leader.bus)  # guard removed before the disconnect


def test_teleoperation_stops_when_an_arm_stays_silent(teleop, monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab.utils import bus_guard

    monkeypatch.setattr(bus_guard, "MAX_OUTAGE_S", 1.0)
    teleop, arms = teleop
    assert _start(teleop)["success"] is True

    arms["leader"].bus.dead = True
    _wait_for(lambda: teleop.handle_teleoperation_status()["arm_link_issues"])
    _wait_for(lambda: not teleop.teleoperation_active)

    status = teleop.handle_teleoperation_status()
    assert "Lost contact with the leader arm" in status["error"]
    assert status["arm_link_issues"] == []
    assert arms["leader"].disconnected and arms["follower"].disconnected
