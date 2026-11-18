"""Tests for lelab.interactive: when the arm counts as having settled, and the command log of an interactive
session (what can be sent when, how commands end, and the captions the run video shows)."""

from __future__ import annotations

import numpy as np
import pytest

START = np.zeros(6)


def _feed(detector, samples: list[tuple[float, list[float]]]) -> list[bool]:
    return [detector.update(t, np.array(state, float)) for t, state in samples]


def test_an_arm_that_never_moved_has_not_settled() -> None:
    from lelab.interactive import SettleDetector

    detector = SettleDetector(START)
    assert not any(_feed(detector, [(t / 10, [0.5] * 6) for t in range(200)]))


def test_moving_then_staying_still_settles_after_the_window() -> None:
    from lelab.interactive import SETTLE_S, SettleDetector

    detector = SettleDetector(START)
    moving = [(t / 10, [t * 5.0] * 6) for t in range(20)]  # 2 s of motion to 95 degrees
    still = [
        (2 + t / 10, [100.0 + (t % 2) * 0.5] * 6) for t in range(1, 60)
    ]  # then 0.5-degree noise from 2.1 s
    results = _feed(detector, moving + still)
    assert not any(results[: len(moving)])
    first = results.index(True)
    settled_at = (moving + still)[first][0]
    assert settled_at == pytest.approx(2.1 + SETTLE_S, abs=0.11)  # within one sample


def test_a_jittering_arm_does_not_settle() -> None:
    from lelab.interactive import SETTLE_DEG, SettleDetector

    detector = SettleDetector(START)
    _feed(detector, [(t / 10, [t * 2.0] * 6) for t in range(10)])
    jitter = [(1 + t / 10, [20.0 + (t % 2) * (SETTLE_DEG + 1)] * 6) for t in range(100)]
    assert not any(_feed(detector, jitter))


def test_one_joint_moving_is_enough_and_one_joint_moving_keeps_it_going() -> None:
    from lelab.interactive import SettleDetector

    detector = SettleDetector(START)
    _feed(detector, [(t / 10, [0, 0, 0, 0, 0, t * 2.0]) for t in range(10)])  # only the gripper opens
    wrist_turning = [(1 + t / 10, [0, 0, 0, t * 0.5, 0, 18.0]) for t in range(50)]
    assert not any(_feed(detector, wrist_turning))
    assert any(_feed(detector, [(6 + t / 10, [0, 0, 0, 24.5, 0, 18.0]) for t in range(40)]))


def test_settling_needs_the_minimum_run_time() -> None:
    from lelab.interactive import SettleDetector

    detector = SettleDetector(START, settle_s=0.5, min_run_s=3.0)
    _feed(detector, [(0.0, [10.0] * 6)])  # moved on the first reading
    results = _feed(detector, [(t / 10, [10.0] * 6) for t in range(1, 40)])
    assert results.index(True) + 1 == 30  # t = 3.0 s


def test_clearing_restarts_the_still_window() -> None:
    from lelab.interactive import SettleDetector

    detector = SettleDetector(START, settle_s=1.0, min_run_s=0.0)
    _feed(detector, [(t / 10, [10.0] * 6) for t in range(9)])
    detector.clear()
    assert not any(_feed(detector, [(0.9 + t / 10, [10.0] * 6) for t in range(10)]))
    assert _feed(detector, [(2.0, [10.0] * 6)]) == [True]


def test_waiting_for_the_model_does_not_count_as_settled() -> None:
    from lelab.interactive import SettleDetector

    detector = SettleDetector(START, settle_s=1.0, min_run_s=0.0)
    _feed(detector, [(t / 10, [t * 5.0] * 6) for t in range(5)])  # moved to 20 degrees
    # Then still, but with a 2 s wait for the next chunk in the middle: no reading during it.
    assert _feed(detector, [(0.5, [20.0] * 6), (2.5, [20.0] * 6), (2.6, [20.0] * 6)]) == [False] * 3
    assert any(_feed(detector, [(2.6 + t / 10, [20.0] * 6) for t in range(1, 12)]))


# ------------------------------------------------------------------------------------------- command log
def test_a_command_can_be_sent_only_when_nothing_else_runs() -> None:
    from lelab.interactive import CommandLog

    log = CommandLog(command_time_s=60)
    with pytest.raises(ValueError, match="Type a command"):
        log.submit("   ")
    cmd = log.submit("  pick   up the cube ")
    assert cmd.text == "pick up the cube"
    assert log.wake.is_set()
    assert log.state() == "running"
    with pytest.raises(ValueError, match="still running"):
        log.submit("again")

    assert log.take_pending() is cmd
    log.mark_started(cmd, 100.0)
    assert log.take_pending() is None
    log.finish(cmd, "completed", "arm settled")
    assert log.state() == "ready"

    log.set_returning(True)
    assert log.state() == "returning"
    with pytest.raises(ValueError, match="start pose"):
        log.submit("next")
    log.set_returning(False)
    assert log.submit("next").id == 2


def test_done_and_stop_command_need_a_running_command() -> None:
    from lelab.interactive import CommandLog

    log = CommandLog(command_time_s=60)
    with pytest.raises(ValueError, match="No command"):
        log.request_end(True)
    cmd = log.submit("wave")
    log.wake.clear()
    log.request_end(False)
    assert log.end_requested(cmd) is False
    assert log.wake.is_set()


def test_finishing_twice_keeps_the_first_outcome_and_close_stops_the_rest() -> None:
    from lelab.interactive import CommandLog

    log = CommandLog(command_time_s=60)
    first = log.submit("one")
    log.finish(first, "not_completed", "time limit")
    log.finish(first, "completed", "marked done")
    assert (first.status, first.how) == ("not_completed", "time limit")

    second = log.submit("two")
    log.close()
    assert (second.status, second.how) == ("stopped", "session ended")
    assert log.summary() == {"not_completed": 1, "stopped": 1}


def test_the_history_lists_every_command_with_its_time() -> None:
    from lelab.interactive import CommandLog

    log = CommandLog(command_time_s=60)
    cmd = log.submit("pick")
    log.mark_started(cmd, 1000.0)
    rows = log.to_list(now=1012.4)
    assert rows == [
        {
            "id": 1,
            "text": "pick",
            "status": "running",
            "how": None,
            "submitted_at": cmd.submitted_at,
            "started_at": 1000.0,
            "ended_at": None,
            "elapsed_s": 12.4,
        }
    ]


def test_the_video_shows_the_last_three_commands() -> None:
    from lelab.interactive import CommandLog

    log = CommandLog(command_time_s=60)
    outcomes = [
        ("completed", "arm settled", 23),
        ("not_completed", "time limit", 60),
        ("stopped", "stopped", 5),
    ]
    for i, (status, how, seconds) in enumerate(outcomes):
        cmd = log.submit(f"command {i + 1}")
        log.mark_started(cmd, 0.0)
        log.finish(cmd, status, how)
        cmd.ended_at = float(seconds)
    assert log.captions(now=100.0) == (
        ("completed", "command 1", "0:23"),
        ("not_completed", "command 2", "1:00 time limit"),
        ("stopped", "command 3", "0:05"),
    )

    fourth = log.submit("command 4")
    assert log.captions(now=100.0)[-1] == ("pending", "command 4", "starting")
    log.mark_started(fourth, 90.0)
    captions = log.captions(now=102.7)
    assert [c[1] for c in captions] == ["command 2", "command 3", "command 4"]  # the first one dropped off
    assert captions[-1] == ("running", "command 4", "0:12 / 1:00")
    assert len(log.to_list(now=102.7)) == 4  # the page keeps them all
