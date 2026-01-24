# python
import pytest
from cyber_trainer.preprocessing import JointAngleCalculator
from analysis.exercise_rules import ShoulderPressRules, JointStatus, Repetition
import numpy as np

class _LM:
    """Prosty obiekt symulujący MediaPipe landmark (x, y, visibility)."""
    def __init__(self, x: float, y: float, visibility: float = 0.99):
        self.x = float(x)
        self.y = float(y)
        self.visibility = float(visibility)


def _make_empty_landmarks():
    return [None] * 33


def test_left_elbow_angle_approx_90_deg():
    calc = JointAngleCalculator(visibility_threshold=0.5)
    lm = _make_empty_landmarks()
    lm[11] = _LM(0.0, 0.0, 0.9)
    lm[13] = _LM(1.0, 0.0, 0.9)
    lm[15] = _LM(1.0, 1.0, 0.9)

    angle = calc.get_joint_angle(lm, "left_elbow", (100, 100, 3))
    assert angle is not None
    assert angle == pytest.approx(90.0, abs=1.0)


def test_get_all_angles_with_none_returns_none_values():
    calc = JointAngleCalculator()
    res = calc.get_all_angles(None, (100, 100, 3))
    assert isinstance(res, dict)
    assert all(v is None for v in res.values())


def test_shoulder_rules_detects_angle_error():
    rules = ShoulderPressRules(view_type="front")
    angles = {"left_shoulder": 10.0}
    assert rules.has_angle_errors(angles) is True

    statuses = rules.check_angles(angles)
    assert statuses["left_shoulder"] == JointStatus.ERROR


def test_repetition_summary_empty():
    rules = ShoulderPressRules(view_type="front")
    summary = rules.get_repetition_summary()
    assert summary["total_reps"] == 0
    assert summary["complete_reps"] == 0
    assert summary["incomplete_reps"] == 0
    assert summary["avg_rom"] == 0.0


def _make_rep(start, end, min_angle, max_angle, is_complete, rom=None):
    if rom is None:
        rom = max_angle - min_angle
    return Repetition(
        start_frame=start,
        end_frame=end,
        min_angle=min_angle,
        max_angle=max_angle,
        rom=rom,
        is_complete=is_complete,
        errors=[]
    )


def test_repetition_summary_counts_and_avg_rom():
    rules = ShoulderPressRules(view_type="front")
    reps = [
        _make_rep(0, 10, 20.0, 140.0, True),   # ROM 120
        _make_rep(11, 20, 40.0, 140.0, True),  # ROM 100
        _make_rep(21, 30, 60.0, 140.0, False), # ROM 80
    ]
    rules.repetitions.extend(reps)

    summary = rules.get_repetition_summary()
    assert summary["total_reps"] == 3
    assert summary["complete_reps"] == 2
    assert summary["incomplete_reps"] == 1
    # średni ROM z [120, 100, 80] = 100.0
    assert summary["avg_rom"] == pytest.approx(100.0, rel=1e-6)


def test_effectiveness_metric_matches_expected_percentage():
    rules = ShoulderPressRules(view_type="front")
    total = 25
    complete = 23
    for i in range(complete):
        rules.repetitions.append(_make_rep(i * 10, i * 10 + 9, 30.0, 140.0, True))
    for i in range(total - complete):
        rules.repetitions.append(_make_rep((complete + i) * 10, (complete + i) * 10 + 9, 50.0, 120.0, False))

    summary = rules.get_repetition_summary()
    assert summary["total_reps"] == total
    assert summary["complete_reps"] == complete

    effectiveness = (summary["complete_reps"] / summary["total_reps"]) * 100.0
    assert effectiveness == pytest.approx(92.0, abs=0.1)


def test_side_view_summary_behaviour():
    # dla widoku side logika detekcji ukończenia powtórzenia jest inna,
    # tutaj testujemy, że get_repetition_summary działa poprawnie również dla side
    rules = ShoulderPressRules(view_type="side")
    reps = [
        _make_rep(0, 10, 100.0, 130.0, True),
        _make_rep(11, 20, 105.0, 128.0, True),
    ]
    rules.repetitions.extend(reps)

    summary = rules.get_repetition_summary()
    assert summary["total_reps"] == 2
    assert summary["complete_reps"] == 2
    assert summary["incomplete_reps"] == 0
    assert float(summary["avg_rom"]) == pytest.approx(np.mean([r.rom for r in reps]), rel=1e-6)

