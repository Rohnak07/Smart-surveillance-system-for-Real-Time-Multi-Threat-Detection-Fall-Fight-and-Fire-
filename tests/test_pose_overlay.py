import numpy as np
import yaml

from kavach.fall import FallDetector
from kavach.pipeline import Engine
from tests.synth import person, skeleton


FALL_CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))["fall"]


def tilted_skeleton(angle_deg, vertical_shift=0.0):
    keypoints, confidence = skeleton(300, 460, 300)
    hip = (keypoints[11] + keypoints[12]) / 2
    angle = np.radians(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    keypoints = (keypoints - hip) @ rotation.T + hip
    keypoints[:, 1] += vertical_shift
    return keypoints, confidence


def make_engine():
    engine = Engine.__new__(Engine)
    engine.cfg = {"tracker": {"max_age_s": 2.0}}
    engine.fall = FallDetector(FALL_CFG)
    engine._pose_overlay = {}
    return engine


def test_pose_overlay_shows_normal_falling_and_fallen_per_person():
    engine = make_engine()

    engine._update_pose_overlay([person(1, *skeleton(300, 460, 300))], 0.0)
    assert engine._pose_overlay_current[1]["status"] == "normal"

    engine._update_pose_overlay([person(1, *tilted_skeleton(45, 50))], 0.2)
    assert engine._pose_overlay_current[1]["status"] == "falling"

    engine._update_pose_overlay([person(1, *skeleton(300, 460, 300, lying=True))], 0.4)
    assert engine._pose_overlay_current[1]["status"] == "fallen"


def test_pose_overlay_keeps_people_states_independent():
    engine = make_engine()
    standing = person(1, *skeleton(200, 460, 300))
    lying = person(2, *skeleton(500, 460, 300, lying=True))

    engine._update_pose_overlay([standing, lying], 0.0)

    assert engine._pose_overlay_current[1]["status"] == "normal"
    assert engine._pose_overlay_current[2]["status"] == "fallen"


def test_pose_overlay_draws_annotations_without_changing_input_frame():
    engine = make_engine()
    engine.people = [person(1, *skeleton(300, 460, 300))]
    engine._update_pose_overlay(engine.people, 0.0)
    engine.fire = None

    frame = np.zeros((520, 640, 3), dtype=np.uint8)
    original = frame.copy()
    annotated = Engine._draw(engine, frame)

    assert not np.array_equal(annotated, original)
    assert np.array_equal(frame, original)
