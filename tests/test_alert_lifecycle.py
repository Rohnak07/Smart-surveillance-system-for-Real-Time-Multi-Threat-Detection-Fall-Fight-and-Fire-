import cv2
import numpy as np
from kavach.pipeline import Engine
from kavach.types import Event


def engine():
    value = Engine.__new__(Engine)
    value.cfg = {
        "camera": {"id": "CAM-02", "location": "Entrance"},
        "fire": {"alert_clear_s": 1.25},
        "fight": {"alert_clear_s": 4.0},
    }
    value._active_alerts = {}
    value.banner = []
    value.fall = None
    return value


def test_ongoing_fire_only_opens_one_event_until_clear():
    detector = engine()
    first = Event("fire", 0.8, 0.0)
    second = Event("fire", 0.9, 0.5)
    assert detector._update_active_alerts([first], 0.0) == [first]
    assert detector._update_active_alerts([second], 0.5) == []
    assert detector._update_active_alerts([Event("fire", 0.8, 1.0)], 1.0) == []
    resumed = Event("fire", 0.8, 2.4)
    assert detector._update_active_alerts([resumed], 2.4) == [resumed]


def test_medical_alerts_for_different_people_are_independent():
    detector = engine()
    first = Event("person_down", 0.9, 5.0, subject_id=1)
    second = Event("person_down", 0.9, 5.0, subject_id=2)
    assert detector._update_active_alerts([first, second], 5.0) == [first, second]


def test_alert_banner_identifies_camera_and_event(monkeypatch):
    detector = engine()
    event = Event("fire", 0.8, 0.0)
    detector._active_alerts[(event.kind, event.subject_id)] = (event, 0.0)
    drawn = []
    monkeypatch.setattr(cv2, "rectangle", lambda image, *args, **kwargs: image)
    monkeypatch.setattr(cv2, "putText",
                        lambda image, text, *args, **kwargs: drawn.append(text) or image)
    Engine._draw_banner(detector, np.zeros((120, 640, 3), dtype=np.uint8), 0.0)
    assert any("CAM-02 | FIRE DETECTED" in text for text in drawn)
