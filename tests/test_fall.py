import yaml
import numpy as np
import cv2
from types import SimpleNamespace
from kavach.fall import FallDetector
from kavach.pipeline import Engine
from tests.synth import skeleton, person

CFG = yaml.safe_load(open("config.yaml"))["fall"]
H, FOOT, FPS = 300, 460, 10


def run(frames):
    """frames: list of (t, kpts, conf). Returns list of events."""
    d, evs = FallDetector(CFG), []
    for t, k, c in frames:
        evs += d.update([person(1, k, c)], t)
    return evs


def seq(parts):
    out, t = [], 0.0
    for dur, fn in parts:
        n = int(dur * FPS)
        for i in range(n):
            out.append((t, *fn(i / max(n - 1, 1))))
            t += 1.0 / FPS
    return out


stand = lambda f: skeleton(300, FOOT, H)
lie = lambda f: skeleton(300, FOOT, H, lying=True)
sit = lambda f: skeleton(300, FOOT, 0.7 * H)            # smaller, still upright torso
bend = lambda f: skeleton(300, FOOT, H * (1 - 0.35 * f), lean=60 * f)


def tilt(f):                                           # torso rotates 0 -> 90 deg
    k, c = skeleton(300, FOOT, H)
    import numpy as np
    hp = (k[11] + k[12]) / 2
    ang = np.radians(90 * f)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    k = (k - hp) @ R.T + hp + np.array([0, 0])
    return k, c


def test_fast_fall_then_stays_down_alerts():
    evs = run(seq([(2, stand), (0.5, tilt), (3, lie)]))
    assert [e.kind for e in evs] == ["fall"]


def test_short_lying_pose_under_0_2s_no_alert():
    evs = run(seq([(2, stand), (0.1, lie), (2, stand)]))
    assert evs == []


def test_fall_that_recovers_has_no_medical_escalation():
    evs = run(seq([(2, stand), (0.5, tilt), (0.8, lie), (2, stand)]))
    assert [e.kind for e in evs] == ["fall"]


def test_stationary_visible_person_alerts_after_five_seconds():
    events = run(seq([(8, stand)]))
    assert [event.kind for event in events] == ["person_down"]
    assert abs(events[0].t - 5.0) <= 1e-6


def test_stationary_sitting_person_alerts_after_five_seconds():
    events = run(seq([(2, stand), (8, sit)]))
    assert [event.kind for event in events] == ["person_down"]
    assert 7.5 <= events[0].t <= 7.7


def test_bending_person_alerts_after_five_seconds_without_movement():
    events = run(seq([(2, stand), (1, bend), (7, lambda f: bend(1.0))]))
    assert [event.kind for event in events] == ["person_down"]


def test_slow_lie_down_is_reported_once_lying():
    evs = run(seq([(2, stand), (4, tilt), (3, lie)]))
    assert [e.kind for e in evs] == ["fall"]


def test_found_lying_reports_person_down():
    evs = run(seq([(12, lie)]))
    assert [e.kind for e in evs] == ["fall", "person_down"]
    assert evs[0].t <= 0.2
    assert abs(evs[1].t - 5.0) <= 1e-6
    assert evs[1].label == "POSSIBLE MEDICAL EMERGENCY"


def test_witnessed_fall_alerts_immediately_then_escalates():
    evs = run(seq([(2, stand), (0.5, tilt), (12, lie)]))
    assert [e.kind for e in evs] == ["fall", "person_down"]
    assert evs[0].t < 3.0
    assert 7.5 <= evs[1].t <= 8.3
    assert "medical emergency" in evs[1].info


def test_visible_limb_movement_blocks_medical_emergency():
    frames = []
    for i in range(120):
        k, c = lie(0)
        k[10, 0] += 20 if i % 2 else -20
        frames.append((i / FPS, k, c))
    evs = run(frames)
    assert [e.kind for e in evs] == ["fall"]


def test_slow_continuous_movement_restarts_stillness_timer():
    frames = []
    for i in range(120):
        k, c = lie(0)
        k[10, 0] += i * 1.5
        frames.append((i / FPS, k, c))
    evs = run(frames)
    assert [e.kind for e in evs] == ["fall"]


def test_movement_restarts_medical_stillness_timer():
    frames = []
    for i in range(160):
        k, c = lie(0)
        if i >= 40:
            k[10, 0] += 40
        frames.append((i / FPS, k, c))
    evs = run(frames)
    assert [e.kind for e in evs] == ["fall", "person_down"]
    assert evs[1].t >= 9.0


def test_visible_head_and_arms_are_enough_when_legs_are_occluded():
    frames = []
    for i in range(120):
        k, c = lie(0)
        c[[13, 14, 15, 16]] = 0.0
        frames.append((i / FPS, k, c))
    evs = run(frames)
    assert [e.kind for e in evs] == ["fall", "person_down"]
    assert abs(evs[1].t - 5.0) <= 1e-6


def test_brief_upright_pose_jitter_does_not_clear_a_latched_fall():
    detector = FallDetector(CFG)
    events = []
    for i in range(4):
        events += detector.update([person(1, *lie(0))], i / FPS)
    assert [e.kind for e in events] == ["fall"]

    detector.update([person(1, *stand(0))], 0.4)
    assert detector.states[1].status == "FALL!"
    detector.update([person(1, *lie(0))], 0.5)
    assert detector.states[1].status == "FALL!"
    assert detector.states[1].lying_since == 0.0

    for i in range(6, 18):
        detector.update([person(1, *stand(0))], i / FPS)
    assert detector.states[1].status == "OK"


def test_fall_banner_stays_visible_after_the_event_banner_expires():
    engine = Engine.__new__(Engine)
    engine.people = []  # pose model can temporarily miss someone fully lying down
    engine.fall = SimpleNamespace(states={
        13: SimpleNamespace(status="FALL!", fall_reported=True, down_reported=False,
                            last_box=(30, 30, 80, 80))
    })
    engine.banner = [(5.0, "FALL", (0, 0, 255))]

    active = Engine._active_fall_banners(engine)
    assert active[0][1] == "CAMERA | FALL DETECTED"
    assert Engine._draw_banner(engine, np.zeros((120, 640, 3), dtype=np.uint8), 20.0).any()


def test_last_fall_box_remains_visible_when_pose_is_missing():
    engine = Engine.__new__(Engine)
    engine.people = []
    engine.fall = SimpleNamespace(states={
        13: SimpleNamespace(status="FALL!", fall_reported=True, down_reported=False,
                            last_box=(20, 20, 80, 80))
    })
    engine.fire = None
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert Engine._draw(engine, frame).any()


def test_missing_pose_keeps_fall_but_does_not_count_as_stillness():
    detector = FallDetector(CFG)
    events = []
    for i in range(40):
        events += detector.update([person(13, *lie(0))], i / FPS)
    assert [e.kind for e in events] == ["fall"]

    for i in range(40, 46):
        events += detector.update([], i / FPS)
    assert [e.kind for e in events] == ["fall"]
    assert detector.states[13].status == "FALL!"

    for i in range(46, 97):
        events += detector.update([person(13, *lie(0))], i / FPS)
    assert [e.kind for e in events] == ["fall", "person_down"]
    assert detector.states[13].status == "MEDICAL!"


def test_camera_observes_stillness_when_blanket_hides_a_confirmed_person():
    detector = FallDetector(CFG)
    p = person(1, *lie(0))
    room = np.full((520, 640, 3), 180, dtype=np.uint8)
    room[400:472, 180:400] = (90, 75, 140)
    events = []
    for i in range(62):
        events += detector.update([p] if i < 15 else [], i / FPS, room)
    assert [ev.kind for ev in events] == ["fall", "person_down"]
    assert events[1].t == 5.0


def test_camera_observes_movement_after_pose_is_lost():
    detector = FallDetector(CFG)
    p = person(1, *lie(0))
    room = np.full((520, 640, 3), 180, dtype=np.uint8)
    room[400:472, 180:400] = (90, 75, 140)
    events = []
    for i in range(65):
        fr = room.copy()
        if i >= 15 and i % 2 == 1:
            fr[425:450, 230:282] = (185, 185, 185)
        events += detector.update([p] if i < 15 else [], i / FPS, fr)
    assert [ev.kind for ev in events] == ["fall"]


def falling_scene():
    room = np.full((520, 640, 3), 180, dtype=np.uint8)
    room[400:472, 180:400] = (90, 75, 140)
    return room, person(1, *lie(0))


def test_siren_stops_on_movement_and_waits_five_seconds_to_restart():
    room, p = falling_scene()
    d = FallDetector(CFG)
    eng = Engine.__new__(Engine)
    eng.fall = d
    eng._siren_active = False
    calls = []

    def start():
        eng._siren_active = True
        calls.append(("start", t))

    def stop():
        eng._siren_active = False
        calls.append(("stop", t))

    eng._start_siren, eng._stop_siren = start, stop
    shifted = room.copy()
    shifted[418:455, 230:292] = (185, 185, 185)
    for i in range(113):
        t = i / FPS
        d.update([p], t, room if t < 6 else shifted)
        eng._sync_siren()
    assert calls[:2] == [("start", 5.0), ("stop", 6.0)]
    assert calls[2][0] == "start" and calls[2][1] >= 11.0
    t = 11.3
    d.update([person(1, *stand(0))], t)
    eng._sync_siren()
    assert calls[-1] == ("stop", 11.3)


def test_person_leaving_clears_emergency_and_bounding_box():
    room, p = falling_scene()
    d = FallDetector(CFG)
    for i in range(61):
        d.update([p], i / FPS, room)
    assert d.states[1].down_reported
    d.update([], 6.1, room)
    assert d.states[1].down_reported  # a single occluded frame can be tolerated
    empty = np.full_like(room, 180)
    d.update([], 6.8, empty)
    assert not d.states


def test_matched_image_cannot_keep_missing_emergency_forever():
    room, p = falling_scene()
    d = FallDetector(CFG)
    for i in range(51):
        d.update([p], i / FPS, room)
    assert d.states[1].down_reported
    for i in range(51, 111):
        d.update([], i / FPS, room)
    assert not d.states


def test_sustained_non_lying_posture_ends_medical_state():
    d = FallDetector(CFG)
    for i in range(51):
        d.update([person(1, *lie(0))], i / FPS)
    assert d.states[1].down_reported
    d.posture = lambda _p: ("mid", 45)
    for i in range(51, 62):
        d.update([person(1, *lie(0))], i / FPS)
    assert d.states[1].status == "MEDICAL!"
    assert d.states[1].down_reported


def test_camera_shake_does_not_restart_stillness_for_a_motionless_person():
    detector = FallDetector(CFG)
    rng = np.random.default_rng(11)
    room = rng.integers(40, 200, (520, 640, 3), dtype=np.uint8)
    room = cv2.GaussianBlur(room, (11, 11), 0)
    room[70:230, 45:140] = (170, 190, 70)
    room[400:472, 180:400] = (90, 75, 140)
    readings = []
    for dx, dy in [(0, 0), (5, -3), (-5, 2), (0, 0)]:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        fr = cv2.warpAffine(room, matrix, (640, 520), borderMode=cv2.BORDER_REFLECT)
        k, c = lie(0)
        # Model estimates can jump by more than the actual camera displacement.
        k[10, 0] += 3 * dx
        readings.append((fr, person(1, k + [dx, dy], c)))
    events = []
    for i in range(40):
        fr, p = readings[i % len(readings)]
        events += detector.update([p], i / 6.2, fr)
    assert [ev.kind for ev in events] == ["fall", "person_down"]
    assert 5 <= events[-1].t < 5.2


def test_wide_box_with_only_stray_ankle_points_never_becomes_a_fall():
    detector = FallDetector(CFG)
    p = person(1, *lie(0))
    p.conf[:] = 0.05
    p.conf[[15, 16]] = [0.34, 0.41]
    for i in range(110):
        assert detector.update([p], i / FPS) == []
    assert detector.states == {}


def test_switching_from_a_person_to_a_fire_scene_clears_fall():
    detector = FallDetector(CFG)
    room = np.full((520, 640, 3), 180, dtype=np.uint8)
    fire = np.full_like(room, (5, 20, 115))
    for i in range(5):
        detector.update([person(1, *lie(0))], i / FPS, room)
    detector.update([], 0.5, room)
    assert detector.states[1].status == "FALL!"
    assert detector.update([], 0.6, fire) == []
    assert detector.states == {}


def test_lying_person_with_new_id_keeps_original_fall_without_duplicate():
    detector = FallDetector(CFG)
    for i in range(4):
        detector.update([person(1, *lie(0))], i / FPS)
    detector.update([], 0.6)
    assert detector.update([person(2, *lie(0))], 0.9) == []
    assert list(detector.states) == [2]
    assert detector.states[2].status == "FALL!"


def test_mid_pose_jitter_does_not_erase_stillness_after_fall():
    detector = FallDetector(CFG)
    for i in range(40):
        detector.update([person(1, *lie(0))], i / FPS)
    previous_still_since = detector.states[1].still_since
    detector.posture = lambda _p: ("mid", None)
    detector.update([person(1, *lie(0))], 4.0)
    assert detector.states[1].still_since == previous_still_since


def test_missing_tracker_id_clears_after_same_person_stays_upright():
    detector = FallDetector(CFG)
    for i in range(4):
        detector.update([person(13, *lie(0))], i / FPS)
    assert detector.states[13].status == "FALL!"

    detector.update([person(14, *stand(0))], 0.4)
    assert 13 in detector.states
    for i in range(5, 18):
        detector.update([person(14, *stand(0))], i / FPS)
    assert 13 not in detector.states


def test_siren_only_tracks_medical_emergencies(monkeypatch):
    engine = Engine.__new__(Engine)
    engine.cfg = {"alerts": {"siren_enabled": True}}
    state = SimpleNamespace(down_reported=False)
    engine.fall = SimpleNamespace(states={1: state})
    engine._siren_active = False
    calls = []

    def start():
        calls.append("start")
        engine._siren_active = True

    def stop():
        calls.append("stop")
        engine._siren_active = False

    monkeypatch.setattr(engine, "_start_siren", start)
    monkeypatch.setattr(engine, "_stop_siren", stop)
    Engine._sync_siren(engine)
    assert calls == []
    state.down_reported = True
    Engine._sync_siren(engine)
    Engine._sync_siren(engine)
    assert calls == ["start"]
    state.down_reported = False
    Engine._sync_siren(engine)
    assert calls == ["start", "stop"]


def test_each_visible_person_gets_an_independent_inactivity_alert():
    detector = FallDetector(CFG)
    first = person(1, *stand(0))
    second = person(2, *stand(0))
    events = []
    for i in range(62):
        events += detector.update([first, second], i / FPS)
    medical = [event for event in events if event.kind == "person_down"]
    assert [event.subject_id for event in medical] == [1, 2]


def test_brief_pose_gap_does_not_restart_an_active_medical_alarm():
    room, p = falling_scene()
    detector = FallDetector(CFG)
    for i in range(51):
        detector.update([p], i / FPS, room)
    assert detector.states[1].down_reported
    detector.update([], 5.1, room)
    assert detector.states[1].down_reported
    detector.update([p], 5.2, room)
    assert detector.states[1].down_reported


def test_movement_or_disappearance_clears_the_medical_alarm():
    room, p = falling_scene()
    detector = FallDetector(CFG)
    engine = Engine.__new__(Engine)
    engine.fall = detector
    engine.cfg = {"alerts": {"siren_enabled": True}}
    engine._siren_active = False
    calls = []

    def start():
        calls.append("start")
        engine._siren_active = True

    def stop():
        calls.append("stop")
        engine._siren_active = False

    engine._start_siren, engine._stop_siren = start, stop
    for i in range(51):
        detector.update([p], i / FPS, room)
        engine._sync_siren()
    assert calls == ["start"]
    blank = np.full_like(room, 180)
    for i in range(51, 59):
        detector.update([], i / FPS, blank)
    engine._sync_siren()
    assert calls == ["start", "stop"]
    assert detector.states == {}


def test_no_torso_keypoints_box_fallback():
    import numpy as np
    def hide(fn):
        def g(f):
            k, c = fn(f); c = c.copy(); c[[5, 6, 11, 12]] = 0.0; return k, c
        return g
    # with hidden torso keypoints the box aspect must still reveal the fall
    evs = run(seq([(2, hide(stand)), (0.5, hide(tilt)), (3, hide(lie))]))
    assert [e.kind for e in evs] == ["fall"]
