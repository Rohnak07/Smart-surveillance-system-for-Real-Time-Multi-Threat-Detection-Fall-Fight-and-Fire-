import numpy as np
import cv2
import yaml
from kavach.fire import FireDetector

CFG = yaml.safe_load(open("config.yaml"))["fire"]
CFG["mode"] = "heuristic"
W, H, FPS = 640, 360, 10


def background(rng):
    bg = np.full((H, W, 3), (90, 95, 100), np.uint8)
    return cv2.add(bg, rng.integers(0, 6, bg.shape, dtype=np.uint8))


def flame(frame, rng, cx=320, base=300, size=90, with_core=True):
    """A tapered, bottom-anchored flame whose tip and height flicker each frame."""
    h = int(size * rng.uniform(1.25, 1.9))
    w = int(size * rng.uniform(0.55, 0.9))
    tip_x = cx + int(rng.uniform(-0.12, 0.12) * w)
    pts = np.array([
        (tip_x, base - h),
        (cx + int(0.15 * w), base - int(0.72 * h)),
        (cx + int(0.42 * w), base - int(0.36 * h)),
        (cx + int(0.50 * w), base - int(0.12 * h)),
        (cx + int(0.36 * w), base),
        (cx - int(0.38 * w), base),
        (cx - int(0.50 * w), base - int(0.16 * h)),
        (cx - int(0.30 * w), base - int(0.48 * h)),
        (cx - int(0.12 * w), base - int(0.76 * h)),
    ], np.int32)
    cv2.fillPoly(frame, [pts], (20, 120, 255))
    if with_core:
        core = np.array([
            (tip_x, base - int(0.72 * h)), (cx + int(0.20 * w), base - int(0.35 * h)),
            (cx + int(0.22 * w), base), (cx - int(0.22 * w), base),
            (cx - int(0.18 * w), base - int(0.37 * h)),
        ], np.int32)
        cv2.fillPoly(frame, [core], (60, 200, 255))


def run(make, seconds=6, **overrides):
    d, evs, rng = FireDetector(dict(CFG, **overrides)), [], np.random.default_rng(0)
    for i in range(int(seconds * FPS)):
        evs += d.update(make(rng), i / FPS)
    return evs


def test_flickering_flame_is_fire():
    def f(rng):
        fr = background(rng); flame(fr, rng); return fr
    evs = run(f)
    assert evs and evs[0].kind == "fire"
    assert evs[0].t <= 1.0


def test_moderately_flickering_flame_is_detected():
    def f(rng):
        t = f.index / FPS
        h = 180 + int(16 * np.sin(2 * np.pi * 0.5 * t))
        w, cx, base = 130, 320, 300
        tip = cx + int(5 * np.sin(2 * np.pi * 0.5 * t))
        fr = background(rng)
        pts = np.array([
            (tip, base - h), (cx + int(0.15 * w), base - int(0.72 * h)),
            (cx + int(0.42 * w), base - int(0.36 * h)),
            (cx + int(0.50 * w), base - int(0.12 * h)),
            (cx + int(0.36 * w), base), (cx - int(0.38 * w), base),
            (cx - int(0.50 * w), base - int(0.16 * h)),
            (cx - int(0.30 * w), base - int(0.48 * h)),
            (cx - int(0.12 * w), base - int(0.76 * h)),
        ], np.int32)
        cv2.fillPoly(fr, [pts], (20, 120, 255))
        core = np.array([
            (tip, base - int(0.72 * h)), (cx + int(0.20 * w), base - int(0.35 * h)),
            (cx + int(0.22 * w), base), (cx - int(0.22 * w), base),
            (cx - int(0.18 * w), base - int(0.37 * h)),
        ], np.int32)
        cv2.fillPoly(fr, [core], (60, 200, 255))
        f.index += 1
        return fr
    f.index = 0
    evs = run(f)
    assert evs and evs[0].kind == "fire"
    assert evs[0].t <= 4.0


def test_flickering_fire_video_on_laptop_screen_is_detected():
    def f(rng):
        fr = background(rng)
        # A laptop display fully inside the camera view (not cropped by frame edges).
        cv2.rectangle(fr, (100, 80), (540, 280), (35, 35, 35), -1)
        cv2.rectangle(fr, (100, 80), (540, 280), (180, 180, 180), 6)
        flame(fr, rng, cx=320, base=270, size=65)
        return fr
    assert run(f)
    demo = run(f, screen_demo=True)
    assert demo and demo[0].label == "FIRE DETECTED"


def test_phone_exposure_does_not_hide_a_flickering_flame_core():
    def f(rng):
        fr = background(rng)
        flame(fr, rng)
        fr[(fr == (60, 200, 255)).all(axis=2)] = (95, 139, 190)
        fr[(fr == (20, 120, 255)).all(axis=2)] = (25, 87, 188)
        return fr
    assert run(f)


def test_small_flame_on_portrait_phone_feed_is_detected():
    d = FireDetector(dict(CFG))
    rng = np.random.default_rng(0)
    events = []
    for i in range(24):
        fr = np.full((736, 414, 3), (90, 95, 100), np.uint8)
        flame(fr, rng, cx=260, base=410, size=16)
        events += d.update(fr, i / 5)
    assert events and events[0].kind == "fire"


def test_screen_demo_toggle_keeps_stationary_flame_images_from_alerting():
    frame = background(np.random.default_rng(0))
    cv2.rectangle(frame, (100, 80), (540, 280), (35, 35, 35), -1)
    flame(frame, np.random.default_rng(1), cx=320, base=270, size=65)
    d = FireDetector(dict(CFG))
    d.toggle_screen_demo()
    assert d.screen_demo
    assert not any(d.update(frame, i / FPS) for i in range(20))
    d.toggle_screen_demo()
    assert not d.screen_demo
    assert d.prev_mask is None


def test_rigid_moving_flame_coloured_object_cannot_hide_a_separate_flame():
    d = FireDetector(dict(CFG))
    rng = np.random.default_rng(9)
    events = []
    for i in range(40):
        fr = np.full((H, W, 3), (90, 95, 100), np.uint8)
        flame(fr, rng, cx=190, base=300, size=45)
        flame(fr, np.random.default_rng(2), cx=455, base=190 + (20 if i % 2 else 0), size=70)
        events += d.update(fr, i / FPS)
    assert events and all(ev.box[2] < 300 for ev in events)


def test_clipped_orange_yellow_object_at_side_is_not_fire():
    for demo in (False, True):
        for size, center in ((30, 625), (45, 635), (75, 635)):
            d = FireDetector(dict(CFG, screen_demo=demo))
            for i in range(35):
                t = i / FPS
                fr = np.full((H, W, 3), (90, 95, 100), np.uint8)
                # A rigid two-color object slides in and out of the frame.
                flame(fr, np.random.default_rng(2), cx=center + int(12 * np.sin(2 * np.pi * t)),
                      base=270, size=size)
                assert d.update(fr, t) == []


def test_bright_screen_without_flame_does_not_alert():
    def f(rng):
        fr = background(rng)
        cv2.rectangle(fr, (65, 40), (575, 320), (180, 180, 180), -1)
        cv2.rectangle(fr, (85, 60), (555, 300), (255, 255, 255), -1)
        cv2.circle(fr, (320, 180), 48, (0, 255, 255), -1)
        return fr
    assert run(f) == []


def test_flickering_solid_orange_flame_shape_without_core_is_not_fire():
    def f(rng):
        fr = background(rng); flame(fr, rng, with_core=False); return fr
    assert run(f) == []


def test_static_orange_curtain_no_alert():
    def f(rng):
        fr = background(rng)
        cv2.rectangle(fr, (100, 60), (260, 300), (20, 120, 255), -1)           # orange curtain
        return fr
    assert run(f) == []


def test_static_orange_object_with_sensor_noise_no_alert():
    def f(rng):
        fr = background(rng)
        cv2.circle(fr, (400, 200), 70, (20, 130, 255), -1)
        noise = rng.integers(-4, 5, fr.shape).astype(np.int16)
        return np.clip(fr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    assert run(f) == []


def test_plain_scene_no_alert():
    assert run(lambda rng: background(rng)) == []


def test_whole_frame_orange_sunset_no_alert():
    def f(rng):
        return np.full((H, W, 3), (30, 140, 255), np.uint8)
    assert run(f) == []


def test_moving_person_skin_is_not_fire():
    for use_person_box in (False, True):
        detector = FireDetector(dict(CFG))
        rng = np.random.default_rng(0)
        events = []
        for i in range(60):
            frame = background(rng)
            x = 120 + (i * 9) % 400
            cv2.ellipse(frame, (x, 180), (70, 110), 0, 0, 360, (60, 110, 200), -1)
            person_box = [(x - 70, 70, x + 70, 290)] if use_person_box else None
            events += detector.update(frame, i / FPS, person_box)
        assert events == []


def test_moving_red_object_is_not_fire():
    def f(rng):
        fr = background(rng)
        t = f.index / FPS
        y = 120 + int(18 * np.sin(2 * np.pi * 0.6 * t))
        cv2.rectangle(fr, (275, y), (345, y + 55), (10, 10, 255), -1)
        f.index += 1
        return fr
    f.index = 0
    assert run(f) == []


def test_red_wire_moving_up_and_down_is_not_fire():
    def f(rng):
        fr = background(rng)
        t = f.index / FPS
        y = 100 + int(20 * np.sin(2 * np.pi * 0.5 * t))
        cv2.line(fr, (320, y), (320, y + 150), (10, 10, 255), 12)
        f.index += 1
        return fr
    f.index = 0
    assert run(f) == []


def test_red_shirt_on_detected_person_is_ignored():
    detector = FireDetector(dict(CFG))
    rng = np.random.default_rng(0)
    events = []
    for i in range(60):
        frame = background(rng)
        x = 250 + (i % 10) * 2
        cv2.circle(frame, (x, 100), 35, (10, 10, 255), -1)
        cv2.rectangle(frame, (x - 45, 120), (x + 45, 300), (10, 10, 255), -1)
        events += detector.update(frame, i / FPS, [(x - 55, 60, x + 55, 310)])
    assert events == []
