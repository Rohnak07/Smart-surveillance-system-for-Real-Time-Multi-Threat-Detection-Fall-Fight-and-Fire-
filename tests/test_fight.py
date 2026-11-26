import math
import yaml
import numpy as np
from kavach.fight import FightDetector
from kavach.pipeline import Engine
from kavach.types import Event
from tests.synth import skeleton, person

CFG = yaml.safe_load(open("config.yaml"))["fight"]
H, FOOT, FPS = 300, 460, 10


def two(t, ax, bx, a_arm=None, b_arm=None, a_leg=None, b_leg=None):
    ka, ca = skeleton(ax, FOOT, H, arms=a_arm, legs=a_leg)
    kb, cb = skeleton(bx, FOOT, H, arms=b_arm, legs=b_leg)
    return [person(1, ka, ca), person(2, kb, cb)]


def run(gen, seconds, cfg=CFG):
    d, evs = FightDetector(cfg), []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        evs += d.update(gen(t), t)
    return evs


def punch_arm(t, toward_x, rate=2.5, reach=0.55):
    """Right wrist of attacker thrusts toward target at `rate` punches/s."""
    ph = (t * rate) % 1.0
    ext = math.sin(math.pi * min(ph * 2, 1.0)) if ph < 0.5 else 0.0
    sh = (None)
    return ext


def test_sustained_punching_is_fight():
    def gen(t):
        ext = punch_arm(t, 0)
        wx = 300 + 0.12 * H + ext * 0.30 * H           # hand lands on B (standing at x=420)
        return two(t, 300, 420, a_arm={10: (wx, 280), 8: (300 + 0.2 * H + ext * 0.2 * H, 270)})
    evs = run(gen, 6)
    assert evs and evs[0].kind == "fight"
    assert evs[0].t <= 1.0


def test_walking_side_by_side_no_alert():
    assert run(lambda t: two(t, 100 + 30 * t, 220 + 30 * t), 8) == []


def test_standing_close_no_alert():
    assert run(lambda t: two(t, 300, 400), 8) == []


def test_waving_to_someone_far_away_no_alert():
    def gen(t):
        wy = 200 + 60 * math.sin(2 * math.pi * 2.5 * t)
        return two(t, 100, 700, a_arm={10: (150, wy), 8: (140, 260)})
    assert run(gen, 8) == []


def test_slow_handshake_no_alert():
    def gen(t):
        off = 8 * math.sin(2 * math.pi * 0.8 * t)
        return two(t, 300, 400, a_arm={10: (350 + off, 300)}, b_arm={9: (350 - off, 300)})
    assert run(gen, 8) == []


def test_single_high_five_burst_no_alert():
    def gen(t):
        ext = math.sin(math.pi * min(max((t - 2.0) / 0.4, 0), 1)) if 2.0 <= t <= 2.4 else 0.0
        return two(t, 300, 420, a_arm={10: (312 + ext * 160, 280)})
    assert run(gen, 6) == []


def test_one_person_shadow_boxing_no_alert():
    def gen(t):
        ext = punch_arm(t, 0)
        k, c = skeleton(300, FOOT, H, arms={10: (312 + ext * 160, 280)})
        return [person(1, k, c)]
    assert run(gen, 8) == []


def test_mutual_scuffle_is_fight():
    def gen(t):
        s = math.sin(2 * math.pi * 2.2 * t)
        return two(t, 300, 400,
                   a_arm={10: (340 + 60 * s, 280 + 40 * s), 9: (260 - 40 * s, 300)},
                   b_arm={9: (360 - 60 * s, 280 - 40 * s), 10: (440 + 40 * s, 300)})
    evs = run(gen, 8)
    assert evs and evs[0].kind == "fight"


def test_repeated_kicks_are_fight():
    def gen(t):
        ext = punch_arm(t, 0)
        ankle_x = 324 + ext * 150
        knee_x = 324 + ext * 90
        return two(t, 300, 420, a_leg={14: (knee_x, 390), 16: (ankle_x, 430)})
    evs = run(gen, 6)
    assert evs and evs[0].kind == "fight"


def test_hair_tug_held_at_head_does_not_repeat_hits():
    def gen(t):
        tug = 60 * math.sin(2 * math.pi * 2.5 * t)
        wrist_x = 410 + tug
        elbow_x = wrist_x - 25
        return two(t, 300, 420, a_arm={8: (elbow_x, 175), 10: (wrist_x, 175)})
    assert run(gen, 6) == []


def test_resting_hand_on_shoulder_no_alert():
    def gen(t):
        return two(t, 300, 420, a_arm={8: (370, 205), 10: (390, 205)})
    assert run(gen, 6) == []


def test_pair_fight_detected_with_seven_people_in_scene():
    def gen(t):
        ext = punch_arm(t, 0)
        wx = 300 + 0.12 * H + ext * 0.30 * H
        fighting = two(t, 300, 420, a_arm={10: (wx, 280), 8: (300 + 0.2 * H + ext * 0.2 * H, 270)})
        crowd = []
        for pid, cx in enumerate((30, 90, 700, 780, 860), start=3):
            k, c = skeleton(cx, FOOT, H)
            crowd.append(person(pid, k, c))
        return fighting + crowd
    evs = run(gen, 6)
    assert evs and evs[0].kind == "fight"


def test_connected_hits_across_four_people_alert_as_group():
    def gen(t):
        centers = (300, 420, 540, 660)
        starts = (0.0, 0.6, 1.2)
        people = []
        for index, cx in enumerate(centers):
            arms = None
            if index < len(starts):
                phase = (t - starts[index]) / 0.4
                ext = math.sin(math.pi * phase) if 0.0 <= phase <= 1.0 else 0.0
                arms = {10: (cx + 0.12 * H + ext * 0.30 * H, 280),
                        8: (cx + 0.20 * H + ext * 0.20 * H, 270)}
            k, c = skeleton(cx, FOOT, H, arms=arms)
            people.append(person(index + 1, k, c))
        return people
    evs = run(gen, 6)
    assert evs and evs[0].kind == "fight"
    assert "#1, #2, #3, #4" in evs[0].info


def test_fight_and_fire_alerts_can_emit_on_same_frame():
    class Detector:
        def __init__(self, event, mode=None):
            self.event = event
            self.mode = mode

        def update(self, *_args):
            return [self.event]

    class Sink:
        def __init__(self):
            self.kinds = []

        def emit(self, event, _frame):
            self.kinds.append(event.kind)
            return True

    engine = Engine.__new__(Engine)
    engine.pose = None
    engine.fall = None
    engine.people = []
    engine.fight = Detector(Event("fight", 0.8, 1.0))
    engine.cnn = None
    engine.fire = Detector(Event("fire", 0.8, 1.0), mode="yolo")
    engine.banner = []
    engine.sink = Sink()
    engine._draw = lambda frame: frame.copy()
    engine._draw_banner = lambda frame, _t: frame
    engine._hud = lambda frame, _t: frame

    events, _ = engine.process(np.zeros((32, 32, 3), dtype=np.uint8), 1.0)
    assert [event.kind for event in events] == ["fight", "fire"]
    assert engine.sink.kinds == ["fight", "fire"]
