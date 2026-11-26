"""Fast fight detection from repeated limb strikes between tracked people.

Arms detect punches/slaps and legs detect kicks. A hit counts only when a fast
limb enters another person's box; it must leave before it can count again.
Repeated hits are accumulated per pair, so unrelated greetings in a crowd do
not combine into one fight alert. Connected multi-person fights can accumulate
as a group. Static contact (a hand on a shoulder), close standing, and one-off
contact do not alert.
"""
from collections import defaultdict, deque
import numpy as np
from .types import Event, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR
from .types import L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK

LIMBS = {
    "arm": ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR), 0.35),
    "leg": ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK), 0.12),
}


class _P:
    def __init__(self):
        self.prev = {}                 # (limb, side) -> (time, tip relative to anchor)
        self.h_ema = None
        self.last_strike = {}          # (limb, side) -> time
        self.contact_target = {}       # latch until the limb leaves the target box


class FightDetector:
    def __init__(self, cfg):
        self.c = cfg
        self.ps = {}
        self.points = deque()          # (time, weight, description, person pair)
        self.last_score = 0.0

    def _tip(self, p, limb, side):
        chains = LIMBS[limb][:2]
        anchor, joint, tip = chains[side]
        if p.conf[tip] < 0.30 or p.conf[anchor] < 0.25:
            return None
        point = p.kpts[tip]
        if p.conf[joint] >= 0.20:
            point = point + LIMBS[limb][2] * (point - p.kpts[joint])
        return point

    def _speed(self, p, st, limb, side, t):
        anchor, _, tip = LIMBS[limb][:2][side]
        key = (limb, side)
        if p.conf[tip] < 0.30 or p.conf[anchor] < 0.25:
            st.prev.pop(key, None)
            return None
        rel = p.kpts[tip] - p.kpts[anchor]
        prev = st.prev.get(key)
        if prev is None or t - prev[0] < 0.06:
            st.prev[key] = (t, rel) if prev is None else prev
            return None
        st.prev[key] = (t, rel)
        if t - prev[0] > 0.6:
            return None
        scale = max(st.h_ema or p.h, 1.0)
        return float(np.linalg.norm(rel - prev[1]) / (t - prev[0]) / scale)

    def _target(self, point, attacker, people):
        margin = self.c["contact_margin"]
        candidates = []
        for target in people:
            if target.id == attacker.id:
                continue
            mx, my = margin * target.w, margin * target.h
            if (target.box[0] - mx <= point[0] <= target.box[2] + mx and
                    target.box[1] - my <= point[1] <= target.box[3] + my):
                distance = np.linalg.norm(point - target.center) / max(target.h, 1.0)
                candidates.append((distance, target))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def update(self, people, t, cnn_prob=None):
        c, events = self.c, []
        alive = {p.id for p in people}
        self.ps = {k: v for k, v in self.ps.items() if k in alive}
        by_id = {p.id: p for p in people}

        if len(people) >= c.get("min_people", 2):
            for attacker in people:
                st = self.ps.setdefault(attacker.id, _P())
                st.h_ema = attacker.h if st.h_ema is None else 0.9 * st.h_ema + 0.1 * attacker.h
                for limb in LIMBS:
                    for side in (0, 1):
                        key = (limb, side)
                        speed = self._speed(attacker, st, limb, side, t)
                        speed = speed if speed is not None and speed <= c["max_speed"] else None
                        tip = self._tip(attacker, limb, side)
                        target = self._target(tip, attacker, people) if tip is not None else None
                        old_target = st.contact_target.get(key)

                        if target is None:
                            st.contact_target.pop(key, None)
                            continue

                        # Count one hit on entry. A hand held on someone's shoulder/head
                        # cannot repeatedly score; it must leave and make a new impact.
                        st.contact_target[key] = target.id
                        if target.id == old_target or speed is None or speed < c["strike_speed"]:
                            continue
                        last = st.last_strike.get(key, -9.0)
                        if t - last < c["strike_refractory_s"]:
                            continue
                        st.last_strike[key] = t
                        pair = tuple(sorted((attacker.id, target.id)))
                        self.points.append((t, 1.0, f"{limb} {attacker.id}->{target.id}", pair))

        while self.points and t - self.points[0][0] > c["window_s"]:
            self.points.popleft()

        pair_scores = defaultdict(float)
        for _, value, _, pair in self.points:
            pair_scores[pair] += value
        pair_scores = dict(pair_scores)

        # A pair with two distinct impacts alerts quickly. In a group fight,
        # three impacts connected through shared people can also alert.
        groups = _connected_groups(pair_scores)
        group_candidates = []
        group_min = c.get("group_alert_score", 3.0)
        for ids in groups:
            score = sum(value for pair, value in pair_scores.items() if set(pair).issubset(ids))
            if score >= group_min:
                group_candidates.append((score, ids))

        cnn_agrees = cnn_prob is not None and cnn_prob >= c.get("cnn_threshold", 0.85)
        need = c["alert_score"]
        pair_candidates = [(score, pair) for pair, score in pair_scores.items()
                           if score >= need or (cnn_agrees and score >= 0.4 * need)]
        self.last_score = max([*pair_scores.values(), *(score for score, _ in group_candidates)], default=0.0)

        if pair_candidates or group_candidates:
            pair_choice = max(pair_candidates, default=None, key=lambda item: item[0])
            group_choice = max(group_candidates, default=None, key=lambda item: item[0])
            if pair_choice is not None and (group_choice is None or pair_choice[0] >= need):
                score, pair = pair_choice
                ids = set(pair)
            else:
                score, ids = group_choice
            involved = [by_id[pid] for pid in ids if pid in by_id]
            if len(involved) >= 2:
                label = "people " + ", ".join(f"#{pid}" for pid in sorted(ids))
                conf = min(0.99, 0.60 + 0.10 * score)
                events.append(Event("fight", conf, t,
                                    f"repeated strikes: {label}, score {score:.1f} in {c['window_s']:.0f}s",
                                    tuple(map(int, _union(involved))), tuple(sorted(ids))))
                self.points.clear()
        return events


def _connected_groups(pair_scores):
    adjacency = defaultdict(set)
    for a, b in pair_scores:
        adjacency[a].add(b)
        adjacency[b].add(a)
    groups, seen = [], set()
    for person_id in adjacency:
        if person_id in seen:
            continue
        group, stack = set(), [person_id]
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            seen.add(current)
            stack.extend(adjacency[current] - group)
        groups.append(group)
    return groups


def _union(people):
    boxes = np.array([p.box for p in people])
    return boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
