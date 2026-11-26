"""Fall detection and person-specific inactivity alerts.

Pose and box geometry distinguish lying from sitting or bending. A lying posture
confirmed for a short interval raises a fall alert. Independently, a visible
person with no detected movement for the configured interval raises a possible
medical-emergency alert. Both alerts are for human review.
"""
import math
import cv2
import numpy as np
from .tracker import _iou
from .types import (Event, L_SH, R_SH, L_HIP, R_HIP, NOSE,
                    L_EL, R_EL, L_WR, R_WR, L_KNEE, R_KNEE, L_ANK, R_ANK)

HEAD_POINTS = (NOSE, 1, 2, 3, 4)
ARM_POINTS = (L_EL, R_EL, L_WR, R_WR)
LEG_POINTS = (L_KNEE, R_KNEE, L_ANK, R_ANK)
UPPER_BODY_POINTS = (*HEAD_POINTS, L_SH, R_SH, *ARM_POINTS)
LOWER_BODY_POINTS = (L_HIP, R_HIP, *LEG_POINTS)


def has_body_pose(keypoints, confidence, min_conf=0.30):
    """Reject person boxes supported by only a face or a couple of stray joints."""
    visible = ((confidence >= min_conf) & np.isfinite(keypoints).all(axis=1)
               & (keypoints > 0).any(axis=1))
    return (np.count_nonzero(visible) >= 4
            and np.any(visible[list(UPPER_BODY_POINTS)])
            and np.any(visible[list(LOWER_BODY_POINTS)]))


def torso_angle_deg(p, min_conf=0.30):
    """Angle of shoulder-mid -> hip-mid vector from vertical (0 = upright, 90 = horizontal)."""
    c = p.conf
    if min(c[L_SH], c[R_SH], c[L_HIP], c[R_HIP]) < min_conf:
        return None
    sh = (p.kpts[L_SH] + p.kpts[R_SH]) / 2
    hp = (p.kpts[L_HIP] + p.kpts[R_HIP]) / 2
    v = hp - sh
    if np.linalg.norm(v) < 8:           # too small / degenerate
        return None
    return math.degrees(math.atan2(abs(v[0]), abs(v[1]) + 1e-6))


class _State:
    def __init__(self):
        self.track_id = None
        self.lying_since = None         # t when posture first became lying (debounced)
        self.pending_lying = None       # t of first raw lying sample
        self.last_up_t = None
        self.last_up_cy = None
        self.last_up_h = None
        self.fall_t = None              # lying_since of a fall awaiting confirmation
        self.fall_reported = False
        self.down_reported = False
        self.up_since = None
        self.status = "OK"
        self.motion_points = {}
        self.motion_anchor = {}
        self.motion_t = None
        self.still_since = None
        self.last_seen_t = None
        self.last_box = None
        self.recovery_person = None
        self.recovery_since = None
        self.last_view = None
        self.visual_box = None
        self.visual_anchor = None
        self.visual_previous = None
        self.visual_t = None
        self.visual_since = None
        self.last_camera_offset = np.zeros(2)
        self.still_progress = None
        self.pose_moved = False
        self.visual_moved = False


class FallDetector:
    def __init__(self, cfg):
        self.c = cfg
        self.states = {}
        self.scene = None
        self.camera_reference = None
        self.camera_offset = np.zeros(2)
        self.camera_window = cv2.createHanningWindow((192, 144), cv2.CV_32F)

    def posture(self, p):
        c = self.c
        asp = p.w / max(p.h, 1.0)
        ang = torso_angle_deg(p)
        if ang is not None:
            if ang >= c["lying_angle_deg"] and asp >= c["lying_min_aspect"]:
                return "lying", ang
            if ang <= c["upright_angle_deg"] and asp < 0.9:
                return "up", ang
            return "mid", ang
        if asp >= c["lying_aspect_no_torso"]:
            return "lying", None
        if asp < 0.8:
            return "up", None
        return "mid", None

    def update(self, people, t, frame=None):
        c, events = self.c, []
        people = [p for p in people if has_body_pose(p.kpts, p.conf)]
        if frame is not None:
            scene = cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA)
            if (self.scene is not None
                    and np.mean(cv2.absdiff(scene, self.scene).mean(axis=2) > 35) > 0.65):
                self.states.clear()
                self.camera_reference = None
            self.scene = scene
            self._estimate_camera_offset(frame, people)
        alive = {p.id for p in people}
        self._reattach_lying_tracks(people, alive, t, frame)
        for pid in [k for k in self.states if k not in alive]:
            s = self.states[pid]
            s.up_since = None
            gap = t - (s.last_seen_t if s.last_seen_t is not None else t)
            if s.fall_reported:
                if self._clear_missing_state_after_recovery(s, people, t):
                    del self.states[pid]
                    continue
                same_view = self._same_view(s, frame)
                # Only a matching live image can extend a confirmed fall through
                # a temporary pose-estimation gap.
                hold = c.get("medical_visual_hold_s", 6.0) if same_view else c.get("missing_hold_s", 0.8)
                if gap + 1e-6 >= hold:
                    s.down_reported = False
                    del self.states[pid]
                    continue
                if same_view and s.recovery_since is None and s.visual_box is not None:
                    still_for = self._visual_stillness(s, frame, t, pose_visible=False)
                    self._update_medical(s, t, still_for, s.visual_moved, events)
                else:
                    self._reset_visual_motion(s)
                    s.still_progress = None
                continue
            if s.down_reported and gap + 1e-6 >= c.get("missing_hold_s", 0.8):
                s.down_reported = False
                del self.states[pid]
                continue
            if gap > 3.0:
                del self.states[pid]
            else:
                s.pending_lying = None
                s.lying_since = None
                s.fall_t = None
                if gap > c.get("medical_max_gap_s", 0.75):
                    s.motion_points.clear()
                    s.motion_anchor.clear()
                    s.motion_t = s.still_since = None
                    self._reset_visual_motion(s)
                if not s.down_reported:
                    s.status = "OK"

        for p in people:
            s = self.states.setdefault(p.id, _State())
            s.track_id = p.id
            s.last_seen_t = t
            s.last_box = tuple(map(float, p.box))
            s.last_camera_offset = self.camera_offset.copy()
            post, _ = self.posture(p)
            s.last_view = self._box_view(frame, s.last_box)
            cy = float(p.center[1])

            pose_still = self._stationary_for(s, p, t)
            visual_still = self._visual_stillness(s, frame, t, pose_visible=True)
            still_for = visual_still if visual_still is not None else pose_still
            moved = s.visual_moved or (visual_still is None and s.pose_moved)
            self._update_medical(s, t, still_for, moved, events)

            if post == "up":
                s.last_up_t, s.last_up_cy, s.last_up_h = t, cy, p.h
                s.up_since = s.up_since if s.up_since is not None else t
                s.pending_lying = None
                if s.up_since is not None and t - s.up_since >= c["recover_s"]:
                    s.fall_t = None
                    s.fall_reported = False
                    s.lying_since = None
                    s.status = "MEDICAL!" if s.down_reported else "OK"
                elif not s.fall_reported:
                    # Before a fall is confirmed, an upright sample cancels lying.
                    s.lying_since = None
                    s.status = "MEDICAL!" if s.down_reported else "OK"
                else:
                    # Keep a reported fall latched through brief pose jitter. It
                    # clears only after the person is upright for recover_s.
                    s.status = "MEDICAL!" if s.down_reported else "FALL!"
                continue

            if post == "lying":
                s.up_since = None
                if s.pending_lying is None:
                    s.pending_lying = t
                if s.lying_since is None and t - s.pending_lying >= c["lying_debounce_s"]:
                    s.lying_since = s.pending_lying
                    self._classify_drop(s, p)
                s.status = "LYING" if s.lying_since is not None else s.status
            else:
                s.pending_lying = None
                if not s.fall_reported:
                    s.up_since = None
                    s.lying_since = None
                    s.fall_t = None
                    s.status = "MEDICAL!" if s.down_reported else "OK"
                else:
                    # A sustained change to a seated or other non-lying pose
                    # ends the fall. Allow brief ambiguous pose samples.
                    s.up_since = s.up_since if s.up_since is not None else t
                    if t - s.up_since + 1e-6 >= c["recover_s"]:
                        s.fall_reported = False
                        s.fall_t = s.lying_since = None
                        s.status = "MEDICAL!" if s.down_reported else "OK"
                continue

            if s.lying_since is not None:
                lying_for = t - s.lying_since
                if not s.fall_reported and lying_for + 1e-6 >= c["confirm_lying_s"]:
                    s.fall_reported = True
                    s.status = "FALL!"
                    reason = (f"person fell and stayed down {lying_for:.1f}s"
                              if s.fall_t is not None else
                              "person found lying; reported as a fall")
                    events.append(Event("fall", min(0.99, 0.70 + 0.05 * lying_for), t,
                                        reason, tuple(map(int, p.box)), p.id))
                if s.down_reported:
                    s.status = "MEDICAL!"
                elif s.fall_reported:
                    s.status = "FALL!"
        return events

    def _cancel_medical(self, s):
        s.down_reported = False
        s.status = "FALL!" if s.fall_reported else "OK"
        self._reset_motion(s)

    def _update_medical(self, s, t, still_for, moved, events):
        c = self.c
        if s.down_reported and moved:
            self._cancel_medical(s)
            return
        s.still_progress = still_for
        if (c.get("report_person_down", True) and not s.down_reported
                and still_for is not None
                and still_for + 1e-6 >= c.get("medical_still_s", c["person_down_s"])):
            s.down_reported = True
            s.status = "MEDICAL!"
            events.append(Event("person_down", 0.90, t,
                                f"visible person showed no detected movement for {still_for:.1f}s; possible medical emergency",
                                tuple(map(int, s.last_box)), s.track_id))

    def _estimate_camera_offset(self, frame, people):
        """Measure camera translation against a stable low-resolution scene."""
        if not people and not self.states:
            self.camera_reference = None
            self.camera_offset.fill(0)
            return
        gray = cv2.cvtColor(cv2.resize(frame, (192, 144), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        current = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        if self.camera_reference is None:
            self.camera_reference = current
            self.camera_offset.fill(0)
            return
        shift, response = cv2.phaseCorrelate(self.camera_reference.copy(), current, self.camera_window)
        if response >= 0.45 and max(abs(shift[0]), abs(shift[1])) < 15:
            self.camera_offset = np.array((shift[0] * frame.shape[1] / 192,
                                           shift[1] * frame.shape[0] / 144))

    @staticmethod
    def _shift_box(box, shift):
        dx, dy = shift
        return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy

    def _reattach_lying_tracks(self, people, alive, t, frame):
        """Keep the original alert when a briefly lost lying person gets a new ID."""
        for p in people:
            if p.id in self.states or self.posture(p)[0] != "lying":
                continue
            matches = [(_iou(s.last_box, p.box), pid) for pid, s in self.states.items()
                       if pid not in alive and s.fall_reported
                       and (t - s.last_seen_t < self.c.get("missing_hold_s", 0.8)
                            or (t - s.last_seen_t < self.c.get("medical_visual_hold_s", 6.0)
                                and self._same_view(s, frame)))]
            if matches:
                overlap, pid = max(matches)
                if overlap >= 0.3:
                    self.states[p.id] = self.states.pop(pid)

    def _same_view(self, s, frame):
        if frame is None or s.last_view is None:
            return False
        shift = self.camera_offset - s.last_camera_offset
        view = self._box_view(frame, self._shift_box(s.last_box, shift))
        return (view is not None
                and np.mean(cv2.absdiff(view, s.last_view).max(axis=2) > 25) <= 0.22)

    @staticmethod
    def _box_view(frame, box, size=(24, 12)):
        if frame is None or box is None:
            return None
        x1, y1, x2, y2 = map(round, box)
        h, w = frame.shape[:2]
        view = frame[max(0, y1):max(0, min(h, y2)), max(0, x1):max(0, min(w, x2))]
        return cv2.resize(view, size, interpolation=cv2.INTER_AREA) if view.size else None

    @staticmethod
    def _reset_visual_motion(s):
        s.visual_box = s.visual_anchor = s.visual_previous = None
        s.visual_t = s.visual_since = None

    def _visual_stillness(self, s, frame, t, pose_visible):
        s.visual_moved = False
        if frame is None:
            return None
        stable_box = self._shift_box(s.last_box, -s.last_camera_offset)
        if pose_visible and (s.visual_box is None or _iou(stable_box, s.visual_box) < 0.45):
            s.visual_moved = s.visual_box is not None
            self._reset_visual_motion(s)
            s.visual_box = stable_box
        box = self._shift_box(s.visual_box, self.camera_offset) if s.visual_box is not None else None
        view = self._box_view(frame, box, (64, 32))
        if view is None:
            return None
        # Small resampling differences and monitor moire should not look like
        # motion. At this scale a 3x3 blur retains meaningful body movement.
        view = cv2.GaussianBlur(view, (3, 3), 0)
        if s.visual_t is None or t - s.visual_t > self.c.get("medical_max_gap_s", 0.75):
            # Start only on a confirmed person and a crop with visible detail.
            if not pose_visible or view.std(axis=(0, 1)).mean() < 7:
                self._reset_visual_motion(s)
                return None
            s.visual_anchor = s.visual_previous = view
            s.visual_since = s.visual_t = t
            return 0.0
        step = cv2.absdiff(view, s.visual_previous).max(axis=2)
        drift = cv2.absdiff(view, s.visual_anchor).max(axis=2)
        if np.mean(step > 18) > 0.025 or np.mean(drift > 18) > 0.045:
            s.visual_moved = True
            s.visual_since = t
            s.visual_anchor = view
        s.visual_t = t
        s.visual_previous = view
        return t - s.visual_since

    def _clear_missing_state_after_recovery(self, state, people, t):
        """Clear a latched alert if the same area is occupied by a standing person."""
        candidates = []
        if state.last_box is not None:
            a = np.asarray(state.last_box, dtype=float)
            ac = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
            ah = max(a[3] - a[1], 1.0)
            for p in people:
                if self.posture(p)[0] != "up":
                    continue
                b = np.asarray(p.box, dtype=float)
                bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                bh = max(b[3] - b[1], 1.0)
                scale = max(ah, bh)
                distance = float(np.linalg.norm(ac - bc)) / scale
                x1, y1 = max(a[0], b[0]), max(a[1], b[1])
                x2, y2 = min(a[2], b[2]), min(a[3], b[3])
                intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = ((a[2] - a[0]) * (a[3] - a[1])
                         + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
                iou = intersection / union if union > 0 else 0.0
                if iou >= 0.10 or distance < 0.40:
                    candidates.append((distance - iou, p.id))

        if not candidates:
            state.recovery_person = None
            state.recovery_since = None
            return False
        _, person_id = min(candidates)
        if state.recovery_person != person_id:
            state.recovery_person = person_id
            state.recovery_since = t
            return False
        if state.recovery_since is not None and t - state.recovery_since >= self.c["recover_s"]:
            state.fall_t = None
            state.fall_reported = False
            state.down_reported = False
            state.status = "OK"
            return True
        return False

    def _reset_motion(self, s):
        self._reset_visual_motion(s)
        s.still_progress = None
        s.motion_points = {}
        s.motion_anchor = {}
        s.motion_t = None
        s.still_since = None

    def _stationary_for(self, s, p, t):
        """Return continuous visible still time, or None if motion cannot be checked."""
        s.pose_moved = False
        conf_min = self.c.get("medical_min_kpt_conf", 0.30)
        indices = (*HEAD_POINTS, *ARM_POINTS, *LEG_POINTS)
        visible = {i: p.kpts[i] - self.camera_offset for i in indices if p.conf[i] >= conf_min}
        groups = (HEAD_POINTS, ARM_POINTS, LEG_POINTS)
        min_points = self.c.get("medical_min_visible_points", 3)
        min_groups = self.c.get("medical_min_visible_groups", 2)
        visible_groups = sum(any(i in visible for i in group) for group in groups)
        has_groups = len(visible) >= min_points and visible_groups >= min_groups
        if not has_groups:
            s.motion_points.clear()
            s.motion_anchor.clear()
            s.motion_t = s.still_since = None
            return None

        if s.motion_t is None or t - s.motion_t > self.c.get("medical_max_gap_s", 0.75):
            s.motion_points = visible
            s.motion_anchor = visible.copy()
            s.motion_t = t
            s.still_since = t
            return 0.0

        common = set(visible).intersection(s.motion_points)
        common_groups = sum(any(i in common for i in group) for group in groups)
        if len(common) < min_points or common_groups < min_groups:
            s.motion_points = visible
            s.motion_anchor = visible.copy()
            s.motion_t = t
            s.still_since = t
            return 0.0

        threshold = max(self.c.get("medical_motion_min_px", 4.0),
                        self.c.get("medical_motion_frac", 0.02) * max(p.h, p.w, 1.0))
        smoothed = {i: 0.5 * point + 0.5 * s.motion_points.get(i, point)
                    for i, point in visible.items()}
        frame_step = max(float(np.linalg.norm(smoothed[i] - s.motion_points[i])) for i in common)
        drift = max(float(np.linalg.norm(smoothed[i] - s.motion_anchor.get(i, smoothed[i])))
                    for i in common)
        if max(frame_step, drift) > threshold:
            s.pose_moved = True
            s.still_since = t
            s.motion_anchor = smoothed.copy()
        elif s.still_since is None:
            s.still_since = t

        # Smooth small pose-estimation jitter so a stationary person can accumulate time.
        s.motion_points = smoothed
        s.motion_t = t
        return max(0.0, t - s.still_since)

    def _classify_drop(self, s, p):
        """Called once when lying becomes established: was it a fast drop from upright?"""
        c = self.c
        if s.last_up_t is None:
            return
        transition = s.lying_since - s.last_up_t
        if transition > c["max_transition_s"]:
            return
        h_drop = 1.0 - (p.h / max(s.last_up_h, 1.0))
        cy_drop = (float(p.center[1]) - s.last_up_cy) / max(s.last_up_h, 1.0)
        if h_drop >= c["min_height_drop"] or cy_drop >= 0.25:
            s.fall_t = s.lying_since
