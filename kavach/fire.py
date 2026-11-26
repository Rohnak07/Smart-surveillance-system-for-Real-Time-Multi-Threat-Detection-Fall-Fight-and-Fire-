"""Fire / smoke detection.

mode "yolo"      : a YOLO fire/smoke model (train with tools/train_fire.py on the D-Fire dataset).
mode "heuristic" : saturated flame colour, tapered silhouette, and a flickering tip above a
                   mostly stable base. Pose-detected people and camera movement are excluded.
                   No smoke.
mode "auto"      : YOLO if the weights file exists, otherwise heuristic.
Both modes need the cue to persist for `confirm_s` seconds before alerting.
"""
import os
from collections import deque
import cv2
import numpy as np
from .types import Event


class FireDetector:
    def __init__(self, cfg):
        self.c = cfg
        self.mode = cfg["mode"]
        self.yolo = None
        path = cfg["yolo_model"]
        if self.mode in ("auto", "yolo") and os.path.exists(path):
            from ultralytics import YOLO
            self.yolo = YOLO(path)
            self.mode = "yolo"
        elif self.mode == "yolo":
            raise FileNotFoundError(f"fire.mode=yolo but {path} not found (see tools/train_fire.py)")
        else:
            self.mode = "heuristic"
        self.normal_mode = self.mode
        self.screen_demo = bool(cfg.get("screen_demo"))
        if self.screen_demo:
            self.mode = "heuristic"
        self.hist = deque()          # (t, bool_cue)
        self.prev_mask = None
        self.prev_gray = None
        self.flame_tracks = []       # (recent shapes, recent motion cues) for each candidate
        self.last_boxes = []
        self.last_kind = "fire"

    def toggle_screen_demo(self):
        self.screen_demo = not self.screen_demo
        self.mode = "heuristic" if self.screen_demo else self.normal_mode
        self.hist.clear()
        self.flame_tracks.clear()
        self.prev_gray = self.prev_mask = None
        self.last_boxes = []

    # --------------------------------------------------------------- public
    def update(self, frame, t, exclude_boxes=None):
        cue, boxes, conf, kind = (self._yolo(frame) if self.mode == "yolo"
                                  else self._heuristic(frame, exclude_boxes, t))
        self.last_boxes, self.last_kind = boxes, kind
        if self.mode != "yolo":
            return [Event(kind, conf, t, "heuristic detector", boxes[0])] if cue else []
        self.hist.append((t, cue))
        win = self.c["yolo_confirm_s"]
        while self.hist and t - self.hist[0][0] > win + 1e-6:
            self.hist.popleft()
        hits = sum(1 for _, v in self.hist if v)
        n = len(self.hist)
        span = self.hist[-1][0] - self.hist[0][0] if n > 1 else 0.0
        ok = hits >= self.c["yolo_min_hits"] and span >= 0.5 * win
        if cue and ok:
            box = boxes[0] if boxes else None
            return [Event(kind, conf, t, f"{self.mode} detector", box)]
        return []

    # ----------------------------------------------------------------- yolo
    def _yolo(self, frame):
        r = self.yolo.predict(frame, conf=self.c["yolo_conf"], imgsz=640, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return False, [], 0.0, "fire"
        names = r.names
        candidates = []
        for i, cls in enumerate(r.boxes.cls.tolist()):
            label = str(names[int(cls)]).lower()
            if "fire" in label or "smoke" in label:
                candidates.append((i, "smoke" if "smoke" in label else "fire"))
        if not candidates:
            return False, [], 0.0, "fire"
        best, kind = max(candidates, key=lambda item: float(r.boxes.conf[item[0]]))
        box = tuple(map(int, r.boxes.xyxy[best].tolist()))
        return True, [box], float(r.boxes.conf[best]), kind

    # ------------------------------------------------------------ heuristic
    def _heuristic(self, frame, exclude_boxes=None, t=0.0):
        c = self.c
        h0, w0 = frame.shape[:2]
        sc = 320.0 / max(w0, 1)
        small = cv2.resize(frame, (320, max(int(h0 * sc), 1)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        b, g, r = [small[..., i].astype(np.int16) for i in range(3)]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev_gray, self.prev_gray = self.prev_gray, gray
        if prev_gray is None or prev_gray.shape != gray.shape:
            self.prev_mask = None
            self.flame_tracks.clear()
            return False, [], 0.0, "fire"

        scene_motion = float(cv2.absdiff(gray, prev_gray).mean())
        if scene_motion > c.get("camera_motion_max", 12.0):
            self.prev_mask = None
            self.flame_tracks.clear()
            return False, [], 0.0, "fire"

        min_sat = c.get("min_saturation", 150)
        min_hue = c.get("min_flame_hue", 4)
        max_hue = c.get("max_flame_hue", 35)
        red_delta = c.get("min_red_green_delta", 40)
        orange = ((cv2.inRange(hsv, (min_hue, min_sat, 170), (max_hue, 255, 255)) > 0)
                  & (r >= g + red_delta) & (g >= b - 10))
        # Phone exposure and screen glare can wash out the yellow inside a flame.
        # Retain warm cores and their orange fringe as a single component.
        warm_core = ((cv2.inRange(hsv, (10, 65, 160), (42, 255, 255)) > 0)
                     & (r >= 160) & (g >= b + 22)
                     & (g >= c.get("min_yellow_core_green", 115))
                     & (g >= c.get("min_yellow_core_green_red_ratio", 0.62) * r))
        # Overexposed white pixels beside amber pixels can belong to the same
        # flame. Reconnect them without turning a white object into a candidate.
        hot = (r >= 225) & (g >= 200) & (r >= b - 2)
        amber_nearby = cv2.dilate(warm_core.astype(np.uint8), np.ones((9, 9), np.uint8))
        color = orange | warm_core | (hot & (amber_nearby != 0))
        if exclude_boxes:
            block = np.zeros(color.shape, np.uint8)
            pad_frac = c.get("person_box_pad_frac", 0.05)
            for box in exclude_boxes:
                x1, y1, x2, y2 = map(float, box)
                pad = max(2, int((y2 - y1) * pad_frac * sc))
                x1, y1 = int(x1 * sc) - pad, int(y1 * sc) - pad
                x2, y2 = int(x2 * sc) + pad, int(y2 * sc) + pad
                cv2.rectangle(block, (x1, y1), (x2, y2), 255, -1)
            color[block > 0] = False
        mask = cv2.morphologyEx(color.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        prev, self.prev_mask = self.prev_mask, mask
        area_total = mask.shape[0] * mask.shape[1]
        if prev is None or prev.shape != mask.shape:
            return False, [], 0.0, "fire"
        if (mask > 0).sum() / area_total > c["max_area_frac"]:
            self.flame_tracks.clear()
            return False, [], 0.0, "fire"

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        flick = cv2.absdiff(mask, prev) > 0
        candidates = []
        min_aspect = c.get("min_flame_aspect", 0.28)
        max_aspect = c.get("max_flame_aspect", 1.30)
        max_taper = c.get("max_top_base_width_ratio", 0.95)
        min_core_frac = c.get("min_yellow_core_frac", 0.08)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < c["min_area_frac"] * area_total:
                continue
            x, y, w, h = stats[i, :4]
            # Clipping a rigid orange object can make its shape look like
            # flicker. Ignore thin slivers, while keeping larger edge flames.
            side_margin = 3
            if ((x < side_margin or x + w > mask.shape[1] - side_margin)
                    and w < max(12, 0.07 * mask.shape[1])):
                continue
            aspect = w / max(float(h), 1.0)
            if aspect < min_aspect or aspect > max_aspect:
                continue
            component = labels[y:y + h, x:x + w] == i
            row_widths = component.sum(axis=1).astype(float)
            split_top = max(1, int(0.4 * h))
            split_bottom = min(h - 1, int(0.6 * h))
            top_width = row_widths[:split_top].mean()
            bottom_width = row_widths[split_bottom:].mean()
            taper = top_width / max(bottom_width, 1.0)
            core = component & warm_core[y:y + h, x:x + w]
            core_y, core_x = np.where(core)
            core_frac = len(core_x) / float(area)
            if len(core_x) < c.get("min_yellow_core_pixels", 6) or core_frac < min_core_frac:
                continue
            # Broad or partially cropped flames can have several tips. Require
            # a stronger core and a distinct orange fringe for those shapes.
            if taper > max_taper and (taper > 1.45 or not 0.15 <= core_frac <= 0.70):
                continue
            if (abs(float(core_x.mean()) - (w - 1) / 2.0) > c.get("max_core_offset_frac", 0.30) * w
                    or not (0.15 * h <= float(core_y.mean()) <= 0.90 * h)):
                continue
            roi_flick = flick[max(y - 3, 0):y + h + 3, max(x - 3, 0):x + w + 3]
            churn = roi_flick.sum() / float(area)
            if churn >= c["min_churn"]:
                shape = (x + w / 2.0, y, y + h, h, w, area)
                candidates.append((churn, (x, y, w, h), shape))
        # Flame appearance and motion remain required inside or outside a display.
        # This allows a visible fire demonstration video to be recognized while
        # static bright screens, lamps, and ordinary warm objects still fail.
        candidates.sort(key=lambda item: item[0], reverse=True)
        motion, confirmed = self._track_flames(candidates[:6], t, mask.shape[0])
        ranking = confirmed or motion
        if not ranking:
            return False, [], 0.0, "fire"
        churn, (x, y, w, h) = max(ranking)
        box = (int(x / sc), int(y / sc), int((x + w) / sc), int((y + h) / sc))
        return bool(confirmed), [box], float(min(0.9, 0.5 + churn)), "fire"

    def _track_flames(self, candidates, t, frame_height):
        gap = self.c.get("shape_gap_s", 0.50)
        pending = [track for track in self.flame_tracks if t - track[0][-1][0] <= gap]
        self.flame_tracks = []
        motion, confirmed = [], []
        for churn, box, shape in candidates:
            matching = [i for i, (shapes, _) in enumerate(pending)
                        if self._same_flame(shapes[-1][1:], shape)]
            if matching:
                nearest = min(matching, key=lambda i: abs(pending[i][0][-1][1] - shape[0])
                              + abs(pending[i][0][-1][3] - shape[2]))
                shapes, cues = pending.pop(nearest)
            else:
                shapes, cues = deque(), deque()
            self.flame_tracks.append((shapes, cues))
            has_motion = self._flame_motion(shapes, shape, t, frame_height)
            if has_motion:
                motion.append((churn, box))
            if self._confirm_flame(cues, t, has_motion):
                confirmed.append((churn, box))
        for shapes, cues in pending[:6 - len(self.flame_tracks)]:
            self._confirm_flame(cues, t, False)
            self.flame_tracks.append((shapes, cues))
        return motion, confirmed

    def _confirm_flame(self, cues, t, has_motion):
        cues.append((t, has_motion))
        window = self.c["confirm_s"]
        while cues and t - cues[0][0] > window + 1e-6:
            cues.popleft()
        return (has_motion and len(cues) >= 3 and t - cues[0][0] >= 0.6 * window
                and sum(hit for _, hit in cues) / len(cues) >= self.c["confirm_ratio"])

    @staticmethod
    def _same_flame(previous, shape):
        px, _, pb, ph, pw, _ = previous
        x, _, b, h, w, _ = shape
        return (abs(x - px) <= max(5, 0.5 * max(pw, w))
                and abs(b - pb) <= max(5, 0.4 * max(ph, h)))

    def _flame_motion(self, history, shape, t, frame_height):
        """Require the flame tip/height to flicker while its base stays roughly put."""
        c = self.c
        history.append((t, *shape))
        window = c.get("motion_window_s", 0.8)
        while history and t - history[0][0] > window + 1e-6:
            history.popleft()

        if (len(history) < c.get("min_motion_samples", 4)
                or history[-1][0] - history[0][0] < c.get("min_motion_span_s", 0.2)):
            return False
        tops = [obs[2] for obs in history]
        bases = [obs[3] for obs in history]
        heights = [obs[4] for obs in history]
        areas = [obs[6] for obs in history]
        mean_height = max(float(np.mean(heights)), 1.0)
        height_variation = (max(heights) - min(heights)) / mean_height
        tip_motion = max(tops) - min(tops)
        base_motion = max(bases) - min(bases)
        top_steps = np.diff(tops)
        motion_scale = min(frame_height, 2 * mean_height)
        min_step = max(1.0, c.get("min_vertical_step_frac", 0.006) * motion_scale)
        reverses = bool(np.any(top_steps <= -min_step) and np.any(top_steps >= min_step))
        # Perspective and refresh artifacts make a filmed display less stable.
        tip_frac = 0.025 if self.screen_demo else c.get("min_tip_motion_frac", 0.035)
        base_frac = 0.23 if self.screen_demo else c.get("max_base_motion_frac", 0.15)
        # Adjacent flames can join and separate. Their base may jump while their
        # tips, height and area change much more than for a moving rigid object.
        connected_flames = (
            height_variation >= 0.30
            and (max(areas) - min(areas)) >= 0.55 * float(np.mean(areas))
            and base_motion <= 0.60 * mean_height
        )
        return (
            tip_motion >= max(3.0, tip_frac * motion_scale)
            and height_variation >= c.get("min_shape_variation", 0.12)
            and reverses
            and (base_motion <= base_frac * mean_height or connected_flames)
        )
