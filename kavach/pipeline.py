import io
import os
import sys
import subprocess
import tempfile
import time
import threading
import wave
from pathlib import Path
import cv2
import numpy as np
from .tracker import Tracker
from .fall import FallDetector, has_body_pose
from .fight import FightDetector
from .fire import FireDetector
from .events import EventSink
from .pose import PoseEstimator

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]
GREEN, ORANGE, RED, YELLOW = (0, 200, 0), (0, 140, 255), (0, 0, 255), (0, 220, 255)


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pose = PoseEstimator(cfg["pose"]) if (cfg["fall"]["enabled"] or cfg["fight"]["enabled"]) else None
        self.tracker = Tracker(cfg["tracker"]["max_age_s"])
        self.fall = FallDetector(cfg["fall"]) if cfg["fall"]["enabled"] else None
        fcfg = dict(cfg["fight"])
        fcfg["cnn_threshold"] = cfg["violence_cnn"]["threshold"]
        self.fight = FightDetector(fcfg) if cfg["fight"]["enabled"] else None
        self.fire = FireDetector(cfg["fire"]) if cfg["fire"]["enabled"] else None
        self.cnn = None
        if cfg["violence_cnn"]["enabled"]:
            from .violence_cnn import ViolenceCNN
            self.cnn = ViolenceCNN(cfg["violence_cnn"])
        self.sink = EventSink(cfg["alerts"], cfg["camera"])
        self.banner = []                 # (until_t, text, color)
        self._active_alerts = {}         # (event kind, tracked subject) -> (event, last seen)
        self._fps, self._tprev = 0.0, time.perf_counter()
        self.people = []
        self._pose_overlay = {}
        self._siren_active = False
        self._siren_stop = None
        self._siren_thread = None
        self._siren_test_thread = None
        self._fire_sound_active = False
        self._fire_sound_failed = False
        self._fire_sound_process = None
        self._fire_sound_last_seen_t = None
        self._fire_sound_control = Path(tempfile.gettempdir()) / (
            f"kavach_fire_audio_{os.getpid()}_{id(self):x}.cmd")
        if self.fire is not None and cfg.get("alerts", {}).get("fire_sound_enabled", True):
            self._ensure_fire_audio_player()

    def process(self, frame, t):
        """Run all detectors on one frame. Returns (accepted_events, annotated_frame)."""
        raw_events = []
        tracked = []
        if self.pose is not None:
            tracked = self.tracker.update(self.pose.detect(frame), t)
        self.people = [p for p in tracked if has_body_pose(p.kpts, p.conf)]
        if self.fall is not None:
            raw_events += self.fall.update(self.people, t, frame=frame)
        self._update_pose_overlay(self.people, t)
        if self.fight is not None:
            prob = self.cnn.update(frame, t) if self.cnn else None
            raw_events += self.fight.update(tracked, t, prob)
        fire_events = []
        if self.fire is not None:
            person_boxes = [p.box for p in self.people] if self.fire.mode == "heuristic" else None
            fire_events = self.fire.update(frame, t, person_boxes)
            raw_events += fire_events
        self._sync_fire_sound(any(ev.kind == "fire" for ev in fire_events), t)
        self._sync_siren()
        new_alerts = self._update_active_alerts(raw_events, t)

        vis = self._draw(frame)
        accepted = []
        for ev in raw_events:
            color = ORANGE if ev.kind in ("person_down",) else RED
            if ev.box is not None:
                cv2.rectangle(vis, ev.box[:2], ev.box[2:], color, 3)
        for ev in new_alerts:
            vis = self._draw_banner(vis, t)      # evidence should show the alert text
            if self.sink.emit(ev, vis):
                accepted.append(ev)
        vis = self._hud(self._draw_banner(vis, t), t)
        return accepted, vis

    def _update_active_alerts(self, events, t):
        """Keep event banners active through short detector gaps and log each onset once."""
        active = getattr(self, "_active_alerts", None)
        if active is None:
            active = self._active_alerts = {}
        for key, (event, last_seen) in list(active.items()):
            if t - last_seen > self._alert_clear_s(event.kind):
                del active[key]
        new_events = []
        for event in events:
            key = (event.kind, event.subject_id)
            if key not in active:
                new_events.append(event)
            active[key] = (event, t)
        return new_events

    def _alert_clear_s(self, kind):
        cfg = getattr(self, "cfg", {})
        if kind in ("fire", "smoke"):
            return float(cfg.get("fire", {}).get("alert_clear_s", 1.25))
        if kind == "fight":
            return float(cfg.get("fight", {}).get("alert_clear_s",
                                                   cfg.get("fight", {}).get("window_s", 4.0)))
        return 0.8

    def on_source_gap(self, t):
        self._fire_sound_last_seen_t = None
        self._stop_fire_sound()
        self._fire_sound_failed = False
        if self.fall is not None:
            self.fall.update([], t)
            self._sync_siren()

    # ------------------------------------------------------------------ drawing
    def _update_pose_overlay(self, people, t):
        """Compute display-only posture and motion details for each visible person."""
        tracks = getattr(self, "_pose_overlay", None)
        if tracks is None:
            tracks = self._pose_overlay = {}
        current_ids = {p.id for p in people}
        tracker_cfg = getattr(self, "cfg", {}).get("tracker", {})
        max_age = float(tracker_cfg.get("max_age_s", 2.0))
        for pid, previous in list(tracks.items()):
            if pid not in current_ids and t - previous["seen_t"] > max_age:
                del tracks[pid]

        output = {}
        for p in people:
            post, angle = self.fall.posture(p) if self.fall is not None else ("up", None)
            aspect = p.w / max(p.h, 1.0)
            entry = tracks.setdefault(p.id, {"falling_until": -1.0, "v": 0.0, "a": 0.0})
            v, acceleration = 0.0, 0.0
            dt = t - entry.get("seen_t", t)
            if 1e-3 <= dt <= 0.75:
                previous_h = max(entry.get("h", p.h), 1.0)
                v = (float(p.center[1]) - entry.get("cy", float(p.center[1]))) / previous_h / dt
                acceleration = (v - entry.get("v", 0.0)) / dt
                previous_angle = entry.get("angle")
                angle_change = abs(angle - previous_angle) if angle is not None and previous_angle is not None else 0.0
                height_drop = (previous_h - p.h) / previous_h
                if (v >= 0.65 and (angle_change >= 7.0 or height_drop >= 0.10 or post == "mid")):
                    entry["falling_until"] = t + 0.8

            fall_state = self.fall.states.get(p.id) if self.fall is not None else None
            if post == "lying" or (fall_state is not None and fall_state.fall_reported):
                status = "fallen"
            elif t <= entry.get("falling_until", -1.0):
                status = "falling"
            else:
                status = "normal"

            visible_conf = p.conf[np.isfinite(p.conf) & (p.conf >= 0.30)]
            pose_conf = float(visible_conf.mean()) if visible_conf.size else 0.0
            head_y = None
            if p.conf[0] >= 0.30 and p.h > 0:
                head_y = float(np.clip((p.kpts[0, 1] - p.box[1]) / p.h, 0.0, 1.0))
            entry.update(seen_t=t, cy=float(p.center[1]), h=float(p.h),
                         angle=angle, v=v)
            output[p.id] = {
                "status": status, "angle": angle, "aspect": aspect,
                "v": v, "a": acceleration, "head_y": head_y, "confidence": pose_conf,
                "medical": bool(fall_state is not None and fall_state.down_reported),
            }
        self._pose_overlay = tracks
        self._pose_overlay_current = output

    def _draw(self, frame):
        vis = frame.copy()
        overlay = getattr(self, "_pose_overlay_current", {})
        for p in self.people:
            details = overlay.get(p.id, {})
            status = details.get("status", "normal")
            col = {"normal": GREEN, "falling": YELLOW, "fallen": RED}.get(status, GREEN)
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            label = f"ID:{p.id} {status}"
            cv2.putText(vis, label, (x1, max(y1 - 7, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
            angle = details.get("angle")
            angle_text = "--" if angle is None else f"{angle:.1f}"
            aspect_text = f"{details.get('aspect', 0.0):.2f}"
            head_y = details.get("head_y")
            head_text = "--" if head_y is None else f"{head_y:.2f}"
            line1 = f"angle:{angle_text} aspect:{aspect_text} v:{details.get('v', 0.0):.2f}"
            line2 = f"a:{details.get('a', 0.0):.2f} head:{head_text} pose:{details.get('confidence', 0.0):.2f}"
            text_y = min(y1 + 17, vis.shape[0] - 24)
            metrics_width = min(x2 - x1 - 4, 250)
            if metrics_width > 0:
                cv2.rectangle(vis, (x1 + 2, text_y - 13),
                              (x1 + 2 + metrics_width, min(text_y + 19, vis.shape[0] - 1)),
                              (0, 0, 0), -1)
            cv2.putText(vis, line1, (x1 + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            cv2.putText(vis, line2, (x1 + 4, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            if details.get("medical"):
                cv2.putText(vis, "POSSIBLE MEDICAL EMERGENCY",
                            (x1, min(y2 + 18, vis.shape[0] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2)
            for a, b in SKELETON:
                if p.conf[a] > 0.3 and p.conf[b] > 0.3:
                    pa, pb = tuple(map(int, p.kpts[a])), tuple(map(int, p.kpts[b]))
                    cv2.line(vis, pa, pb, (255, 255, 255), 2)
                    cv2.circle(vis, pa, 3, RED, -1)
                    cv2.circle(vis, pb, 3, RED, -1)
        if self.fire is not None:
            for b in self.fire.last_boxes:
                cv2.rectangle(vis, b[:2], b[2:], ORANGE, 2)
                label = self.fire.last_kind.replace("_", " ").upper()
                cv2.putText(vis, label + "?", (b[0], max(b[1] - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2)
        if self.fall is not None:
            visible_ids = {p.id for p in self.people}
            for pid, state in self.fall.states.items():
                if (pid in visible_ids or not (state.fall_reported or state.down_reported)
                        or state.last_box is None):
                    continue
                box = tuple(map(int, state.last_box))
                label = "POSSIBLE MEDICAL EMERGENCY" if state.down_reported else "FALL DETECTED"
                cv2.rectangle(vis, box[:2], box[2:], RED, 2)
                cv2.putText(vis, label, (box[0], max(box[1] - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
        return vis

    def _draw_banner(self, vis, t):
        self.banner = [b for b in self.banner if b[0] > t]
        active = self._active_fall_banners()
        cam = getattr(self, "cfg", {}).get("camera", {})
        camera = str(cam.get("id", "CAMERA"))
        for event, _ in getattr(self, "_active_alerts", {}).values():
            if event.kind in ("fall", "person_down"):
                continue
            color = ORANGE if event.kind == "person_down" else RED
            active.append((None, f"{camera} | {event.label}", color))
        rows = active[:3]
        if len(rows) < 3:
            rows.extend(self.banner[-(3 - len(rows)):])
        for i, (_, text, col) in enumerate(rows):
            y = 56 + 34 * i
            cv2.rectangle(vis, (8, y - 24), (14 + 14 * len(text) + 90, y + 8), (0, 0, 0), -1)
            cv2.putText(vis, f"ALERT: {text}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
        return vis

    def _active_fall_banners(self):
        """Keep active per-person fall and inactivity alerts visible."""
        rows = []
        if self.fall is None:
            return rows
        cam = getattr(self, "cfg", {}).get("camera", {})
        camera = str(cam.get("id", "CAMERA"))
        for state in self.fall.states.values():
            if state.fall_reported:
                rows.append((None, f"{camera} | FALL DETECTED", RED))
            if state.down_reported:
                rows.append((None, f"{camera} | POSSIBLE MEDICAL EMERGENCY", ORANGE))
        return rows

    def _sync_siren(self):
        """Start a background laptop siren only during a medical emergency."""
        cfg = getattr(self, "cfg", {})
        enabled = cfg.get("alerts", {}).get("siren_enabled", True)
        active = (enabled and self.fall is not None
                  and any(s.down_reported for s in self.fall.states.values()))
        siren_active = getattr(self, "_siren_active", False)
        if active and not siren_active:
            self._start_siren()
        elif not active and siren_active:
            self._stop_siren()

    def _fire_sound_config(self):
        alerts = getattr(self, "cfg", {}).get("alerts", {})
        enabled = bool(alerts.get("fire_sound_enabled", True))
        configured = Path(alerts.get("fire_sound_path", "assets/emergency_alert.mp3"))
        path = configured if configured.is_absolute() else Path(__file__).resolve().parents[1] / configured
        return enabled, path

    def _sync_fire_sound(self, detected, t=None):
        """Play the configured repeating alarm only while fire is currently detected."""
        enabled, _ = self._fire_sound_config()
        detected = bool(detected and enabled)
        process = getattr(self, "_fire_sound_process", None)
        if detected and getattr(self, "_fire_sound_active", False) and process is not None:
            if process.poll() is not None:
                detail = process.stderr.read().strip() if process.stderr is not None else ""
                self._fire_sound_process = None
                self._fire_sound_active = False
                self._fire_sound_failed = True
                print(f"[warn] fire alert audio player exited unexpectedly: {detail or process.returncode}")
        if detected:
            self._fire_sound_last_seen_t = t if t is not None else time.monotonic()
            if (not getattr(self, "_fire_sound_active", False)
                    and not getattr(self, "_fire_sound_failed", False)):
                self._start_fire_sound()
            return

        if (getattr(self, "_fire_sound_active", False)
                and t is not None and self._fire_sound_last_seen_t is not None):
            clear_after = float(self.cfg.get("fire", {}).get("alert_clear_s", 1.25))
            if t - self._fire_sound_last_seen_t <= clear_after:
                return
        if getattr(self, "_fire_sound_active", False):
            self._stop_fire_sound()
        self._fire_sound_last_seen_t = None
        self._fire_sound_failed = False

    def _start_fire_sound(self):
        if not self._ensure_fire_audio_player():
            return
        if not self._send_fire_sound_command("PLAY"):
            self._fire_sound_failed = True
            print("[warn] could not signal the fire alert audio player")
            return
        self._fire_sound_active = True

    def _ensure_fire_audio_player(self):
        enabled, path = self._fire_sound_config()
        if not enabled:
            return False
        process = getattr(self, "_fire_sound_process", None)
        if process is not None and process.poll() is None:
            return True
        if sys.platform != "win32":
            self._fire_sound_failed = True
            print("[warn] fire alert audio playback requires Windows desktop media support")
            return False
        if not path.is_file():
            self._fire_sound_failed = True
            print(f"[warn] fire alert audio file not found: {path}")
            return False
        helper = Path(__file__).resolve().parents[1] / "assets" / "fire_audio_player.ps1"
        if not helper.is_file():
            self._fire_sound_failed = True
            print(f"[warn] fire alert audio helper not found: {helper}")
            return False
        try:
            self._fire_sound_control.unlink(missing_ok=True)
            process = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-STA", "-WindowStyle", "Hidden",
                 "-ExecutionPolicy", "Bypass", "-File", str(helper),
                 path.resolve().as_uri(), str(self._fire_sound_control)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self._fire_sound_process = process
            self._fire_sound_failed = False
            return True
        except Exception as exc:
            process = getattr(self, "_fire_sound_process", None)
            if process is not None:
                try:
                    process.terminate()
                except Exception:
                    pass
            self._fire_sound_process = None
            self._fire_sound_active = False
            self._fire_sound_failed = True
            print(f"[warn] fire alert audio unavailable: {exc}")
            return False

    def _send_fire_sound_command(self, command):
        process = getattr(self, "_fire_sound_process", None)
        if process is None or process.poll() is not None:
            return False
        control = getattr(self, "_fire_sound_control", None)
        if control is None:
            return False
        pending = control.with_suffix(".tmp")
        try:
            pending.write_text(command, encoding="ascii")
            os.replace(pending, control)
            return True
        except Exception as exc:
            print(f"[warn] fire alert audio command failed: {exc}")
            return False

    def _stop_fire_sound(self):
        process = getattr(self, "_fire_sound_process", None)
        if process is not None and process.poll() is None and self._fire_sound_active:
            self._send_fire_sound_command("STOP")
        self._fire_sound_active = False

    def _close_fire_audio_player(self):
        process = getattr(self, "_fire_sound_process", None)
        if process is not None and process.poll() is None:
            self._send_fire_sound_command("EXIT")
            try:
                process.wait(timeout=0.4)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
        if process is not None and process.stderr is not None:
            process.stderr.close()
        self._fire_sound_process = None
        self._fire_sound_active = False
        control = getattr(self, "_fire_sound_control", None)
        if control is not None:
            control.unlink(missing_ok=True)
            control.with_suffix(".tmp").unlink(missing_ok=True)

    def _start_siren(self):
        import sys
        if sys.platform != "win32":
            return
        self._siren_stop = threading.Event()
        self._siren_thread = threading.Thread(
            target=self._siren_loop, args=(self._siren_stop,), daemon=True)
        self._siren_thread.start()
        self._siren_active = True

    def test_siren(self):
        if self._siren_active or (self._siren_test_thread is not None
                                  and self._siren_test_thread.is_alive()):
            return
        self._siren_test_thread = threading.Thread(
            target=self._siren_loop, args=(threading.Event(), True), daemon=True)
        self._siren_test_thread.start()

    @staticmethod
    def _siren_loop(stop_event, once=False):
        try:
            import winsound
            # Play a short WAV through the default audio output. The Windows
            # Beep API can fail even when normal speaker playback works.
            rate = 16000
            t = np.arange(rate // 2) / rate
            frequency = 950 + 330 * np.sin(4 * np.pi * t)
            phase = 2 * np.pi * np.cumsum(frequency) / rate
            envelope = np.minimum(1, np.minimum(t, t[-1] - t) / 0.02)
            samples = (18000 * envelope * np.sin(phase)).astype("<i2")
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as audio:
                audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                audio.writeframes(samples.tobytes())
            sound = buffer.getvalue()
            while not stop_event.is_set():
                winsound.PlaySound(sound, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
                if once:
                    break
        except Exception as exc:
            print(f"[warn] medical siren unavailable: {exc}")

    def _stop_siren(self):
        if self._siren_stop is not None:
            self._siren_stop.set()
        try:
            import sys
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception as exc:
            print(f"[warn] could not stop the medical siren immediately: {exc}")
        self._siren_active = False

    def close(self):
        self._fire_sound_last_seen_t = None
        self._close_fire_audio_player()
        self._stop_siren()
        if self._siren_thread is not None:
            self._siren_thread.join(timeout=0.6)
            self._siren_thread = None

    def _hud(self, vis, t):
        now = time.perf_counter()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - self._tprev, 1e-6))
        self._tprev = now
        cam = self.cfg["camera"]
        score = f" fight={self.fight.last_score:.1f}" if self.fight else ""
        txt = f"{cam['id']} | t={t:6.1f}s | {self._fps:4.1f} FPS | people={len(self.people)}{score}"
        cv2.putText(vis, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(vis, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return vis
