"""Event output: cooldown, CSV log, evidence image, optional Telegram (never blocks the video loop)."""
import csv
import os
import threading
from datetime import datetime
import cv2


class EventSink:
    def __init__(self, cfg, camera):
        self.c, self.cam = cfg, camera
        self.last = {}
        self.rows = 0
        os.makedirs(os.path.dirname(cfg["csv_path"]) or ".", exist_ok=True)
        os.makedirs(cfg["evidence_dir"], exist_ok=True)
        if not os.path.exists(cfg["csv_path"]):
            with open(cfg["csv_path"], "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp", "source_time_s", "camera", "location",
                                        "event", "confidence", "info", "evidence"])
        self.tg = None
        tg = cfg.get("telegram", {})
        if tg.get("enabled"):
            tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
            if tok and chat:
                self.tg = (tok, chat)
            else:
                print("[warn] telegram.enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing -> disabled")

    def emit(self, ev, frame):
        """Returns True if the event was accepted (not inside cooldown)."""
        key = (ev.kind, ev.subject_id)
        if ev.t - self.last.get(key, -1e9) < self.c["cooldown_s"]:
            return False
        self.last[key] = ev.t
        now = datetime.now()
        path = ""
        if self.c["save_evidence"] and frame is not None:
            path = os.path.join(self.c["evidence_dir"], f"{ev.kind}_{now:%Y%m%d_%H%M%S}_{int(ev.t*1000)}.jpg")
            cv2.imwrite(path, frame)
        with open(self.c["csv_path"], "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([now.isoformat(timespec="seconds"), f"{ev.t:.2f}", self.cam["id"],
                                    self.cam["location"], ev.label, f"{ev.confidence:.2f}", ev.info, path])
        self.rows += 1
        camera = f"{self.cam['id']} ({self.cam['location']})"
        print(f"[ALERT] {now:%H:%M:%S}  Camera: {camera}  {ev.label}  "
              f"score={ev.confidence:.2f}  {ev.info}")
        if self.tg:
            threading.Thread(target=self._telegram, args=(ev, path), daemon=True).start()
        return True

    def _telegram(self, ev, path):
        try:
            import requests
            tok, chat = self.tg
            cap = (f"ALERT: {ev.label}\nCamera: {self.cam['id']} ({self.cam['location']})\n"
                   f"Confidence: {ev.confidence:.0%}\n{ev.info}")
            if path:
                with open(path, "rb") as f:
                    requests.post(f"https://api.telegram.org/bot{tok}/sendPhoto",
                                  data={"chat_id": chat, "caption": cap}, files={"photo": f}, timeout=15)
            else:
                requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              data={"chat_id": chat, "text": cap}, timeout=15)
        except Exception as e:        # alerts must never crash detection
            print("[warn] telegram failed:", e)
