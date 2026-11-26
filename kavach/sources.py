"""Frame sources with proper timestamps: video file, webcam index, RTSP/HTTP stream, phone 'shot.jpg'."""
import threading
import time
import urllib.request
import cv2
import numpy as np

ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def prep(frame, rotate=0, max_width=960, flip_horizontal=False):
    if rotate in ROT:
        frame = cv2.rotate(frame, ROT[rotate])
    if flip_horizontal:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    return frame


class Source:
    def __init__(self, spec="0", rotate=90, max_width=640, realtime=False,
                 flip_horizontal=False):
        self.spec, self.rotate, self.max_width, self.realtime = str(spec), rotate, max_width, realtime
        self.flip_horizontal = flip_horizontal
        self.is_file = False
        self.snapshot = self.spec.lower().split("?")[0].endswith(("shot.jpg", "snapshot.jpg"))
        self.network_stream = self.spec.lower().startswith(("http://", "https://", "rtsp://"))
        self.cap = None
        if not self.snapshot:
            s = int(self.spec) if self.spec.isdigit() else self.spec
            self.cap = self._open_network_capture() if self.network_stream else cv2.VideoCapture(s)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                raise RuntimeError(f"Cannot open source: {self.spec}")
            self.is_file = not isinstance(s, int) and "://" not in self.spec
            if not self.is_file:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 1 and fps < 240 else 25.0
        self._latest, self._lock, self._stop = None, threading.Lock(), False
        if not self.is_file:                      # live: always process the NEWEST frame (no lag build-up)
            threading.Thread(target=self._reader, daemon=True).start()

    def _open_network_capture(self):
        cap = cv2.VideoCapture()
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000]
        try:
            cap.open(self.spec, cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            cap.release()
            cap = cv2.VideoCapture(self.spec)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap.release()
        return cap

    # live reader -------------------------------------------------------
    def _reader(self):
        retry_delay = 0.5
        last_warning = 0.0
        while not self._stop:
            try:
                if self.snapshot:
                    data = urllib.request.urlopen(self.spec, timeout=3).read()
                    fr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                else:
                    if self.network_stream and (self.cap is None or not self.cap.isOpened()):
                        cap = self._open_network_capture()
                        if self._stop:
                            cap.release()
                            break
                        self.cap = cap
                        if not self.cap.isOpened():
                            now = time.monotonic()
                            if now - last_warning >= 5.0:
                                print("[warn] phone camera is offline; retrying connection")
                                last_warning = now
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 1.5, 3.0)
                            continue
                    ok, fr = self.cap.read()
                    if not ok:
                        if self.network_stream:
                            self.cap.release()
                            self.cap = None
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 1.5, 3.0)
                        else:
                            time.sleep(0.2)
                        continue
                if fr is not None:
                    with self._lock:
                        self._latest = fr
                    retry_delay = 0.5
                    last_warning = 0.0
            except Exception as e:
                now = time.monotonic()
                if now - last_warning >= 5.0:
                    print("[warn] source read failed; reconnecting:", e)
                    last_warning = now
                if self.network_stream and self.cap is not None:
                    self.cap.release()
                    self.cap = None
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 3.0)

    def frames(self):
        t0 = time.monotonic()
        idx = 0
        if self.is_file:
            while True:
                ok, fr = self.cap.read()
                if not ok:
                    return
                t = idx / self.fps
                idx += 1
                if self.realtime:
                    delay = t - (time.monotonic() - t0)
                    if delay > 0:
                        time.sleep(delay)
                yield prep(fr, self.rotate, self.max_width, self.flip_horizontal), t
        else:
            last, wait_since = None, time.monotonic()
            while not self._stop:
                with self._lock:
                    fr = self._latest
                if fr is None or fr is last:
                    time.sleep(0.005)
                    if time.monotonic() - wait_since > 0.5:      # let the UI stay responsive
                        wait_since = time.monotonic()
                        yield None, time.monotonic() - t0
                    continue
                wait_since = time.monotonic()
                last = fr
                yield prep(fr.copy(), self.rotate, self.max_width, self.flip_horizontal), time.monotonic() - t0

    def close(self):
        self._stop = True
        if self.cap is not None:
            self.cap.release()
