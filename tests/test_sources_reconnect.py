import threading

import numpy as np

from kavach.sources import Source


class FakeCapture:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return next(self.responses)

    def release(self):
        self.released = True


def test_network_reader_reopens_stream_after_read_failure(monkeypatch):
    source = Source.__new__(Source)
    source.spec = "http://127.0.0.1:8080/video"
    source.rotate = 0
    source.max_width = 640
    source.realtime = False
    source.flip_horizontal = False
    source.is_file = False
    source.snapshot = False
    source.network_stream = True
    source.cap = FakeCapture([(False, None)])
    source._latest = None
    source._lock = threading.Lock()
    source._stop = False
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    reopened = FakeCapture([(True, frame)])
    attempts = []

    def reopen():
        attempts.append(True)
        return reopened

    monkeypatch.setattr(source, "_open_network_capture", reopen)
    monkeypatch.setattr("kavach.sources.time.sleep", lambda _delay: None)

    original_read = reopened.read

    def finish_after_frame():
        result = original_read()
        source._stop = True
        return result

    reopened.read = finish_after_frame
    source._reader()

    assert attempts == [True]
    assert source._latest is frame
