from types import SimpleNamespace

from run_phone import discover_phone, is_ip_webcam, main


class FakeResponse:
    status = 200

    def __init__(self, content_type, body):
        self.headers = SimpleNamespace(get_content_type=lambda: content_type)
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self, count):
        return self.body[:count]


def test_probe_accepts_jpeg_snapshot(monkeypatch):
    monkeypatch.setattr("run_phone.urllib.request.urlopen",
                        lambda *_args, **_kwargs: FakeResponse("application/octet-stream", b"\xff\xd8\xffjpeg"))
    assert is_ip_webcam("127.0.0.1", 8080, timeout=1)


def test_probe_rejects_unrelated_web_page(monkeypatch):
    monkeypatch.setattr("run_phone.urllib.request.urlopen",
                        lambda *_args, **_kwargs: FakeResponse("text/html", b"not a camera"))
    assert not is_ip_webcam("127.0.0.1", 8080, timeout=1)


def test_phone_discovery_retries_until_ip_webcam_is_ready(monkeypatch):
    probes = iter([False, True])
    waits = []
    monkeypatch.setattr("run_phone.candidates", lambda: ("192.168.43.2", ["192.168.43.1"]))
    monkeypatch.setattr("run_phone.reachable", lambda ip: ip)
    monkeypatch.setattr("run_phone.is_ip_webcam", lambda *_args, **_kwargs: next(probes))
    monkeypatch.setattr("run_phone.time.sleep", lambda delay: waits.append(delay))

    assert discover_phone(port=8080, retry_s=3.0) == "192.168.43.1"
    assert waits == [3.0]


def test_phone_launch_defaults_to_portrait_rotation(monkeypatch):
    launched = {}

    def fake_call(args, cwd):
        launched["args"] = args
        launched["cwd"] = cwd
        return 0

    monkeypatch.setattr("run_phone.subprocess.call", fake_call)
    assert main(["--url", "http://192.168.43.1:8080/video", "--max-seconds", "1"]) == 0
    assert launched["args"][launched["args"].index("--rotate") + 1] == "90"
    assert "--flip-horizontal" in launched["args"]


def test_phone_launch_keeps_explicit_rotation(monkeypatch):
    launched = {}

    def fake_call(args, cwd):
        launched["args"] = args
        return 0

    monkeypatch.setattr("run_phone.subprocess.call", fake_call)
    assert main(["--url", "http://192.168.43.1:8080/video", "--rotate", "0"]) == 0
    assert launched["args"][launched["args"].index("--rotate") + 1] == "0"
