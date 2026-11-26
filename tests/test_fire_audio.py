from kavach.pipeline import Engine
import kavach.pipeline as pipeline


class FakeInput:
    def __init__(self):
        self.text = ""
        self.closed = False

    def write(self, value):
        self.text += value

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def read(self):
        return ""


class FakeProcess:
    def __init__(self):
        self.stderr = FakeInput()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = -1


def test_fire_sound_starts_once_and_stops_when_detection_clears(monkeypatch):
    engine = Engine.__new__(Engine)
    engine.cfg = {"alerts": {"fire_sound_enabled": True}}
    engine._fire_sound_active = False
    engine._fire_sound_failed = False
    engine._fire_sound_process = None
    starts, stops = [], []

    def start():
        starts.append(True)
        engine._fire_sound_active = True

    def stop():
        stops.append(True)
        engine._fire_sound_active = False

    monkeypatch.setattr(engine, "_start_fire_sound", start)
    monkeypatch.setattr(engine, "_stop_fire_sound", stop)

    engine._sync_fire_sound(True)
    engine._sync_fire_sound(True)
    engine._sync_fire_sound(False)

    assert len(starts) == 1
    assert len(stops) == 1
    assert engine._fire_sound_active is False


def test_source_gap_stops_fire_sound_immediately(tmp_path):
    engine = Engine.__new__(Engine)
    engine.cfg = {"alerts": {"fire_sound_enabled": True}}
    engine.fall = None
    engine._fire_sound_active = True
    engine._fire_sound_failed = False
    engine._fire_sound_control = tmp_path / "fire.cmd"
    process = FakeProcess()
    engine._fire_sound_process = process

    engine.on_source_gap(1.0)

    assert engine._fire_sound_control.read_text(encoding="ascii") == "STOP"
    assert process.returncode is None
    assert engine._fire_sound_process is process
    assert engine._fire_sound_active is False


def test_brief_fire_detection_gap_does_not_restart_or_stop_player(tmp_path):
    engine = Engine.__new__(Engine)
    engine.cfg = {"alerts": {"fire_sound_enabled": True},
                  "fire": {"alert_clear_s": 1.25}}
    engine._fire_sound_active = True
    engine._fire_sound_failed = False
    engine._fire_sound_last_seen_t = 4.0
    engine._fire_sound_control = tmp_path / "fire.cmd"
    process = FakeProcess()
    engine._fire_sound_process = process

    engine._sync_fire_sound(False, 4.8)

    assert engine._fire_sound_active is True
    assert engine._fire_sound_process is process

    engine._sync_fire_sound(False, 5.3)

    assert engine._fire_sound_active is False
    assert engine._fire_sound_control.read_text(encoding="ascii") == "STOP"
    assert process.returncode is None


def test_disabled_fire_sound_does_not_start(monkeypatch):
    engine = Engine.__new__(Engine)
    engine.cfg = {"alerts": {"fire_sound_enabled": False}}
    engine._fire_sound_active = False
    engine._fire_sound_failed = False
    starts = []
    monkeypatch.setattr(engine, "_start_fire_sound", lambda: starts.append(True))

    engine._sync_fire_sound(True)

    assert starts == []


def test_player_receives_a_file_url_and_command_file(monkeypatch, tmp_path):
    engine = Engine.__new__(Engine)
    engine.cfg = {"alerts": {"fire_sound_enabled": True}}
    engine._fire_sound_active = False
    engine._fire_sound_failed = False
    engine._fire_sound_process = None
    engine._fire_sound_control = tmp_path / "fire.cmd"
    process = FakeProcess()
    launched = {}
    monkeypatch.setattr(pipeline.sys, "platform", "win32")
    monkeypatch.setattr(
        pipeline.subprocess, "Popen",
        lambda *args, **kwargs: (launched.update(args=args, kwargs=kwargs) or process),
    )

    engine._start_fire_sound()

    media_uri = launched["args"][0][-2]
    assert media_uri.startswith("file:///")
    assert "%20" in media_uri
    assert launched["kwargs"]["stdin"] == pipeline.subprocess.DEVNULL
    assert launched["args"][0][-1] == str(engine._fire_sound_control)
    assert engine._fire_sound_control.read_text(encoding="ascii") == "PLAY"
    assert engine._fire_sound_active is True
