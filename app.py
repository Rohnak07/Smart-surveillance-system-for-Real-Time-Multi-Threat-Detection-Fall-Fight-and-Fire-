"""Smart surveillance system for Real-Time Multi-Threat Detection (Fall, Fight, and Fire): run the CCTV multi-threat detector.

Examples
  python app.py --source videos/test.mp4
  python app.py --source 0                                   # laptop webcam
  python app.py --source http://192.168.1.5:8080/video       # phone (IP Webcam app, MJPEG)
  python app.py --source http://192.168.1.5:8080/shot.jpg --rotate 90
  python app.py --source videos/test.mp4 --no-display --save out.mp4   # headless
Keys: D = toggle screen-fire test settings, T = test siren, Q / Esc = quit
"""
import argparse
import cv2
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def handle_key(engine, key):
    if key in (ord("d"), ord("D")) and engine.fire is not None:
        engine.fire.toggle_screen_demo()
        engine.banner[:] = [row for row in engine.banner
                            if "FIRE" not in row[1] and "SMOKE" not in row[1]]
        if hasattr(engine, "_active_alerts"):
            engine._active_alerts.clear()
    if key in (ord("t"), ord("T")):
        engine.test_siren()
    return key in (ord("q"), ord("Q"), 27)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="video file | webcam index | rtsp/http url (defaults to config.yaml)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270])
    ap.add_argument("--flip-horizontal", action="store_true",
                    help="unmirror a front-camera phone feed")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save", help="write annotated video to this path")
    ap.add_argument("--realtime", action="store_true", help="play video files at their natural speed")
    ap.add_argument("--max-seconds", type=float, help="stop after this many source seconds")
    ap.add_argument("--disable", nargs="*", default=[], choices=["fall", "fight", "fire"])
    ap.add_argument("--fire-demo", action="store_true",
                    help="use screen-test fire motion settings for visible flames on a display")
    a = ap.parse_args()
    if a.fire_demo and "fire" in a.disable:
        ap.error("--fire-demo requires the fire detector")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    cfg = load_config(a.config)
    for k in a.disable:
        cfg[k]["enabled"] = False
    if a.fire_demo:
        cfg["fire"].update(enabled=True, screen_demo=True)
    if a.rotate is not None:
        cfg["camera"]["rotate"] = a.rotate

    from kavach.pipeline import Engine
    from kavach.sources import Source
    eng = Engine(cfg)
    source = a.source if a.source is not None else cfg["camera"].get("source", "0")
    src = Source(source, cfg["camera"]["rotate"], cfg["camera"]["max_width"], a.realtime,
                 a.flip_horizontal)
    writer = None
    title = None
    print(f"Running on '{source}'. D = toggle screen fire test, T = test siren, Q = quit.")
    try:
        for frame, t in src.frames():
            if a.max_seconds and t > a.max_seconds:
                break
            if frame is None:                       # live source waiting for data
                eng.on_source_gap(t)
                if not a.no_display and handle_key(eng, cv2.waitKey(1) & 0xFF):
                    break
                continue
            _, vis = eng.process(frame, t)
            if a.save:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             getattr(src, "fps", 15.0), (w, h))
                writer.write(vis)
            if not a.no_display:
                cv2.imshow("KAVACH-AI", vis)
                demo = eng.fire is not None and eng.fire.screen_demo
                current_title = "KAVACH-AI | SCREEN FIRE TEST" if demo else "KAVACH-AI | LIVE CAMERA"
                if title != current_title:
                    cv2.setWindowTitle("KAVACH-AI", current_title)
                    title = current_title
                if handle_key(eng, cv2.waitKey(1) & 0xFF):
                    break
    finally:
        src.close()
        eng.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    print(f"Done. {eng.sink.rows} alert(s) logged to {cfg['alerts']['csv_path']}")


if __name__ == "__main__":
    main()
