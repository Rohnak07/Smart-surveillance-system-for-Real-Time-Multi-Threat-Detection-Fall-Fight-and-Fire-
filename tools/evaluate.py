"""Measure the system on YOUR labelled clips (clip-level metrics, honest numbers only).

labels.csv columns:  video,label       label in: fall | fight | fire | normal
Example:
    videos/fall_01.mp4,fall
    videos/walk_03.mp4,normal
Usage:  python tools/evaluate.py --labels labels.csv --out results
Writes results/clip_results.csv and results/metrics.md (precision / recall / F1 / false alarms per
minute of normal footage / detection delay / processing FPS).
"""
import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

KIND_MAP = {"fall": {"fall", "person_down"}, "fight": {"fight"}, "fire": {"fire", "smoke"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    cfg["alerts"]["save_evidence"] = False
    cfg["alerts"]["cooldown_s"] = 1e9            # first detection per kind is all we need per clip
    os.makedirs(a.out, exist_ok=True)

    from kavach.pipeline import Engine
    from kavach.sources import Source
    rows = list(csv.DictReader(open(a.labels, encoding="utf-8")))
    res, normal_minutes, frames_total, secs_total = [], 0.0, 0, 0.0
    for r in rows:
        cfg["alerts"]["csv_path"] = os.path.join(a.out, "_tmp_events.csv")
        eng = Engine(cfg)                         # fresh state per clip
        src = Source(r["video"], cfg["camera"]["rotate"], cfg["camera"]["max_width"])
        found, first_t, dur, t0 = {}, {}, 0.0, time.perf_counter()
        for frame, t in src.frames():
            evs, _ = eng.process(frame, t)
            frames_total += 1
            dur = t
            for e in evs:
                found.setdefault(e.kind, e.confidence)
                first_t.setdefault(e.kind, t)
        secs_total += time.perf_counter() - t0
        src.close()
        label = r["label"].strip().lower()
        if label == "normal":
            normal_minutes += dur / 60
        res.append({"video": r["video"], "label": label, "duration_s": round(dur, 1),
                    "detected": ";".join(sorted(found)) or "-",
                    "first_alert_s": ";".join(f"{k}@{v:.1f}" for k, v in first_t.items()) or "-"})
        print(res[-1])

    with open(os.path.join(a.out, "clip_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)

    lines = ["# Evaluation (clip level)", f"Clips: {len(res)}", ""]
    lines += ["| Event | TP | FP | FN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
    for ev, kinds in KIND_MAP.items():
        tp = sum(1 for x in res if x["label"] == ev and set(x["detected"].split(";")) & kinds)
        fn = sum(1 for x in res if x["label"] == ev) - tp
        fp = sum(1 for x in res if x["label"] != ev and set(x["detected"].split(";")) & kinds)
        p = tp / (tp + fp) if tp + fp else float("nan"); rc = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * rc / (p + rc) if p == p and rc == rc and p + rc else float("nan")
        lines.append(f"| {ev} | {tp} | {fp} | {fn} | {p:.2f} | {rc:.2f} | {f1:.2f} |")
    fa = sum(1 for x in res if x["label"] == "normal" and x["detected"] != "-")
    lines += ["", f"Normal clips with any false alert: {fa} over {normal_minutes:.1f} min of normal footage",
              f"Processing speed: {frames_total / max(secs_total, 1e-6):.1f} FPS (this machine, includes drawing)"]
    open(os.path.join(a.out, "metrics.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
