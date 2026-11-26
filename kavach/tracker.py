"""Small IoU/centroid tracker so every person keeps an ID between frames (no extra deps)."""
import numpy as np
from .types import Person


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    def __init__(self, max_age_s=2.0):
        self.max_age_s = max_age_s
        self.tracks = {}      # id -> dict(box, t)
        self.next_id = 1

    def update(self, dets, t):
        """dets: list of (box[4], kpts[17,2], conf[17]). Returns list[Person]."""
        for tid in [k for k, v in self.tracks.items() if t - v["t"] > self.max_age_s]:
            del self.tracks[tid]

        pairs = []
        for di, (box, _, _) in enumerate(dets):
            dh = max(box[3] - box[1], 1.0)
            dc = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            for tid, tr in self.tracks.items():
                tb = tr["box"]
                th = max(tb[3] - tb[1], 1.0)
                tc = np.array([(tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2])
                d = np.linalg.norm(dc - tc) / max(dh, th)
                iou = _iou(box, tb)
                if iou > 0.10 or d < 0.5:
                    pairs.append((d - iou, di, tid))
        pairs.sort()
        used_d, used_t, assign = set(), set(), {}
        for _, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            assign[di] = tid
            used_d.add(di)
            used_t.add(tid)

        people = []
        for di, (box, kp, cf) in enumerate(dets):
            tid = assign.get(di)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
            self.tracks[tid] = {"box": np.asarray(box, float), "t": t}
            people.append(Person(tid, np.asarray(box, float), np.asarray(kp, float), np.asarray(cf, float)))
        return people
