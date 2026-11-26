import os
import numpy as np
from ultralytics import YOLO
from .fall import has_body_pose


def resolve(path):
    """Use the file if present (also looks in ./models); otherwise let Ultralytics download it by name."""
    for p in (path, os.path.join("models", path)):
        if os.path.exists(p):
            return p
    return path


class PoseEstimator:
    def __init__(self, cfg):
        self.c = cfg
        self.model = YOLO(resolve(cfg["model"]))

    def detect(self, frame):
        c = self.c
        r = self.model.predict(frame, imgsz=c["imgsz"], conf=c["conf"], device=c["device"], verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        kxy = r.keypoints.xy.cpu().numpy()
        kcf = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
               else np.zeros(kxy.shape[:2]))
        min_h = c["min_box_h_frac"] * frame.shape[0]
        out = []
        for b, k, cf in zip(boxes, kxy, kcf):
            # Allow a wide, flat person with a reliable pose through the size filter.
            if b[3] - b[1] >= min_h or (b[2] - b[0] >= min_h and has_body_pose(k, cf)):
                out.append((b, k, cf))
        return out
