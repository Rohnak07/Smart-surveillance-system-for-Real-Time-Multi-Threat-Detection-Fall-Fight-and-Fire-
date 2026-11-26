from dataclasses import dataclass
import numpy as np

# COCO-17 keypoint indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16


@dataclass
class Person:
    id: int
    box: np.ndarray    # x1, y1, x2, y2 (pixels)
    kpts: np.ndarray   # (17, 2) pixels
    conf: np.ndarray   # (17,)

    @property
    def w(self): return float(self.box[2] - self.box[0])

    @property
    def h(self): return float(self.box[3] - self.box[1])

    @property
    def center(self): return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


@dataclass
class Event:
    kind: str                 # "fall" | "person_down" | "fight" | "fire" | "smoke"
    confidence: float
    t: float                  # source time in seconds
    info: str = ""
    box: tuple = None         # optional (x1, y1, x2, y2)
    subject_id: object = None # stable person/group identity for per-subject alert handling

    @property
    def label(self):
        return {"fall": "FALL DETECTED", "person_down": "POSSIBLE MEDICAL EMERGENCY",
                "fight": "FIGHT DETECTED", "fire": "FIRE DETECTED",
                "smoke": "SMOKE DETECTED"}.get(self.kind, self.kind.replace("_", " ").upper())
