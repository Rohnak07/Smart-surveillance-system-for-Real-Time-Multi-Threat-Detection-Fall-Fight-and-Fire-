"""Synthetic skeleton generators (no camera / model needed) used by the unit tests."""
import numpy as np
from kavach.types import Person


def skeleton(cx, foot_y, h, lying=False, lean=0.0, arms=None, legs=None):
    """Build a COCO-17 skeleton. Standing: head at foot_y-h. Lying: body horizontal."""
    k = np.zeros((17, 2)); c = np.ones(17)
    if not lying:
        top = foot_y - h
        sh_y, hip_y = top + 0.18 * h, top + 0.55 * h
        sw, hw = 0.12 * h, 0.08 * h
        k[0] = (cx + lean, top + 0.05 * h)
        k[5], k[6] = (cx - sw + lean, sh_y), (cx + sw + lean, sh_y)
        k[11], k[12] = (cx - hw, hip_y), (cx + hw, hip_y)
        k[13], k[14] = (cx - hw, top + 0.77 * h), (cx + hw, top + 0.77 * h)
        k[15], k[16] = (cx - hw, foot_y), (cx + hw, foot_y)
        k[7], k[8] = (cx - sw - 0.02 * h, sh_y + 0.18 * h), (cx + sw + 0.02 * h, sh_y + 0.18 * h)
        k[9], k[10] = (cx - sw - 0.02 * h, sh_y + 0.35 * h), (cx + sw + 0.02 * h, sh_y + 0.35 * h)
    else:  # lying on the floor, head to the left
        y = foot_y - 0.08 * h
        k[0] = (cx - 0.45 * h, y)
        k[5], k[6] = (cx - 0.32 * h, y - 0.06 * h), (cx - 0.32 * h, y + 0.06 * h)
        k[11], k[12] = (cx, y - 0.04 * h), (cx, y + 0.04 * h)
        k[13], k[14] = (cx + 0.25 * h, y - 0.04 * h), (cx + 0.25 * h, y + 0.04 * h)
        k[15], k[16] = (cx + 0.5 * h, y - 0.04 * h), (cx + 0.5 * h, y + 0.04 * h)
        k[7], k[8] = (cx - 0.2 * h, y - 0.1 * h), (cx - 0.2 * h, y + 0.1 * h)
        k[9], k[10] = (cx - 0.05 * h, y - 0.1 * h), (cx - 0.05 * h, y + 0.1 * h)
    for i in (1, 2, 3, 4):                 # eyes / ears sit on the nose
        k[i] = k[0]
    if arms:
        for idx, (x, y) in arms.items():
            k[idx] = (x, y)
    if legs:
        for idx, (x, y) in legs.items():
            k[idx] = (x, y)
    return k, c


def person(pid, k, c):
    pts = k[c > 0]
    pad = 6
    box = np.array([pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad])
    return Person(pid, box, k, c)
