"""Optional frame-level violence CNN (MobileNetV3-Large, same checkpoint format as the original repo)."""
from collections import deque
import numpy as np


class ViolenceCNN:
    def __init__(self, cfg):
        import torch, torch.nn as nn
        from torchvision import models, transforms
        self.torch, self.cfg = torch, cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
        ck = torch.load(cfg["model"], map_location=self.device)
        m.load_state_dict(ck["model_state"])
        self.vidx = ck.get("class_to_idx", {"non_violence": 0, "violence": 1}).get("violence", 1)
        self.m = m.to(self.device).eval()
        self.tfm = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.hist, self.last_t = deque(), -9.0

    def update(self, frame_bgr, t):
        """Returns mean violence probability over the recent window (or None before first sample)."""
        if t - self.last_t >= self.cfg["sample_every_s"]:
            self.last_t = t
            rgb = frame_bgr[..., ::-1].copy()
            with self.torch.no_grad():
                x = self.tfm(rgb).unsqueeze(0).to(self.device)
                p = self.torch.softmax(self.m(x), 1)[0, self.vidx].item()
            self.hist.append((t, p))
        while self.hist and t - self.hist[0][0] > self.cfg["window_s"]:
            self.hist.popleft()
        return float(np.mean([p for _, p in self.hist])) if self.hist else None
