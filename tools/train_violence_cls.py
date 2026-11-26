"""Train the OPTIONAL violence CNN (MobileNetV3-Large) from videos split by tools/split_videos.py.
python tools/train_violence_cls.py --data violence_split --epochs 5 --pretrained
Saves models/violence_v1.pt in the same format the app expects and prints VIDEO-level test accuracy.
Tip: this dataset is studio-like; always check the model on your own CCTV-style clips before trusting it.
"""
import argparse, os, random
import cv2, numpy as np, torch, torch.nn as nn
from torchvision import models, transforms

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="violence_split")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--frames_per_video", type=int, default=12)
ap.add_argument("--pretrained", action="store_true")
ap.add_argument("--out", default="models/violence_v1.pt")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
C2I = {"non_violence": 0, "violence": 1}
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.ToPILImage(), transforms.RandomResizedCrop(224, scale=(0.6, 1)),
                               transforms.RandomHorizontalFlip(), transforms.ColorJitter(.3, .3, .3),
                               transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(), norm])


def frames(path, k):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    for i in np.linspace(0, max(n - 1, 0), k).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if ok: out.append(f[..., ::-1].copy())
    cap.release(); return out


def load(split):
    items = []
    for cls, idx in C2I.items():
        d = os.path.join(a.data, split, cls)
        items += [(os.path.join(d, f), idx) for f in sorted(os.listdir(d))]
    return items


m = models.mobilenet_v3_large(weights="DEFAULT" if a.pretrained else None)
m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
m.to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()
tr = load("train")
for ep in range(a.epochs):
    m.train(); random.shuffle(tr); tot = 0
    for p, y in tr:
        fr = frames(p, a.frames_per_video)
        if not fr: continue
        x = torch.stack([train_tf(f) for f in fr]).to(dev)
        loss = lossf(m(x), torch.full((len(fr),), y, device=dev))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f"epoch {ep+1}/{a.epochs} loss {tot/len(tr):.4f}")


def video_acc(split):
    m.eval(); ok = n = 0
    with torch.no_grad():
        for p, y in load(split):
            fr = frames(p, 16)
            if not fr: continue
            pr = torch.softmax(m(torch.stack([eval_tf(f) for f in fr]).to(dev)), 1)[:, 1].mean().item()
            ok += int((pr >= 0.5) == bool(y)); n += 1
    return ok / max(n, 1), n


for s in ("val", "test"):
    acc, n = video_acc(s); print(f"{s}: video-level accuracy {acc:.3f} on {n} videos")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({"model_state": m.state_dict(), "class_to_idx": C2I}, a.out)
print("saved", a.out)
