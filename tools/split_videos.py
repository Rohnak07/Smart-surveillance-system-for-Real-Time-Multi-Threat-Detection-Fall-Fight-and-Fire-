"""Split raw violence videos 80/10/10 BY VIDEO (so frames of one clip never leak across splits).
raw layout:   violence_raw/violence/*.mp4   violence_raw/non_violence/*.mp4
"""
import os, random, shutil, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "violence_raw"
DST = sys.argv[2] if len(sys.argv) > 2 else "violence_split"
EXT = (".mp4", ".avi", ".mov", ".mkv")
random.seed(42)

for cls in ("violence", "non_violence"):
    vids = sorted(f for f in os.listdir(os.path.join(SRC, cls)) if f.lower().endswith(EXT))
    random.shuffle(vids)
    n = len(vids); a = int(n * 0.8); b = a + int(n * 0.1)
    for split, items in (("train", vids[:a]), ("val", vids[a:b]), ("test", vids[b:])):
        os.makedirs(os.path.join(DST, split, cls), exist_ok=True)
        for v in items:
            shutil.copy2(os.path.join(SRC, cls, v), os.path.join(DST, split, cls, v))
print("split done ->", DST)
