"""Fine-tune a YOLO fire+smoke detector on the D-Fire dataset (YOLO format already).
1) git clone https://github.com/gaia-solutions-on-demand/DFireDataset   (see its README for the data download)
2) edit data/dfire.yaml so `path:` points to the dataset folder
3) python tools/train_fire.py --epochs 50         # GPU strongly recommended (Colab / Kaggle are fine)
Result: runs/detect/fire/weights/best.pt  ->  copy to models/fire_smoke.pt (the app picks it up automatically).
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dfire.yaml")
ap.add_argument("--base", default="yolo11n.pt")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device", default="0")
a = ap.parse_args()
m = YOLO(a.base)
m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, device=a.device, project="runs/detect", name="fire", patience=15)
print(m.val(data=a.data, split="test"))      # report test-set metrics honestly in your report
