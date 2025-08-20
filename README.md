# Quickstart


## 0) Create & activate venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1


# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate


## 1) Install deps
pip install -r requirements.txt


## 2) Download & prepare dataset (first run only)
python prepare_dataset.py --output data


This will:
- Download `gti-upm/leapgestrecog` via kagglehub
- Normalize the folder layout into `data/raw_by_class/<gesture>/...`
- Split into `data/train`, `data/val`, `data/test` (70/15/15)
- Write `data/class_names.json`


## 3) Train model
python train.py --data-root data --epochs 12 --batch-size 64 --lr 3e-4 --img-size 224


Artifacts saved to `artifacts/`:
- `best_model.pth` (weights)
- `class_names.json` (labels)
- `metrics.json` (accuracy, f1)
- `confusion_matrix.png`


## 4) Test on a single image
python predict_image.py --image /path/to/your_image.png --weights artifacts/best_model.pth --labels artifacts/class_names.json


## 5) Run on a video file
python predict_video.py --video /path/to/video.mp4 --weights artifacts/best_model.pth --labels artifacts/class_names.json


## 6) Real-time webcam demo
python demo_webcam.py --weights artifacts/best_model.pth --labels artifacts/class_names.json


## Tips
- If pretrained weights fail to download, training falls back to random init automatically.
- Increase `--epochs` to improve accuracy.
- If your webcam index isn’t 0, pass `--camera 1` (or 2, 3...).
