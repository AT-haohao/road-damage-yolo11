# road-damage-yolo11

**Four-class road scene detection with YOLO11 — person, car, traffic cone and pothole.**

Built for a provincial data-application competition (Guangxi Public Data Open
Innovation Competition, second prize). The goal was a model light enough to run
on roadside or in-vehicle hardware while still catching small, low-contrast
targets such as potholes at night.

| Class | Why it matters |
|---|---|
| person | pedestrian intrusion warnings |
| car | traffic flow and congestion estimation |
| trafficcone | temporary works, lane closure detection |
| pothole | road asset inspection, suspension damage prevention |

## Results

Reported during the competition, single-stage training run, 100 epochs:

| Metric | Value |
|---|---|
| mAP@0.5 | 0.923 |
| mAP@0.5:0.95 | 0.65 |
| Recall at lowest confidence threshold | ~97% |

YOLO11n baseline, 640 px input. The deck also described experiments with an
EfficientNet-B7 backbone and a CBAM attention block; those were exploratory and
are **not** what produced the numbers above — see `docs/project.md` for an
honest write-up, including why that combination cannot keep a 3M-parameter budget.

## Quick start

```bash
pip install -r requirements.txt

# 1. put images + YOLO labels under datasets/road/ and edit configs/road_damage.yaml
# 2. train
python train.py --data configs/road_damage.yaml --epochs 100 --imgsz 640 --batch 16

# 3. run inference on a folder
python infer.py --weights runs/detect/train/weights/best.pt --source samples/
```

## Dataset layout

```text
datasets/road/
├── images/
│   ├── train/  *.jpg
│   └── val/    *.jpg
└── labels/
    ├── train/  *.txt      # class cx cy w h, normalised
    └── val/    *.txt
```

Class ids: `0 person`, `1 car`, `2 trafficcone`, `3 pothole`.

If your annotations come as VOC xml or COCO json, ```python src/prepare_dataset.py --format voc --src raw/ --dst datasets/road/```
converts them.

## What is in here

| File | Purpose |
|---|---|
| train.py | training entry point, wraps Ultralytics YOLO |
| infer.py | batch inference, saves annotated images |
| src/prepare_dataset.py | annotation conversion (VOC / COCO / YOLO) |
| configs/road_damage.yaml | dataset paths + class names |
| docs/project.md | competition context, decisions, lessons |

## Notes

- The competition dataset belongs to the organiser and is **not** included here.
  The pipeline is dataset-agnostic; point `configs/road_damage.yaml` at your
  own data.
- Training was done on a 6 GB laptop GPU, so batch size stayed at 16 with
  `amp=True`. Larger cards should raise both.
- For potholes specifically, night-time and rainy images matter more than extra
  epochs. Half the gain came from data, not hyperparameters.

## License

MIT for the code. The competition dataset and any derived weights follow the
organiser's terms.
