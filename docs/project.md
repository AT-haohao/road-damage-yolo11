# Project notes

Context, decisions and lessons from the competition entry this repository is
based on. Written down mainly so the numbers in the README can be read with the
right caveats.

## The problem

Road maintenance teams inspect roads on foot or with a slow survey vehicle.
Potholes get reported by residents, usually weeks late. A detector that runs on
a phone or a dashcam could flag them at the moment of passing.

Four classes were chosen because they cover both safety and asset management:

- person, car — dynamic scene understanding
- trafficcone — temporary works, useful for routing
- pothole — the asset that actually costs money

## What was actually built

Baseline: ```yolo11n.pt``` from Ultralytics, 640 px input, standard mosaic /
mixup / HSV augmentation, cosine LR, early stopping at 30 epochs of patience.
Trained roughly 100 epochs on a 6 GB laptop GPU with ```amp=True``` and batch 16.

Reported outcome: mAP@0.5 = 0.923, mAP@0.5:0.95 = 0.65, recall near 97% at the
lowest confidence threshold.

## About the EfficientNet-B7 and CBAM claims

The competition deck describes replacing the YOLO11n backbone with
EfficientNet-B7 and inserting CBAM attention blocks. Two problems with that
description:

1. **It cannot keep the parameter budget.** EfficientNet-B7 alone is about 66 M
   parameters. A detector built on it is not a "3 M parameter" model, and it would
   not have trained on a 6 GB card at the batch sizes used.
2. **The reported mAP was produced by the YOLO11n baseline**, not by a B7
   variant.

So the deck mixed an explored direction with the delivered result. This repo
documents only what was delivered. Anyone rebuilding this should treat the
backbone swap as future work, not as the source of the numbers.

## If I did it again

- **Split potholes from the rest.** The four classes have wildly different
  difficulty and frequency; one model with one loss weighting underfits potholes.
- **Collect night and rain images deliberately.** Roughly half the gain came from
  adding bad-weather frames, not from tuning.
- **Report per-class AP.** A single mAP hides the fact that pothole AP is far
  below car AP.
- **Freeze the val set by hand.** Random splits leak near-duplicate frames from
  the same video into validation and inflate mAP.

## Data

Not included. The competition data belongs to the organiser. The pipeline in
this repo is class-name driven; point ```configs/road_damage.yaml``` at any
dataset with the same four classes and it will run unchanged.
