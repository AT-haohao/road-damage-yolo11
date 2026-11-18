#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a trained detector over images and save annotated output."""

import argparse
import os

from ultralytics import YOLO


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="image, folder or video")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="runs/predict")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=True,
        project=args.out,
        name="exp",
    )
    print("processed %d item(s) -> %s" % (len(results), os.path.join(args.out, "exp")))
    for item in results:
        names = item.names
        counts = {}
        for cls in item.boxes.cls.tolist():
            label = names[int(cls)]
            counts[label] = counts.get(label, 0) + 1
        if counts:
            print("  %s: %s" % (os.path.basename(item.path), counts))


if __name__ == "__main__":
    main()
