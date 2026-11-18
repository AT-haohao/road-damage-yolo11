#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a YOLO11 detector on the road-damage dataset."""

import argparse

from ultralytics import YOLO


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/road_damage.yaml")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="yolo11n / yolo11s / path to a .pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu' otherwise")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        amp=True,
        cache=False,
        patience=30,
        cos_lr=True,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,
        fliplr=0.5,
    )

    metrics = model.val()
    print("mAP50   : %.4f" % metrics.box.map50)
    print("mAP50-95: %.4f" % metrics.box.map)


if __name__ == "__main__":
    main()
