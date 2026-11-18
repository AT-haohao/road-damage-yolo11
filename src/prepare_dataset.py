#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert VOC xml / COCO json annotations into YOLO txt labels.

YOLO format per line: <class_id> <x_center> <y_center> <width> <height>
all normalised to 0..1.
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET

CLASSES = ["person", "car", "trafficcone", "pothole"]


def clip(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def voc_to_yolo(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASSES:
            continue
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        cx = clip((xmin + xmax) / 2.0 / width)
        cy = clip((ymin + ymax) / 2.0 / height)
        w = clip((xmax - xmin) / width)
        h = clip((ymax - ymin) / height)
        lines.append("%d %.6f %.6f %.6f %.6f" % (CLASSES.index(name), cx, cy, w, h))
    return lines


def coco_to_yolo(json_path):
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    images = {img["id"]: img for img in data.get("images", [])}
    name_by_id = {}
    for cat in data.get("categories", []):
        if cat["name"] in CLASSES:
            name_by_id[cat["id"]] = cat["name"]

    per_image = {}
    for ann in data.get("annotations", []):
        if ann["category_id"] not in name_by_id:
            continue
        img = images[ann["image_id"]]
        width = float(img["width"])
        height = float(img["height"])
        x, y, w, h = ann["bbox"]
        cx = clip((x + w / 2.0) / width)
        cy = clip((y + h / 2.0) / height)
        cls = CLASSES.index(name_by_id[ann["category_id"]])
        per_image.setdefault(img["file_name"], []).append(
            "%d %.6f %.6f %.6f %.6f" % (cls, cx, cy, clip(w / width), clip(h / height))
        )
    return per_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["voc", "coco"], required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    count = 0

    if args.format == "voc":
        for name in os.listdir(args.src):
            if not name.lower().endswith(".xml"):
                continue
            lines = voc_to_yolo(os.path.join(args.src, name))
            stem = os.path.splitext(name)[0]
            with open(os.path.join(args.dst, stem + ".txt"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + ("\n" if lines else ""))
            count += 1
    else:
        for name in os.listdir(args.src):
            if not name.lower().endswith(".json"):
                continue
            per_image = coco_to_yolo(os.path.join(args.src, name))
            for image_name, lines in per_image.items():
                stem = os.path.splitext(image_name)[0]
                with open(os.path.join(args.dst, stem + ".txt"), "w", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")
                count += 1

    print("converted %d file(s) -> %s" % (count, args.dst))


if __name__ == "__main__":
    main()
