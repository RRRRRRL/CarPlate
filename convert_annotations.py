import os
import xml.etree.ElementTree as ET
import argparse
from PIL import Image


def convert_voc_to_yolo(xml_path, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return labels


def process(img_dir, ann_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(ann_dir) if f.lower().endswith(".xml")]

    if not xml_files:
        print(f"No XML files found in {ann_dir}")
        return

    for xml_file in xml_files:
        stem = os.path.splitext(xml_file)[0]

        # try common image extensions
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(img_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"Warning: no image found for {xml_file}, skipping")
            continue

        with Image.open(img_path) as img:
            w, h = img.size
        lines = convert_voc_to_yolo(os.path.join(ann_dir, xml_file), w, h)

        out_path = os.path.join(out_dir, stem + ".txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Converted {len(xml_files)} annotations → {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC XML annotations to YOLO .txt labels."
    )
    parser.add_argument("--images-dir", default="images", help="Path to input images directory")
    parser.add_argument("--annotations-dir", default="annotations", help="Path to VOC XML annotations directory")
    parser.add_argument("--output-dir", default="labels", help="Path to output YOLO labels directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(args.images_dir, args.annotations_dir, args.output_dir)
