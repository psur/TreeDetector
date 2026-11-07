"""Export YOLO crown polygons without writing into a training experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "runs/yolo_seg_20260912_193929_824528/training/weights/best.pt"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(args):
    from PIL import Image
    from ultralytics import YOLO
    import ultralytics

    source = args.source.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise ValueError("No JPG images found")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; choose a new output path")
    model_id = "sha256:" + sha256(checkpoint)
    model = YOLO(str(checkpoint))
    if model.task != "segment":
        raise ValueError("A segmentation checkpoint is required")
    records = []
    # Bound memory independently of directory size. Each file is processed once.
    for start in range(0, len(files), args.batch):
        batch = files[start:start + args.batch]
        expected = {str(p.resolve()): p for p in batch}
        seen = set()
        for result in model.predict(source=[str(p) for p in batch], stream=True,
                                    batch=args.batch, conf=args.conf, iou=args.iou,
                                    imgsz=args.imgsz, device=args.device,
                                    retina_masks=True, save=False, save_txt=False,
                                    save_crop=False, verbose=False):
            key = str(Path(result.path).resolve())
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate inference result: {key}")
            seen.add(key)
            path = expected[key]
            height, width = map(int, result.orig_shape)
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError(f"Original dimensions disagree: {path}")
            predictions = []
            boxes, masks = result.boxes, result.masks
            if boxes is not None and len(boxes):
                if masks is None or len(masks.xy) != len(boxes):
                    raise ValueError(f"Masks and boxes disagree: {path}")
                for index, (box, polygon) in enumerate(zip(boxes, masks.xy)):
                    class_id = int(box.cls.item())
                    model_class = result.names[class_id]
                    label = {"tree": "tree", "TreeGroup": "treeGroup", "treeGroup": "treeGroup"}.get(model_class)
                    if label is None:
                        raise ValueError(f"Unknown classroom class: {model_class}")
                    points = polygon.astype(float).tolist()
                    if len(points) < 3:
                        continue  # Degenerate masks cannot be rendered as polygons.
                    if any(not math.isfinite(x) or not math.isfinite(y) or
                           not 0 <= x <= width or not 0 <= y <= height for x, y in points):
                        raise ValueError(f"Invalid original-image polygon: {path}")
                    confidence = float(box.conf.item())
                    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise ValueError("Invalid confidence")
                    predictions.append({"id": f"{path.name}:{index}", "polygon": points,
                                        "class": label, "class_id": class_id,
                                        "model_class": model_class, "confidence": confidence,
                                        "model_checkpoint_id": model_id})
            records.append({"filename": path.name, "width": width, "height": height,
                            "image_sha256": sha256(path), "predictions": predictions})
        if seen != set(expected):
            raise ValueError("Missing inference results")
    payload = {"schema_version": 1, "run_id": args.run_id, "stage": args.stage,
               "coordinate_system": "original-image pixels; origin top-left; x right, y down",
               "polygon_format": "[[x,y],...]; implicitly closed; Ultralytics masks.xy exterior contours",
               "model": {"checkpoint": str(checkpoint), "id": model_id,
                         "ultralytics_version": ultralytics.__version__},
               "settings": {"confidence": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                            "retina_masks": True, "batch": args.batch, "device": args.device},
               "source_directory": str(source), "image_count": len(records), "images": records}
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another process creating the file.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(serialized + "\n")
    print(f"Exported {len(records)} images to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=REFERENCE)
    parser.add_argument("--run-id", default="class-20261108")
    parser.add_argument("--stage", choices=("baseline", "finetuned"), default="baseline")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch < 1 or args.imgsz < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch, imgsz and limit must be positive")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("conf and iou must be in [0,1]")
    export(args)


if __name__ == "__main__":
    main()
