"""Validate a classroom export and create a standalone browser preview (no inference)."""
import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.predictions.read_text(encoding="utf-8"))
    names, ids, sections = set(), set(), []
    count = 0
    for record in data["images"]:
        name = record["filename"]
        assert Path(name).name == name and name not in names, "Invalid or duplicate filename"
        names.add(name)
        path = Path(data["source_directory"]) / name
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["image_sha256"]
        with Image.open(path) as image:
            width, height = image.size
        assert (width, height) == (record["width"], record["height"])
        shapes = []
        for prediction in record["predictions"]:
            assert prediction["id"] not in ids, "Duplicate prediction ID"
            ids.add(prediction["id"])
            assert prediction["model_checkpoint_id"] == data["model"]["id"]
            assert prediction["class"] in {"tree", "treeGroup"}
            assert 0 <= prediction["confidence"] <= 1
            polygon = prediction["polygon"]
            assert len(polygon) >= 3
            assert all(math.isfinite(x) and math.isfinite(y) and
                       0 <= x <= width and 0 <= y <= height for x, y in polygon)
            points = " ".join(f"{x},{y}" for x, y in polygon)
            color = "#00ff80" if prediction["class"] == "tree" else "#ffb000"
            title = html.escape(f"{prediction['class']} {prediction['confidence']:.3f}")
            shapes.append(f'<polygon points="{points}" fill="{color}" fill-opacity="0.15" '
                          f'stroke="{color}" stroke-width="2"><title>{title}</title></polygon>')
            count += 1
        encoded = base64.b64encode(raw).decode("ascii")
        sections.append(f'<section><h2>{html.escape(name)}</h2><p>{width} × {height}; '
                        f'{len(record["predictions"])} polygons</p><svg viewBox="0 0 {width} {height}">'
                        f'<image href="data:image/jpeg;base64,{encoded}" width="{width}" height="{height}"/>'
                        + "".join(shapes) + '</svg></section>')
    assert len(names) == data["image_count"]
    page = ('<!doctype html><meta charset="utf-8"><title>Classroom prediction validation</title>'
            '<style>body{font:16px sans-serif;background:#222;color:white;margin:24px}'
            'section{max-width:900px}svg{width:100%;height:auto}</style>'
            '<h1>Classroom baseline validation</h1><p>Green: tree. Orange: treeGroup. '
            'Hover polygons for confidence. Coordinates use the original image viewBox.</p>'
            + "".join(sections))
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    with args.preview.open("x", encoding="utf-8") as stream:
        stream.write(page)
    print(f"Validated {len(names)} unique images, {count} polygons; preview: {args.preview}")


if __name__ == "__main__":
    main()
