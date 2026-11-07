"""Exporter contract tests; fake inference avoids touching any real images or models."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from scripts.predict_classroom import export


def setup_export(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    for name in ("a.JPG", "b.jpg"):
        Image.new("RGB", (1200, 300)).save(source / name)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    return argparse.Namespace(source=source, checkpoint=checkpoint,
                              output=tmp_path / "output.json", limit=None,
                              batch=2, conf=0.25, iou=0.7, imgsz=640,
                              device="cpu", run_id="class-20261108", stage="baseline")


def fake_result(path, empty=False):
    return SimpleNamespace(path=str(path), orig_shape=(300, 1200),
                           names={0: "tree", 1: "TreeGroup"},
                           boxes=[] if empty else [SimpleNamespace(cls=np.array(1), conf=np.array(0.9))],
                           masks=None if empty else SimpleNamespace(xy=[np.array([[800, 100], [1100, 100], [1100, 280]])]))


def test_filename_alignment_original_pixels_empty_image_and_class_mapping(tmp_path):
    args = setup_export(tmp_path)
    # Return results out of order to ensure identity comes from result.path.
    model = SimpleNamespace(task="segment", predict=lambda **kwargs: iter([
        fake_result(args.source / "b.jpg", empty=True), fake_result(args.source / "a.JPG")]))
    with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model), __version__="test")}):
        export(args)
    records = {r["filename"]: r for r in json.loads(args.output.read_text())["images"]}
    assert len(records) == 2 and records["b.jpg"]["predictions"] == []
    record = records["a.JPG"]
    assert (record["width"], record["height"]) == (1200, 300)
    prediction = record["predictions"][0]
    assert prediction["polygon"] == [[800, 100], [1100, 100], [1100, 280]]
    assert prediction["class"] == "treeGroup" and prediction["model_class"] == "TreeGroup"
    original = args.output.read_bytes()
    loader = Mock()
    with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=loader)}), pytest.raises(FileExistsError):
        export(args)
    loader.assert_not_called()
    assert args.output.read_bytes() == original


@pytest.mark.parametrize("failure", ["duplicate", "missing", "outside"])
def test_reject_invalid_results_without_publishing(tmp_path, failure):
    args = setup_export(tmp_path)
    first = fake_result(args.source / "a.JPG")
    results = [first, fake_result(args.source / "b.jpg")]
    if failure == "duplicate":
        results = [first, first]
    elif failure == "missing":
        results = [first]
    else:
        first.masks.xy[0][0][0] = 1201
    model = SimpleNamespace(task="segment", predict=lambda **kwargs: iter(results))
    with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model), __version__="test")}), pytest.raises(ValueError):
        export(args)
    assert not args.output.exists()
