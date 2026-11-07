# Classroom baseline predictions — class-20261108

## Reference checkpoint and benchmark audit

Selected experiment: `yolo_seg_20260912_193929_824528` (YOLO11m-seg).
Exact checkpoint: `D:\TreeDetector\runs\yolo_seg_20260912_193929_824528\training\weights\best.pt`.
Checkpoint SHA-256: `6ffc771810afaa74025d51c62ac18ae1d36201da9b4edddca89ccfd9b5fcac7a`.

Its `experiment.json`, `metrics.json`, and `results/benchmark_results.csv` identify
mask mAP50 **0.7188771978416729** and mask mAP50–95 **0.3646056466315642**,
which round to the requested scores. Selection was by these scores, not timestamp.

| Experiments | Dataset | Test images / instances | Dataset fingerprint |
| --- | --- | --- | --- |
| YOLO September 12; Mask2Former September 15 | benchmark_v4 | 25 / 492 | `9c822683ddab22925bcdde5b6b577baea80f2ca8f4353f83d6edd41632042174` |
| YOLO September 17; Mask2Former September 18 | benchmark_v5 | 40 / 618 | `eb8c714787175ffe52ababcb2750fc159d0ac622996bb852cbc275e443e666bc` |

Recomputed both manifest fingerprints and matched them to all four experiment
records. The v4 YOLO training args and Mask2Former saved configuration both point
to benchmark_v4. Its YOLO test manifest IDs exactly match the COCO test image IDs
(25 images, 492 annotations), and the COCO metadata carries the same fingerprint.
Thus each dataset's model pair shares a test set; the two YOLO experiments do
**not** share a test set. Native Ultralytics and COCO metrics also use different
evaluators, as recorded in the benchmark CSV. The newer YOLO scores are
0.6125794566893706 / 0.32122159686248647 on v5; they do not replace the requested v4 baseline.

## Commands

Run from `D:\TreeDetector`. Existing `.venv-yolo` supplies Ultralytics 8.4.150.
The new command defaults to the exact checkpoint above, CPU, confidence 0.25,
IoU 0.7, image size 640, retina masks, and batch size 1. No dataset preparation,
training, or benchmark command is involved.

Five-image validation (executed):

```powershell
.venv-yolo/Scripts/python.exe scripts/predict_classroom.py --source images/randomSelect --limit 5 --output outputs/class-20261108/baseline-validation-5.json
.venv-yolo/Scripts/python.exe scripts/validate_classroom_predictions.py outputs/class-20261108/baseline-validation-5.json --preview outputs/class-20261108/baseline-validation-5.html
.venv-prep/Scripts/python.exe -m pytest tests/test_classroom_predictions.py -q
```

Full classroom export (prepared, **not executed**):

```powershell
.venv-yolo/Scripts/python.exe scripts/predict_classroom.py --source images/randomSelect --output outputs/class-20261108/baseline-200.json
```

`images/randomSelect` currently contains exactly 200 JPG images. Any other JPG
directory can be supplied with `--source`. `--limit 5` selects five filenames in
sorted order. Processing is bounded by `--batch`, so a 200-image directory is
not loaded into memory all at once. Every source filename gets one image record,
including images with zero detections. Repeated assignments across the 42
students should join to this single image record by filename; student assignments
and student exports are not part of inference.

Both commands refuse existing output files. A repeat of the five-image inference
command was rejected before loading the model; the existing JSON hash was
unchanged. Choose a new output filename for a new export. Nothing is appended,
and no Ultralytics image, label, or experiment output is saved.

For a later fine-tuned comparison, explicitly pass `--checkpoint PATH_TO_NEW_BEST_PT`,
`--stage finetuned`, and a distinct `--output`. Keep the same source images and
inference settings to compare predictions. No fine-tuning was performed here.

## JSON and browser coordinates

The JSON includes schema version, classroom run ID, stage, source directory,
checkpoint path and SHA-256 identifier, library version, inference settings, and
an `images` array. Image records contain `filename`, original `width` and `height`,
an image SHA-256, and a `predictions` array. Predictions contain a polygon,
class ID, classroom `class`, original `model_class`, confidence, prediction ID,
and checkpoint identifier.

Polygons are arrays of `[x,y]` in **original-image pixels**, with origin at the
top left, x increasing rightward, and y downward. They are implicitly closed.
The implementation uses `result.masks.xy`; the installed Ultralytics implementation
explicitly scales these contours to `orig_shape` with `normalize=False`.
Degenerate contours with fewer than three vertices are omitted. These exterior
contours use Ultralytics' mask-to-segment conversion, which may connect disjoint
pieces and does not encode holes; they are intended for crown overlay comparison,
not a lossless raster-mask archive.

Model `tree` maps to classroom `tree`; model `TreeGroup` maps to classroom
`treeGroup`. Raw class names are retained, and unexpected classes cause failure.
Unknown, duplicate, or missing result paths, mismatched dimensions, and invalid
coordinates fail before JSON publication.

In a browser, use an SVG `viewBox="0 0 WIDTH HEIGHT"`, an image with that width
and height, and `<polygon points="x,y x,y ...">`. Both then scale together.
Alternatively multiply x and y by the canvas display width/original width and
display height/original height. Do not interpret the values as normalized or
640-pixel model-input coordinates. Label Studio percentages, if needed later,
are `100*x/width` and `100*y/height`.

The generated HTML is standalone: it embeds the five source JPEGs and polygon
overlays in SVG, avoiding local-file fetch restrictions. Open it directly in a
browser. Green represents tree and orange represents treeGroup; polygon tooltips
show class and confidence. The artifact was generated and its geometry validated;
an interactive browser session was not run.

## Validation results

| Filename | Original size | Exported polygons |
| --- | --- | ---: |
| DJI_0293_W_34.JPG | 1024 × 1024 | 31 |
| DJI_0293_W_35.JPG | 1024 × 1024 | 39 |
| DJI_0293_W_36.JPG | 1024 × 1024 | 40 |
| DJI_0293_W_37.JPG | 1024 × 1024 | 6 |
| DJI_0293_W_38.JPG | 1024 × 1024 | 0 |

Independent JSON validation confirmed five unique filenames, correct image hashes
and dimensions, 116 polygons, unique prediction IDs, valid confidence/class/model
identifiers, and finite coordinates within each original image's bounds.
Four exporter contract tests passed, including a 1200 × 300 image with coordinates
above the model input size, out-of-order results, empty predictions, label mapping,
overwrite protection, and rejection of duplicate/missing/out-of-bounds results.
The YOLO environment lacks pytest; tests ran in the existing prep environment
with inference mocked. Real inference ran only on the five images above.

Outputs are local under `outputs/class-20261108/` (ignored by Git). Existing
experiments, datasets, model weights, and predictions were not overwritten.
