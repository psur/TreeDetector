# TreeDetector

TreeDetector prepares a fixed UAV tree-crown dataset and benchmarks instance segmentation. YOLO11m-seg is completed and working. The Detectron2 / Mask R-CNN benchmark was abandoned because of legacy build/toolchain complexity. Mask2Former is the next planned comparison; it is not implemented or exposed by the CLI.

## Dataset and configuration

`datasets/benchmark_v4` is the canonical shared dataset: 115 train, 25 validation, and 25 test images. Preserve its manifest, existing assignments, and COCO annotations under `coco/annotations/instances_{train,val,test}.json`; these are reusable for transformer segmentation. Source Dropbox annotations remain read-only.

`config/config.yaml` is machine-local and ignored. On a new checkout copy `config/config.example.yaml` to that path and configure source paths. Do not overwrite an existing local config. Keep the existing benchmark dataset when reproducing the comparison; do not rebuild its split. The current local batch is 4; the portable example retains its original batch of 8.

## Environments and commands

Use separate environments for YOLO and future Mask2Former work. The existing `.venv-yolo` remains unchanged. For a new YOLO environment install a compatible PyTorch build and `requirements-yolo.txt`; test dependencies are in `requirements-dev.txt`. Shared data dependencies are in `requirements.txt`.

```powershell
python main.py --help
python main.py info
python -m pytest
python main.py train --model yolo
python main.py benchmark --model yolo
python main.py evaluate --model yolo --experiment runs/NEW_OR_COPIED_RUN
python main.py predict --model yolo --experiment runs/NEW_OR_COPIED_RUN --source D:/images/example.jpg
python main.py compare
```

Evaluate/predict write into their experiment directory. Use a separate copy when exploring the completed benchmark. Training with an existing experiment resumes its newest `last.pt`. `prepare` validates and exports source annotations and reuses existing manifest assignments; it is unnecessary for the preserved benchmark.

## Architecture and results

`src/dataset/` handles scanning, validation, splitting, and YOLO/COCO export. `src/models/` contains the backend contract, registry, and YOLO adapter. Only implemented backends are registered. Adapters provide result metadata; `src/benchmark/runner.py` writes experiment JSON and the shared benchmark CSV. No placeholder Mask2Former implementation or speculative hyperparameters are included.

The completed run `runs/yolo_seg_20260912_193929_824528` and contour outputs `runs/predictions_mask_contours` are preserved outside Git. See [benchmark results](docs/benchmark_results.md), [Mask2Former plan](docs/model_notes.md), and [cleanup report](docs/cleanup_report.md).

Dataset validation accepts YOLO TXT, LabelMe polygons, and polygon COCO JSON. Missing annotations and invalid geometry are reported; source COCO RLE decoding is not implemented. Existing exclusions and seed 42 remain unchanged. Random image splits can contain overlapping UAV views; preserve this benchmark split for comparison and document that limitation.
