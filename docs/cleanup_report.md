# Detectron2 cleanup report

## Git safety

Initial branch: main, tracked files clean at 7aa376a. Untracked: `.venv-prep/`, `.venv-yolo/`, `Python`, `Visual`, and `src/utils/predict_mask_contours.py`. The latest contour source and machine-local configuration were not committed. Safety commit `94c10ab8` (Save YOLO benchmark and pre-Mask2Former state) saves the contour script and reviewed `docs/pre_mask2former_config.yaml` snapshot. The snapshot contains local dataset paths, no credentials; it is historical, not active configuration.

GitHub origin: https://github.com/psur/TreeDetector.git. Live read-only check found main at 7aa376a1054d1502635946b817a5a48e7c5e12df, so local main is ahead 1, behind 0. Nothing was pushed. Cleanup changes are deliberately uncommitted for review.

Ignored content includes config/config.yaml, datasets/benchmark_v4, data_staging, runs (completed YOLO and contour outputs), results (audit reports, CSVs, plots and historical config), weights, root .pt checkpoints, bytecode, and installed compiled packages. Added `.venv-*/` to ignore both local environments completely. No datasets, environments, weights, caches or large outputs entered the safety commit.

## Occurrence classification and outcome

| Class | Paths / occurrences | Decision |
| --- | --- | --- |
| A: remove | src/models/detectron2_model.py; requirements-detectron2.txt | Removed; recoverable from Git |
| A: remove | Detectron2 setup and command sections in README; both active config Detectree2 blocks | Removed obsolete instructions and settings |
| B: shared | src/dataset/coco_export.py, pipeline, polygon/fingerprint tests; datasets/benchmark_v4/coco | Retained unchanged; COCO is backend independent |
| C: history | docs/pre_mask2former_config.yaml; Git history | Retained for recovery/reference |
| C: history | runs/yolo_seg_20260912_193929_824528/config.yaml; results/config.before_deduplication.yaml | Retained byte-for-byte; historical Detectree2 settings are inactive |
| C: reference | root Python and Visual | Retained; tiny shell/toolchain notes of uncertain provenance, unrelated to active backend |
| D: refactor | main.py; src/models/registry.py; tests/test_milestone2.py | CLI and registry now expose only YOLO |
| D: refactor | src/benchmark/runner.py; base_model.py; yolo_model.py | Backend metadata supplied by adapter; removed implicit Detectron2 fallback and all dispatch |
| D: documentation | README.md; docs/model_notes.md | Updated status and future implementation plan |

The project content search included ignored data/run/config text and hidden files, excluding Git internals and third-party environments. A separate repository filename/directory scan found no local Detectron2 clone, build/dist/egg-info directory or compiled pyd/dll/obj residue outside the virtual environments. Environment binaries and general bytecode caches are retained; there is no identified failed-build artifact to delete. No Detectron2 run directory exists.

External environment confirmed at `C:/Users/surovy/AppData/Local/miniforge3/envs/tree_detectron`; untouched. Optional later removal, never executed here:

```powershell
conda env remove -n tree_detectron
```

## Preservation and validation

All 518 pre-existing files under benchmark_v4, runs, results and data_staging retain their SHA-256 hashes. The local ignored verification manifest is results/cleanup_preservation_before.json. Raw Dropbox sources were never written or scanned for modification. No preparation, training, prediction or evaluation was run against the protected data/run. Canonical COCO has 115/25/25 images and 2063/446/492 instances, with all image paths resolving.

Both active configuration files were compared structurally to their pre-cleanup versions: only models.detectree2 and detectree2_training were removed. Inputs, YOLO, classes, seeds, splits, thresholds and visualization settings are identical. No speculative Mask2Former configuration was added.

Two standard-library regression tests pass (all four CLI actions reject unavailable backends; YOLO logging retains metadata and existing CSV rows). Python syntax and git diff whitespace checks pass. Full pytest suite could not run because pytest is absent from both local environments; no dependencies were installed. YOLO training/inference was not rerun.

See benchmark_results.md for the completed metrics and model_notes.md for the eight-step Mask2Former implementation plan and reuse limitations. Repository ready for implementation: yes. Mask2Former executable: no, intentionally.
