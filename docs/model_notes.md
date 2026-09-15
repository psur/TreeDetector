# Mask2Former benchmark

## Implementation and environment

The active backends are YOLO and Hugging Face Mask2Former. Detectron2 is not a dependency. Use the separate Python 3.11 `tree_mask2former` environment. Tested packages and GPU are in [the smoke report](mask2former_smoke_report.md); requirements-mask2former.txt records the model stack. Keep the current CUDA 12.6 PyTorch build supporting Pascal sm_61 when reproducing this GTX 1080 Ti run; do not replace it with an arbitrary CUDA wheel.

The checkpoint is [facebook/mask2former-swin-tiny-coco-instance](https://huggingface.co/facebook/mask2former-swin-tiny-coco-instance), pinned to revision `22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6`. The [Hugging Face Mask2Former API](https://huggingface.co/docs/transformers/model_doc/mask2former) provides the image processor, universal segmentation model and overlapping binary instance postprocessing. The classifier and no-object loss weights are resized from 80 to two foreground classes. Transformers 5.17 reports newly initialized Swin final layernorm parameters and obsolete relative-position-index buffers on loading this older checkpoint; initialization is seeded. Checkpoint reload was verified tensor-for-tensor.

## Data and preprocessing

Reuse `datasets/benchmark_v4` directly. All COCO files are under `coco/annotations/instances_{train,val,test}.json`. The read-only audit verifies exact split counts, canonical IDs, source identities, paths, decoded pixel duplicates, dimensions, category mapping, polygon bounds/area, decoded masks, per-image object counts and manifest fingerprint. No split generation or annotation conversion is performed. COCO numeric image IDs are split-local; canonical IDs identify images globally.

COCO categories 1=tree and 2=TreeGroup map to model labels 0/1 and are inverted for predictions. Each polygon/RLE annotation is independently decoded into a binary plane, so overlaps and multiple same-class instances remain intact. Empty images have zero mask planes. RGB images pass through AutoImageProcessor for fixed square resize, normalization and padding; each binary mask is resized independently with nearest-neighbor interpolation and padded identically. The model retains 100 queries; the maximum ground-truth count is 56. The initial 512 resolution is divisible by the backbone stride, avoiding extra padding. Fixed square resizing is appropriate for these square tiles; another aspect ratio would be stretched under this documented policy.

`smoke` prints per-split counts, per-class totals and actual preprocessed tensor dimensions. Dataset validation is repeated on run opening, and the run identity binds configuration, manifest fingerprint and annotation SHA-256 hashes. Different configuration/dataset/purpose requires a new experiment. Spatial overlap between distinct UAV views is not proven absent; no identical decoded images or shared canonical IDs were found across splits.

## Training and CLI

```powershell
conda activate tree_mask2former
python main.py --config config/config.yaml smoke --model mask2former
python main.py --config config/config.yaml benchmark --model mask2former
# Training alone, or epoch-boundary resume of an interrupted real run:
python main.py --config config/config.yaml train --model mask2former --experiment runs/mask2former_TIMESTAMP
python main.py --config config/config.yaml evaluate --model mask2former --experiment runs/mask2former_TIMESTAMP
python main.py --config config/config.yaml predict --model mask2former --experiment runs/mask2former_TIMESTAMP
python main.py --config config/config.yaml predict --model mask2former --experiment runs/mask2former_TIMESTAMP --source D:/images/example.jpg
python main.py compare
```

Use a new default run for the first benchmark. Smoke runs are explicitly marked and cannot be resumed as benchmark training. There is no `all` backend. YOLO commands remain supported in its own environment.

Defaults: 512 pixels, batch 1, gradient accumulation 4 (effective batch 4 except the final short group), float32, AdamW learning rate 0.00005, weight decay 0.01, gradient clipping 1.0, seed 42, at most 30 epochs, early stopping after 7 epochs without validation AP improvement. No random augmentation or LR scheduler is enabled initially. DataLoader workers=0 supports Windows reliably. Epoch order is seeded; Python/NumPy/PyTorch/CUDA RNG states and optimizer state are saved for epoch-boundary resume. An interrupted partial epoch repeats from the last completed checkpoint. GPU kernels may still prevent bitwise cross-platform reproducibility.

Every epoch evaluates validation only and saves the best mask AP50-95 checkpoint. A completed benchmark tests that selected best checkpoint, not the last epoch. Training settings may differ from YOLO; dataset and held-out split must match. Do not tune hyperparameters on test. The implementation smoke test ran one crowded training image and one validation image only; no test-set evaluation or full training was run.

## Metrics and visualizations

Evaluation saves scored overlapping binary instance predictions as COCO RLE at original image dimensions. It uses standard `pycocotools.COCOeval` with `iouType=segm`, maxDets=[1,10,100], standard area ranges and IoU 0.50:0.05:0.95. AP is reported on the 0..1 scale overall and per class. Undefined per-class AP is null. No-detection predictions are supported. Evaluation keeps predictions at threshold 0.0; the visualization confidence 0.25 is separate. Installed HF postprocessing uses its own 384-pixel intermediate before restoring original dimensions; this is recorded as part of the versioned evaluation policy.

Single-threshold precision/recall are null because COCO does not define the same native operating-point summaries as Ultralytics. COCO AR100 is reported separately. Box scores are not manufactured from masks. YOLO's completed Ultralytics native AP and these COCOeval scores retain explicit evaluator labels; rigorous common-evaluator comparison would require a future separately stored COCO evaluation of YOLO, leaving its original run untouched.

Prediction uses the same sorted test-file sampling approach, seed 42 and configured count (20) as YOLO. Each call saves its selected filenames and contour-only PNGs, readable class labels and scores, without opaque image-covering fills. Arbitrary source images are also supported. Colors are green for tree and orange for TreeGroup.

## Output and reproducibility

Each experiment is isolated under `runs/mask2former_TIMESTAMP/`:

- `run_identity.json`, `config.yaml`, `dataset_audit.json`, `environment.json` record identity and provenance.
- `training_history.json` records epoch loss, validation AP and elapsed times.
- `best/` contains model and processor; `best_validation_metrics.json` identifies its epoch.
- `last/` contains model/processor plus optimizer and RNG resume state. Only load trusted local resume files.
- `validation_epoch_*` and `test_*` contain predictions, overall/per-class metrics and image IDs.
- `predictions_*` contains selected-image list and contour visualizations.
- `experiment.json` and the shared `results/benchmark_results.csv` receive benchmark summaries through the existing runner after test evaluation.

Epoch time includes validation. `forward_seconds` synchronizes CUDA and excludes loading/postprocessing, but has no warmup; `inference_seconds` includes the evaluation pipeline. These are not directly equivalent to YOLO's timing. Record resolution differences (YOLO 640 versus initial Mask2Former 512), backbone, pretraining, optimizer, budget, hardware and evaluator when reporting results. Existing YOLO artifacts and CSV were not changed by the smoke test.

Model caches and run artifacts are excluded from Git. Full training and its long-run memory/convergence behavior remain untested; the smoke establishes one-batch functionality, not model quality.
