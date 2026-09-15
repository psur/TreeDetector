# Mask2Former implementation and smoke report

Date: 2026-09-15. Implementation PASS; GPU smoke PASS. Full training has not run.

Run: `runs/mask2former_20260915_192624_459002`.

Checkpoint: facebook/mask2former-swin-tiny-coco-instance, revision `22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6`.

| Split | Images | Instances | tree | TreeGroup |
| --- | ---: | ---: | ---: | ---: |
| train | 115 | 2063 | 1649 | 414 |
| val | 25 | 446 | 388 | 58 |
| test | 25 | 492 | 395 | 97 |

Exact COCO paths:

- `D:/TreeDetector/datasets/benchmark_v4/coco/annotations/instances_train.json`
- `D:/TreeDetector/datasets/benchmark_v4/coco/annotations/instances_val.json`
- `D:/TreeDetector/datasets/benchmark_v4/coco/annotations/instances_test.json`

The audit found valid polygon geometry and nonempty masks, matching manifest counts/IDs/dimensions, all images present, and no duplicated decoded image pixels or canonical/source identities across splits. Spatial overlap of distinct views was not assessed. Images and annotations were reused without conversion or split changes.

Smoke: pretrained imports/loading, two-class head, one crowded training image (COCO train ID 105), forward, backward, clipped gradient, AdamW update, one validation image, COCO metrics, checkpoint save and reload. Input tensor [1,3,512,512], mask targets [56,512,512], class targets [56]. Loss was finite at 74.919868; pre-clipping gradient norm was 155.083710. These are startup diagnostics, not trained-model quality estimates. Reload/resave verified all 556 checkpoint tensors unchanged. A contour visualization was produced from the existing validation predictions.

GPU: NVIDIA GeForce GTX 1080 Ti, 11 GiB, compute capability 6.1. Peak allocated 2.113129 GiB; peak reserved 2.513672 GiB. Peaks describe PyTorch allocator usage, not total system GPU usage.

Environment: Python 3.11.16, torch 2.14.0+cu126, torchvision 0.29.0+cu126, Transformers 5.17.0, pycocotools 2.0.11, SciPy 1.17.1, NumPy 2.4.6, Pillow 12.3.0, pandas 3.0.5, OpenCV 5.0.0.93. SciPy was the only added environment package. Exact checkpoint downloads are cached in the ignored repository .cache directory.

Tests cover mask overlaps/empty targets, polygon/RLE decoding, overlapping prediction export, perfect/empty COCO AP, absent categories, contour transparency, run destination protection, gradient accumulation, validation-selected best weights, resume state, CLI options, and unchanged YOLO logging behavior. Standard-library unittest tests require no pytest installation; optional Mask2Former tests skip in environments lacking that stack.

Detailed smoke results: `smoke_test.json`; tensor reload check: `checkpoint_reload_check.json`; diagnostic visualization: `smoke_validation/contours_readable.png`. All are inside the run. Smoke metrics were not appended to the benchmark CSV.

First real benchmark command (not executed):

```powershell
conda activate tree_mask2former
python main.py --config config/config.yaml benchmark --model mask2former
```

Preservation verification: all 519 pre-existing files in benchmark_v4, runs, results and data_staging retain their SHA-256 hashes. Both configs retain every previous setting and only add models.mask2former. Raw Dropbox sources were never written; YOLO implementation files remain unchanged. The ignored verification baseline is results/mask2former_preservation_before.json.
