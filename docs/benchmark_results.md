# Completed YOLO benchmark

Run: `runs/yolo_seg_20260912_193929_824528`  
Model: YOLO11m-seg  
Canonical dataset: `datasets/benchmark_v4`  
Test images: 25; test instances: 492.

Headline metrics supplied by the experiment owner (rounded, Ultralytics native evaluator):

| Metric | Box | Mask |
| --- | ---: | ---: |
| Precision | 0.691 | 0.679 |
| Recall | 0.660 | 0.663 |
| mAP50 | 0.735 | 0.719 |
| mAP50-95 | 0.420 | 0.365 |

| Class mask | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| tree | 0.820 | 0.425 |
| TreeGroup | 0.618 | 0.304 |

Weights, predictions, run configuration, and large outputs remain local and unchanged. These native scores must retain their evaluator label; a future common COCO evaluation is a separate result, not a replacement for this baseline.
