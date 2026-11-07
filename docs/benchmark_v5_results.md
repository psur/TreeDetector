# benchmark_v5: final held-out test results

## Evaluation scope

This comparison reports final held-out **test-set instance-segmentation metrics only** for YOLO11m-seg and Mask2Former using `facebook/mask2former-swin-tiny-coco-instance`. Both models were trained/fine-tuned using exactly the same benchmark_v5 train, validation, and test split. Mask2Former used a COCO-pretrained checkpoint; these results should not be described as training both models from random initialization.

All AP and recall values are on a 0-1 scale. Overall category-averaged metrics are called **mAP50** and **mAP50-95**; individual-class metrics are called **AP50** and **AP50-95**. AP50-95 averages over IoU thresholds 0.50 through 0.95 in increments of 0.05. Mask2Former metrics use standard COCO segmentation evaluation; AR100 is average recall at a maximum of 100 detections per image.

## Dataset context

benchmark_v5 was generated after source auditing, content-based duplicate detection, human review of conflicting annotations, safe duplicate-vertex repair, and exclusion of remaining invalid polygon geometry.

| Dataset property | Final count |
|---|---:|
| Unique images | 262 |
| Train images | 183 |
| Validation images | 39 |
| Test images | 40 |
| tree instances, all splits | 3,170 |
| TreeGroup instances, all splits | 632 |
| All instances, all splits | 3,802 |
| Invalid polygons | 0 |

The test set contains **533 tree instances** and **85 TreeGroup instances**, for **618 instances** total. Final dataset validation found no duplicate content across splits and no shared image IDs across splits. benchmark_v4 remained unchanged.

## Test comparison

The absolute difference is Mask2Former minus YOLO11m-seg in metric units, not a relative percentage change. Differences are calculated from the supplied values; YOLO results are available here to three decimal places, so the extra digits in the differences do not imply greater YOLO measurement precision.

| Metric | YOLO11m-seg | Mask2Former | Absolute difference |
|---|---:|---:|---:|
| Overall mask mAP50 | 0.613 | 0.689694 | +0.076694 |
| Overall mask mAP50-95 | 0.321 | 0.388797 | +0.067797 |
| tree AP50 | 0.838 | 0.893579 | +0.055579 |
| tree AP50-95 | 0.449 | 0.485158 | +0.036158 |
| TreeGroup AP50 | 0.388 | 0.485810 | +0.097810 |
| TreeGroup AP50-95 | 0.193 | 0.292436 | +0.099436 |

## Additional Mask2Former test metrics

| Category | AP50 / overall mAP50 | AP50-95 / overall mAP50-95 | AP75 / overall mAP75 | AR100 |
|---|---:|---:|---:|---:|
| tree | 0.893579 | 0.485158 | 0.483046 | 0.616510 |
| TreeGroup | 0.485810 | 0.292436 | 0.327517 | 0.640000 |
| Overall | 0.689694 | 0.388797 | 0.405281 | 0.628255 |

YOLO AP75 and AR100 are not supplied and are not inferred.

## Why overall mAP differs from tree AP

COCO overall mAP50 is averaged across categories, rather than weighted by each category's instance count. Mask2Former overall mAP50 of approximately **0.690** is therefore consistent with tree AP50 of approximately **0.894**: the overall result also includes the lower TreeGroup AP50.

For the two categories, using the displayed rounded values:

```text
(0.893579 + 0.485810) / 2 = 0.6896945
```

This agrees with reported overall mAP50 approximately 0.690. The six-decimal overall value, 0.689694, was calculated from unrounded evaluation values; averaging already-rounded class values can differ in the last displayed digit.

Likewise:

```text
(0.485158 + 0.292436) / 2 = 0.388797
```

This is the reported overall mAP50-95, approximately **0.389**.

## Interpretation and limitations

Mask2Former outperformed YOLO11m-seg on the benchmark_v5 test set for both classes. The improvement was modest for individual tree crowns: approximately +0.056 AP50 and +0.036 AP50-95. The largest improvement occurred for TreeGroup objects: approximately +0.098 AP50 and +0.099 AP50-95. TreeGroup remained substantially more difficult than individual-tree segmentation for both models, as measured by AP.

These results indicate better performance of Mask2Former under this particular benchmark configuration. They do not establish statistical significance or demonstrate that the transformer architecture caused the difference. The comparison reflects the complete training and evaluation configurations, including pretrained initialization, rather than a controlled isolation of architecture effects.

TreeGroup annotations are unevenly distributed across source datasets/annotators:

| Original annotator/source | TreeGroup instances, all splits |
|---|---:|
| Jakub | 303 |
| Marek | 40 |
| Matej | 17 |
| Oliver | 272 |
| Total | 632 |

Source/domain composition should therefore be considered when interpreting per-class results. These counts describe annotation provenance; they do not establish that annotator identity caused differences in model performance. Absence of exact duplicate content also does not by itself establish geographic independence of UAV views.

## Result provenance and reproduction

Mask2Former run: `runs/mask2former_20260918_131806_706799`. Test-only reevaluation loaded its saved **best checkpoint**, evaluated the same 40 held-out images without retraining, and exactly reproduced the original aggregate metrics before rounding. The dataset and checkpoints were verified unchanged.

Saved per-class outputs:

- [JSON metrics](../runs/mask2former_20260918_131806_706799/per_class_test_metrics.json)
- [CSV metrics](../runs/mask2former_20260918_131806_706799/per_class_test_metrics.csv)

Run artifacts are local and may not be included in a Git checkout. YOLO values in this document are the supplied final held-out test results, retained at their reported precision.

To reproduce the Mask2Former test-only evaluation from the repository root in PowerShell:

```powershell
& "$env:LOCALAPPDATA\miniforge3\envs\tree_mask2former\python.exe" scripts/evaluate_mask2former_per_class.py --experiment runs/mask2former_20260918_131806_706799 --reference runs/mask2former_20260918_131806_706799/test_20260918_133515_034042/metrics.json
```
