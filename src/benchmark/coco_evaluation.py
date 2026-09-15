"""Standard COCO mask evaluation; AP values use the 0..1 scale."""
import copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


def prediction_records(processor, outputs, image_ids, sizes, threshold=0.0):
    # CPU postprocessing keeps full-resolution binary maps off GPU and preserves overlaps.
    from types import SimpleNamespace
    output = SimpleNamespace(class_queries_logits=outputs.class_queries_logits.detach().cpu(),
                             masks_queries_logits=outputs.masks_queries_logits.detach().cpu())
    predictions = processor.post_process_instance_segmentation(
        output, target_sizes=sizes, threshold=threshold, return_binary_maps=True)
    records = []
    for image_id, result in zip(image_ids, predictions):
        for index, segment in enumerate(result["segments_info"]):
            plane = np.asfortranarray(result["segmentation"][index].numpy().astype(np.uint8))
            rle = mask_utils.encode(plane)
            rle["counts"] = rle["counts"].decode("ascii")
            records.append({"image_id": image_id, "category_id": segment["label_id"] + 1,
                            "score": float(segment["score"]), "segmentation": rle})
    return records


def evaluate_coco(annotation_path, predictions, image_ids=None):
    truth = COCO(str(annotation_path))
    if predictions:
        detected = truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = {"images": copy.deepcopy(truth.dataset["images"]),
                            "categories": copy.deepcopy(truth.dataset["categories"]), "annotations": []}
        detected.createIndex()
    evaluator = COCOeval(truth, detected, "segm")
    evaluator.params.imgIds = sorted(image_ids if image_ids is not None else truth.getImgIds())
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"]  # IoU, recall, category, area, maxDets
    def average(values):
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None
    per_class = {}
    for index, category in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        per_class[truth.cats[category]["name"]] = {
            "map50_mask": average(values[0]), "map50_95_mask": average(values)}
    return {"precision_mask": None, "recall_mask": None,
            "map50_mask": average(precision[0, :, :, 0, -1]),
            "map75_mask": average(precision[5, :, :, 0, -1]),
            "map50_95_mask": average(precision[:, :, :, 0, -1]),
            "coco_ar100_mask": float(evaluator.stats[8]) if evaluator.stats[8] >= 0 else None,
            "per_class": per_class, "predicted_instances": len(predictions),
            "metric_source": "pycocotools COCOeval segm; maxDets=100; AP scale 0..1",
            "evaluation_image_ids": evaluator.params.imgIds,
            "precision_recall_note": "Single-threshold precision/recall not computed; COCO AR100 reported separately"}
