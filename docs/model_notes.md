# Model notes

## YOLO

Ultralytics YOLO is the one-stage instance-segmentation baseline. It fine-tunes the configured `yolo11m-seg.pt` checkpoint and reports Ultralytics native box and mask metrics.

## Mask R-CNN

The second baseline is **generic Detectron2 Mask R-CNN**, not the Detectree2 package. It uses the two-stage `mask_rcnn_R_50_FPN_3x` architecture, ResNet-50-FPN backbone, and Detectron2's COCO-pretrained model-zoo checkpoint. It trains on COCO files generated from the same canonical manifest and reports COCOEvaluator metrics.
