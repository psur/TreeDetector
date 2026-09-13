from pathlib import Path
import cv2
from ultralytics import YOLO

model = YOLO(r"D:\TreeDetector\runs\yolo_seg_20260912_193929_824528\training\weights\best.pt")

source_dir = Path(r"D:\TreeDetector\datasets\benchmark_v4\test\images")
out_dir = Path(r"D:\TreeDetector\runs\predictions_mask_contours")
out_dir.mkdir(parents=True, exist_ok=True)

results = model.predict(
    source=str(source_dir),
    save=False,
    conf=0.25,
    retina_masks=True
)

for result in results:
    img = result.orig_img.copy()

    names = result.names
    boxes = result.boxes
    masks = result.masks

    if boxes is not None:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = f"{names[cls_id]} {conf:.2f}"

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # blue boxes
            color = (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if masks is not None and masks.xy is not None:
        for poly in masks.xy:
            pts = poly.astype("int32").reshape((-1, 1, 2))
            # green contour
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    out_path = out_dir / Path(result.path).name
    cv2.imwrite(str(out_path), img)

print(f"Saved contour visualizations to: {out_dir}")