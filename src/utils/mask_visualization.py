"""Contour-only instance visualization from COCO predictions."""
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mask_utils


def draw_predictions(image_path, predictions, destination, classes, threshold=0.25):
    # imdecode/imencode support Windows paths containing non-ASCII characters.
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    labels = []
    for prediction in sorted(predictions, key=lambda p: p["score"], reverse=True):
        if prediction["score"] < threshold:
            continue
        mask = mask_utils.decode(prediction["segmentation"]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (0, 230, 0) if prediction["category_id"] == 1 else (0, 180, 255)
        cv2.drawContours(image, contours, -1, color, 2)
        if not contours:
            continue
        x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
        label = f"{classes[prediction['category_id']-1]} {prediction['score']:.2f}"
        labels.append((x, y, label, color))
    # Draw text after all contours; move colliding labels within image bounds.
    occupied = []
    for x, y, label, color in labels:
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        width, height = min(text_width+4, image.shape[1]), text_height+8
        x = max(0, min(x, image.shape[1]-width))
        y = max(0, min(y-height, image.shape[0]-height))
        chosen = None
        for distance in range(0, image.shape[0], height+2):
            for candidate_y in (y+distance, y-distance):
                if candidate_y < 0 or candidate_y+height > image.shape[0]:
                    continue
                for candidate_x in (x, max(0,x-width-4), min(image.shape[1]-width,x+width+4)):
                    box = (candidate_x, candidate_y, candidate_x+width, candidate_y+height)
                    if not any(box[0] < b[2]+2 and box[2]+2 > b[0] and box[1] < b[3]+2 and box[3]+2 > b[1] for b in occupied):
                        chosen = box
                        break
                if chosen:
                    break
            if chosen:
                break
        if chosen is None:
            chosen = (x,y,x+width,y+height)
        occupied.append(chosen)
        left, top, right, bottom = chosen
        cv2.rectangle(image, (left,top), (right-1,bottom-1), (0,0,0), -1)
        cv2.putText(image, label, (left+2,bottom-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(destination.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode visualization: {destination}")
    encoded.tofile(str(destination))
