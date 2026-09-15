"""Read-only COCO adapter. Every annotation keeps its own binary mask."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from torch.utils.data import Dataset
from src.dataset.coco_export import dataset_fingerprint, polygon_to_coco


def decode_mask(annotation, height, width):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return mask_utils.decode(rle).astype(np.uint8)


class CocoInstanceDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root).resolve()
        self.split = split
        self.annotation_path = self.root / "coco" / "annotations" / f"instances_{split}.json"
        self.data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.images = sorted(self.data["images"], key=lambda image: image["id"])
        self.annotations = {image["id"]: [] for image in self.images}
        for annotation in self.data["annotations"]:
            if annotation["image_id"] not in self.annotations:
                raise ValueError("Annotation references an unknown image ID")
            self.annotations[annotation["image_id"]].append(annotation)
        self.category_to_label = {1: 0, 2: 1}
        if {c["id"]: c["name"] for c in self.data["categories"]} != {1: "tree", 2: "TreeGroup"}:
            raise ValueError("Expected COCO categories 1=tree, 2=TreeGroup")

    def image_path(self, record):
        # Use canonical local images, permitting relocation without rewriting COCO JSON.
        name = Path(record["file_name"].replace("\\", "/")).name
        return self.root / self.split / "images" / name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        record = self.images[index]
        with Image.open(self.image_path(record)) as source:
            image = source.convert("RGB")
        annotations = self.annotations[record["id"]]
        masks = [decode_mask(a, record["height"], record["width"]) for a in annotations]
        return {"image": image, "masks": masks,
                "labels": [self.category_to_label[a["category_id"]] for a in annotations],
                "image_id": record["id"], "size": (record["height"], record["width"])}

    def summary(self):
        counts = Counter(a["category_id"] for a in self.data["annotations"])
        return {"images": len(self), "instances": sum(counts.values()),
                "per_class": {c["name"]: counts[c["id"]] for c in self.data["categories"]},
                "max_instances": max((len(a) for a in self.annotations.values()), default=0),
                "annotation_path": str(self.annotation_path)}


def validate_dataset(root, expected_counts=None):
    root = Path(root)
    manifest = pd.read_csv(root / "dataset_manifest.csv", dtype={"image_id": str})
    fingerprint = dataset_fingerprint(root / "dataset_manifest.csv")
    if manifest.image_id.duplicated().any() or manifest.source_image.duplicated().any():
        raise ValueError("Duplicate manifest image identity")
    if set(manifest.split) != {"train", "val", "test"}:
        raise ValueError("Unexpected manifest splits")
    expected_counts = expected_counts or {"train": 115, "val": 25, "test": 25}
    seen_ids, seen_pixels = set(), {}
    report = {"dataset_fingerprint": fingerprint, "splits": {}}
    for split in ("train", "val", "test"):
        dataset = CocoInstanceDataset(root, split)
        rows = manifest[manifest.split == split].set_index("image_id")
        ids = [i["canonical_image_id"] for i in dataset.images]
        if len(dataset) != expected_counts[split] or set(ids) != set(rows.index) or len(ids) != len(set(ids)):
            raise ValueError(f"COCO/manifest count or identity mismatch: {split}")
        if seen_ids.intersection(ids):
            raise ValueError("Cross-split canonical ID leakage")
        seen_ids.update(ids)
        if dataset.data["info"]["dataset_fingerprint"] != fingerprint:
            raise ValueError("COCO fingerprint differs from manifest")
        if len(dataset.annotations) != len(dataset.images):
            raise ValueError("Duplicate COCO image IDs")
        annotation_ids = [a["id"] for a in dataset.data["annotations"]]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("Duplicate annotation IDs")
        actual_files = {p.resolve() for p in (root / split / "images").iterdir() if p.is_file()}
        if actual_files != {dataset.image_path(i).resolve() for i in dataset.images}:
            raise ValueError("Missing or extra split images")
        for i in dataset.images:
            if dataset.image_path(i).stem != i["canonical_image_id"]:
                raise ValueError("COCO filename differs from canonical image identity")
            row = rows.loc[i["canonical_image_id"]]
            if (i["width"], i["height"]) != (int(row.width), int(row.height)):
                raise ValueError("Manifest dimensions differ")
            with Image.open(dataset.image_path(i)) as image:
                if image.size != (i["width"], i["height"]):
                    raise ValueError("Image dimensions differ")
                digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            if digest in seen_pixels:
                raise ValueError(f"Duplicate decoded image: {split}, {seen_pixels[digest]}")
            seen_pixels[digest] = split
            annotations = dataset.annotations[i["id"]]
            if len(annotations) != int(row.number_of_objects):
                raise ValueError("Instance count differs from manifest")
            for a in annotations:
                if a["category_id"] not in dataset.category_to_label or a.get("iscrowd", 0):
                    raise ValueError("Unknown category or unsupported crowd training target")
                if isinstance(a["segmentation"], list):
                    for polygon in a["segmentation"]:
                        if len(polygon) % 2:
                            raise ValueError("Odd polygon coordinate count")
                        polygon_to_coco(list(zip(polygon[::2], polygon[1::2])), i["width"], i["height"])
                mask = decode_mask(a, i["height"], i["width"])
                if mask.shape != (i["height"], i["width"]) or not mask.any():
                    raise ValueError("Empty or invalid decoded mask")
        report["splits"][split] = dataset.summary()
    report["leakage_check"] = "No shared IDs, source paths or identical decoded images; spatial overlap not assessed"
    return report


class InstanceCollator:
    def __init__(self, processor, image_size):
        self.processor = processor
        self.image_size = image_size

    def __call__(self, samples):
        # Fixed square resize: HF normalizes/resizes/pads RGB; independently resized
        # binary planes avoid losing overlapping instances in a single ID raster.
        batch = self.processor(images=[s["image"] for s in samples],
                               size={"height": self.image_size, "width": self.image_size},
                               return_tensors="pt")
        height, width = batch["pixel_values"].shape[-2:]
        masks, labels = [], []
        for sample in samples:
            planes = [torch.from_numpy(np.array(Image.fromarray(m).resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST), copy=True)).float()
                for m in sample["masks"]]
            target = torch.stack(planes) if planes else torch.zeros((0, self.image_size, self.image_size))
            target = torch.nn.functional.pad(target, (0, width-self.image_size, 0, height-self.image_size))
            masks.append(target)
            labels.append(torch.tensor(sample["labels"], dtype=torch.long))
        batch["mask_labels"], batch["class_labels"] = masks, labels
        return {"inputs": dict(batch), "image_ids": [s["image_id"] for s in samples],
                "sizes": [s["size"] for s in samples]}
