"""Read-only discovery and format identification."""
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import json
from .annotations import parse_yolo,parse_coco,parse_labelme
EXTS={".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp"}
@dataclass
class ScanResult: annotations:list; format:str; missing_annotations:list; orphan_annotations:list; unreadable:list; duplicate_names:list
def scan_dataset(images_dir,annotations_dir,classes):
    if not images_dir.is_dir() or "CHANGE_ME" in str(images_dir): raise FileNotFoundError(f"Set input.images_dir to an existing directory (currently {images_dir})")
    if not annotations_dir.is_dir() or "CHANGE_ME" in str(annotations_dir): raise FileNotFoundError(f"Set input.annotations_dir to an existing directory (currently {annotations_dir})")
    images=sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in EXTS); js=sorted(annotations_dir.rglob("*.json")); txt=sorted(annotations_dir.rglob("*.txt"))
    if js and txt: raise ValueError("Both JSON and YOLO TXT detected; use one annotation format")
    if not js and not txt: raise ValueError("No annotations found; expected YOLO TXT or COCO JSON")
    counts={}; readable=[]; dims={}; unreadable=[]
    for p in images: counts[p.name.casefold()]=counts.get(p.name.casefold(),0)+1
    for p in images:
        try:
            with Image.open(p) as im: im.verify()
            with Image.open(p) as im: dims[p]=im.size
            readable.append(p)
        except Exception as exc: unreadable.append((p,str(exc)))
    duplicates=[p for p in images if counts[p.name.casefold()]>1]
    if js:
        try: sample=json.loads(js[0].read_text(encoding="utf-8-sig"))
        except (OSError,json.JSONDecodeError) as exc: raise ValueError(f"Unreadable JSON annotation {js[0]}: {exc}") from exc
        is_labelme="shapes" in sample and "imagePath" in sample
        if is_labelme:
            labels={p.stem.casefold():p for p in js}; pics={p.stem.casefold():p for p in readable}
            parsed=[parse_labelme(labels[p.stem.casefold()],p,*dims[p],classes) for p in readable if p.stem.casefold() in labels]
            return ScanResult(parsed,"labelme",[p for k,p in pics.items() if k not in labels],[p for k,p in labels.items() if k not in pics],unreadable,duplicates)
        if len(js)>1: raise ValueError("Multiple non-LabelMe JSON files detected; expected one COCO JSON")
        parsed=parse_coco(js[0],readable,classes); covered={a.image_path.resolve() for a in parsed}
        return ScanResult(parsed,"coco",[p for p in readable if p.resolve() not in covered],[],unreadable,duplicates)
    lengths={len(line.split()) for p in txt[:100] for line in p.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    if lengths and lengths=={5}: fmt="yolo_detection"
    elif not lengths or all(n>=7 and (n-1)%2==0 for n in lengths): fmt="yolo_segmentation"
    else: raise ValueError(f"Unrecognized/mixed YOLO rows: {sorted(lengths)} fields")
    labels={p.stem.casefold():p for p in txt}; pics={p.stem.casefold():p for p in readable}
    parsed=[parse_yolo(labels[p.stem.casefold()],p,*dims[p],classes,fmt=="yolo_segmentation") for p in readable if p.stem.casefold() in labels]
    return ScanResult(parsed,fmt,[p for k,p in pics.items() if k not in labels],[p for k,p in labels.items() if k not in pics],unreadable,duplicates)
