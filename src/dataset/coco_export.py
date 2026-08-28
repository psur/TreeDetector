"""COCO instance-segmentation export from the fixed canonical manifest."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import pandas as pd
from src.config import project_path

SPLITS=("train","val","test")

def dataset_fingerprint(manifest: str|Path) -> str:
    frame=pd.read_csv(manifest,dtype=str).fillna("")
    required={"image_id","split","source_image","width","height"}
    if not required.issubset(frame.columns):raise ValueError(f"Manifest lacks columns: {sorted(required-set(frame.columns))}")
    rows=["\x1f".join(str(row[c]) for c in sorted(frame.columns)) for _,row in frame.sort_values("image_id").iterrows()]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()

def polygon_to_coco(points,width,height):
    if len(points)<3:raise ValueError("fewer than three vertices")
    clean=[]
    for x,y in points:
        x,y=float(x),float(y)
        if not math.isfinite(x) or not math.isfinite(y):raise ValueError("non-finite coordinate")
        if not (0<=x<=width and 0<=y<=height):raise ValueError("coordinate outside image")
        clean.append((x,y))
    area=abs(sum(x1*y2-x2*y1 for (x1,y1),(x2,y2) in zip(clean,clean[1:]+clean[:1])))/2
    xs=[p[0] for p in clean];ys=[p[1] for p in clean]
    box=[min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)]
    if area<=0 or box[2]<=0 or box[3]<=0:raise ValueError("zero-area polygon or bounding box")
    return [v for p in clean for v in p],box,area

def _read_yolo_polygons(path,width,height):
    result=[]
    for number,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip():continue
        cells=line.split(); cid=int(cells[0]); values=list(map(float,cells[1:]))
        if len(values)<6 or len(values)%2:raise ValueError(f"{path}, line {number}: not a segmentation polygon")
        yield number,cid,[(values[i]*width,values[i+1]*height) for i in range(0,len(values),2)]

def export_coco(cfg):
    root=project_path(cfg,cfg["dataset"]["output_dir"]); manifest=root/"dataset_manifest.csv"
    if not manifest.is_file():raise RuntimeError("Canonical manifest is missing; run 'python main.py prepare'")
    frame=pd.read_csv(manifest); out=root/"coco"/"annotations";out.mkdir(parents=True,exist_ok=True)
    issues=[]; fingerprint=dataset_fingerprint(manifest)
    for split in SPLITS:
        subset=frame[frame.split==split].sort_values("image_id"); images=[];annotations=[];aid=1
        for image_number,row in enumerate(subset.itertuples(),1):
            image_path=root/split/"images"/(str(row.image_id)+Path(row.source_image).suffix.lower())
            if not image_path.is_file():raise FileNotFoundError(f"Manifest image is missing: {image_path}")
            images.append({"id":image_number,"file_name":str(image_path.resolve()),"width":int(row.width),"height":int(row.height),"canonical_image_id":str(row.image_id)})
            label=root/split/"labels"/f"{row.image_id}.txt"
            if not label.is_file():raise FileNotFoundError(f"Canonical label is missing: {label}")
            for line,cid,points in _read_yolo_polygons(label,int(row.width),int(row.height)):
                try:seg,box,area=polygon_to_coco(points,int(row.width),int(row.height))
                except ValueError as exc:issues.append({"split":split,"image_id":row.image_id,"label":str(label),"line":line,"error":str(exc)});continue
                annotations.append({"id":aid,"image_id":image_number,"category_id":cid+1,"segmentation":[seg],"bbox":box,"area":area,"iscrowd":0});aid+=1
        expected=set(subset.image_id.astype(str)); actual={x["canonical_image_id"] for x in images}
        if actual!=expected:raise RuntimeError(f"COCO {split} image IDs do not match the canonical manifest")
        data={"info":{"dataset_fingerprint":fingerprint,"split":split},"images":images,"annotations":annotations,"categories":[{"id":cid+1,"name":name} for cid,name in sorted(cfg["classes"].items())]}
        (out/f"instances_{split}.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    report=project_path(cfg,"results")/"coco_validation.csv";pd.DataFrame(issues,columns=["split","image_id","label","line","error"]).to_csv(report,index=False)
    if issues:raise ValueError(f"COCO export rejected {len(issues)} annotation(s); see {report}")
    (root/"coco"/"dataset_fingerprint.txt").write_text(fingerprint+"\n",encoding="ascii")
    return {"dataset_fingerprint":fingerprint,"annotations_dir":str(out),"images":len(frame)}
