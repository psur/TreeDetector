"""Validation, fixed splitting, export, and statistics."""
from pathlib import Path
import hashlib,json,logging,shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,yaml
from sklearn.model_selection import train_test_split
from src.config import project_path
from .scanner import ScanResult,scan_dataset
LOG=logging.getLogger(__name__)
def _issues(scan,classes):
    rows=[]
    def add(status,path,message): rows.append({"status":status,"path":str(path),"message":message})
    for p,e in scan.unreadable:add("ERROR_UNREADABLE_IMAGE",p,e)
    for p in scan.missing_annotations:add("ERROR_MISSING_ANNOTATION",p,"Image has no annotation")
    for p in scan.orphan_annotations:add("ERROR_ANNOTATION_WITHOUT_IMAGE",p,"Annotation has no matching image")
    for p in scan.duplicate_names:add("ERROR_DUPLICATE_FILENAME",p,"Filename is not unique")
    for a in scan.annotations:
        if not a.objects:add("VALID_EMPTY_IMAGE",a.image_path,"Legitimate empty annotation")
        for i,o in enumerate(a.objects):
            x1,y1,x2,y2=o.bounding_box; prefix=f"object {i}: "
            if o.class_id not in classes:add("ERROR_INVALID_CLASS",a.annotation_path,prefix+f"unknown label '{o.class_name}'")
            if not all(np.isfinite([x1,y1,x2,y2])) or x2<=x1 or y2<=y1:add("ERROR_ZERO_OR_INVALID_BOX",a.annotation_path,prefix+str(o.bounding_box))
            if x1<0 or y1<0 or x2>a.width or y2>a.height:add("ERROR_OUT_OF_BOUNDS",a.annotation_path,prefix+str(o.bounding_box))
            if o.polygon is not None and len(o.polygon)<3:add("ERROR_SHORT_POLYGON",a.annotation_path,prefix+"fewer than 3 points")
        if a.objects and not any(r["path"]==str(a.annotation_path) and r["status"].startswith("ERROR") for r in rows):add("VALID",a.image_path,"")
    return rows
def _split(ids,cfg):
    ds=cfg["dataset"]; fractions=[float(ds[k]) for k in ("train_fraction","val_fraction","test_fraction")]
    if ds["split_mode"]!="random":raise NotImplementedError("Milestone 1 supports random splitting; grouped/spatial/flight are extension points")
    if not np.isclose(sum(fractions),1):raise ValueError("Split fractions must sum to 1")
    if len(ids)<3:raise ValueError("At least 3 images are required for train/val/test")
    train,temp=train_test_split(ids,test_size=sum(fractions[1:]),random_state=int(ds["random_seed"])); val,test=train_test_split(temp,test_size=fractions[2]/sum(fractions[1:]),random_state=int(ds["random_seed"])); return {x:"train" for x in train}|{x:"val" for x in val}|{x:"test" for x in test}
def _label(o,w,h,seg):
    if seg and o.polygon:return f"{o.class_id} "+" ".join(f"{v:.8f}" for p in o.polygon for v in (p[0]/w,p[1]/h))
    x1,y1,x2,y2=o.bounding_box; return f"{o.class_id} {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}"
def prepare_dataset(cfg):
    if project_path(cfg,cfg["dataset"]["output_dir"]).resolve().name == "benchmark_v4":
        raise ValueError("benchmark_v4 is frozen. Use prepare --dataset benchmark_v5")
    out=project_path(cfg,cfg["dataset"]["output_dir"]); results=project_path(cfg,"results"); results.mkdir(parents=True,exist_ok=True)
    sources=cfg["input"].get("datasets")
    if not sources:sources=[{"id":"default","images_dir":cfg["input"]["images_dir"],"annotations_dir":cfg["input"]["annotations_dir"]}]
    ids=[str(source["id"]) for source in sources]
    if len(ids)!=len(set(ids)):raise ValueError("Every input.datasets entry must have a unique id")
    scans=[]
    for source in sources:
        current=scan_dataset(Path(source["images_dir"]),Path(source["annotations_dir"]),cfg["classes"])
        for annotation in current.annotations:annotation.source_dataset=str(source["id"])
        scans.append(current)
    scan=ScanResult([a for s in scans for a in s.annotations],"+".join(sorted({s.format for s in scans})),[p for s in scans for p in s.missing_annotations],[p for s in scans for p in s.orphan_annotations],[p for s in scans for p in s.unreadable],[p for s in scans for p in s.duplicate_names])
    excluded_labels={str(label).strip().casefold() for label in cfg["dataset"].get("exclude_images_with_labels",[])}
    excluded_shape_types={str(kind).strip().casefold() for kind in cfg["dataset"].get("exclude_images_with_shape_types",[])}
    excluded=[]; included=[]
    for annotation in scan.annotations:
        matched=sorted({obj.class_name for obj in annotation.objects if obj.class_name.strip().casefold() in excluded_labels})
        matched_shapes=sorted({obj.shape_type for obj in annotation.objects if obj.shape_type and obj.shape_type.casefold() in excluded_shape_types})
        if matched or matched_shapes:excluded.append({"source_dataset":annotation.source_dataset,"source_image":str(annotation.image_path),"source_annotation":str(annotation.annotation_path or ""),"matched_labels":";".join(matched),"matched_shape_types":";".join(matched_shapes),"reason":"excluded because image contains a configured label or shape type"})
        else:included.append(annotation)
    scan.annotations=included
    pd.DataFrame(excluded,columns=["source_dataset","source_image","source_annotation","matched_labels","matched_shape_types","reason"]).to_csv(results/"dataset_excluded_images.csv",index=False)
    issues=_issues(scan,cfg["classes"]); pd.DataFrame(issues,columns=["status","path","message"]).to_csv(results/"dataset_validation.csv",index=False)
    errors=[r for r in issues if r["status"].startswith("ERROR")]
    if errors:raise ValueError(f"Validation found {len(errors)} error(s). See {results/'dataset_validation.csv'}")
    manifest=out/"dataset_manifest.csv"; by_source={str(a.image_path.resolve()):a for a in scan.annotations}
    if manifest.exists() and not cfg["dataset"].get("rebuild_split",False):
        old=pd.read_csv(manifest); splits={str(Path(r.source_image).resolve()):r.split for r in old.itertuples()}
        if set(splits)!=set(by_source):raise ValueError("Existing manifest differs from source set. Set rebuild_split: true intentionally")
    else:splits=_split(list(by_source),cfg)
    out.mkdir(parents=True,exist_ok=True); rows=[]; seg=scan.format in {"yolo_segmentation","coco","labelme"}
    for source,a in by_source.items():
        split=splits[source]; image_id=hashlib.sha256(source.encode()).hexdigest()[:16]; name=image_id+a.image_path.suffix.lower(); image_dir=out/split/"images"; label_dir=out/split/"labels"; image_dir.mkdir(parents=True,exist_ok=True); label_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy2(a.image_path,image_dir/name); (label_dir/f"{image_id}.txt").write_text("\n".join(_label(o,a.width,a.height,seg) for o in a.objects),encoding="utf-8")
        rows.append({"source_dataset":a.source_dataset,"source_image":source,"source_annotation":str(a.annotation_path or ""),"image_id":image_id,"width":a.width,"height":a.height,"number_of_objects":len(a.objects),"split":split})
    frame=pd.DataFrame(rows).sort_values("image_id"); frame.to_csv(manifest,index=False)
    with (out/"dataset.yaml").open("w",encoding="utf-8") as f:yaml.safe_dump({"path":str(out.resolve()),"train":"train/images","val":"val/images","test":"test/images","names":cfg["classes"]},f,sort_keys=False)
    stats={"number_of_images":len(frame),"number_of_instances":int(frame.number_of_objects.sum()),"images_per_source_dataset":frame.groupby("source_dataset").size().to_dict(),"objects_per_source_dataset":frame.groupby("source_dataset").number_of_objects.sum().astype(int).to_dict(),"images_per_split":frame.groupby("split").size().to_dict(),"objects_per_split":frame.groupby("split").number_of_objects.sum().astype(int).to_dict(),"image_dimensions":frame.groupby(["width","height"]).size().rename("count").reset_index().to_dict("records"),"mean_objects_per_image":float(frame.number_of_objects.mean()),"median_objects_per_image":float(frame.number_of_objects.median()),"minimum_objects_per_image":int(frame.number_of_objects.min()),"maximum_objects_per_image":int(frame.number_of_objects.max()),"class_distribution":{str(k):sum(o.class_id==k for a in scan.annotations for o in a.objects) for k in cfg["classes"]},"annotation_format":scan.format}
    (results/"dataset_statistics.json").write_text(json.dumps(stats,indent=2),encoding="utf-8"); plt.figure(); frame.number_of_objects.hist(bins=min(30,max(1,len(frame)))); plt.xlabel("Trees per image"); plt.ylabel("Images"); plt.tight_layout(); plt.savefig(results/"tree_counts_per_image.png"); plt.close()
    from .coco_export import export_coco
    coco=export_coco(cfg);stats["dataset_fingerprint"]=coco["dataset_fingerprint"]
    (results/"dataset_statistics.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    LOG.info("Dataset preparation completed: %d images, %d instances; train=%d val=%d test=%d; excluded=%d",len(frame),stats["number_of_instances"],stats["images_per_split"].get("train",0),stats["images_per_split"].get("val",0),stats["images_per_split"].get("test",0),len(excluded)); return stats
