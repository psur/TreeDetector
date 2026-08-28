from datetime import datetime,timezone
from pathlib import Path
import csv,json
import pandas as pd
from src.config import project_path
from src.models.registry import get_model
from src.dataset.coco_export import dataset_fingerprint
from src.utils.reproducibility import runtime_info
COLUMNS="experiment_id model_family implementation model_name architecture backbone task pretrained_weights dataset_fingerprint train_images val_images test_images gt_instances predicted_instances epochs image_size parameters precision_box recall_box map50_box map75_box map50_95_box precision_mask recall_mask map50_mask map75_mask map50_95_mask metric_source training_seconds inference_seconds ms_per_image model_file_size_mb gpu timestamp".split()
def append_result(cfg,model,metrics):
 manifest=project_path(cfg,cfg["dataset"]["output_dir"])/"dataset_manifest.csv";frame=pd.read_csv(manifest);counts=frame.split.value_counts();test=int(counts.get("test",0));elapsed=metrics.get("inference_seconds");info=runtime_info();is_yolo=model.__class__.__name__=="YoloModel";weights=Path(model.spec.get("weights", ""));params=None
 try:params=sum(p.numel() for p in model.model.model.parameters()) if is_yolo else sum(p.numel() for p in model.model.parameters())
 except (AttributeError,TypeError):pass
 row={"experiment_id":model.experiment_dir.name,"model_family":"YOLO" if is_yolo else "Mask R-CNN","implementation":"Ultralytics" if is_yolo else "Detectron2","model_name":model.name,"architecture":"one-stage segmentation" if is_yolo else "Mask R-CNN","backbone":None if is_yolo else "ResNet-50-FPN","task":"segmentation","pretrained_weights":model.spec.get("weights") or model.spec.get("config"),"dataset_fingerprint":dataset_fingerprint(manifest),"train_images":int(counts.get("train",0)),"val_images":int(counts.get("val",0)),"test_images":test,"gt_instances":int(frame.loc[frame.split=="test","number_of_objects"].sum()),"epochs":cfg["training"]["epochs"] if is_yolo else None,"image_size":cfg["training"]["imgsz"] if is_yolo else None,"parameters":params,"training_seconds":model.training_seconds,"ms_per_image":elapsed*1000/test if elapsed is not None and test else None,"model_file_size_mb":weights.stat().st_size/2**20 if weights.is_file() else None,"gpu":info["gpu"],"timestamp":datetime.now(timezone.utc).isoformat()}|metrics
 model.experiment_dir.mkdir(parents=True,exist_ok=True);(model.experiment_dir/"experiment.json").write_text(json.dumps(row|{"environment":info},indent=2,default=str),encoding="utf8");path=project_path(cfg,"results")/"benchmark_results.csv";path.parent.mkdir(parents=True,exist_ok=True);old=pd.read_csv(path).to_dict("records") if path.exists() else [];old.append(row)
 with path.open("w",newline="",encoding="utf8") as f:
  writer=csv.DictWriter(f,fieldnames=COLUMNS,extrasaction="ignore");writer.writeheader();writer.writerows({k:r.get(k) for k in COLUMNS} for r in old)
def run(cfg,action,model_name="yolo",experiment=None,source=None):
 if model_name=="all":
  if experiment:raise ValueError("--experiment cannot be combined with --model all")
  return [run(cfg,action,n,None,source) for n,s in cfg["models"].items() if s.get("enabled")]
 model=get_model(model_name,cfg,experiment)
 if action in {"train","benchmark"}:model.train()
 if action in {"evaluate","benchmark"}:
  metrics=model.evaluate();append_result(cfg,model,metrics)
  if cfg["evaluation"].get("save_visualizations"):model.predict()
 if action=="predict":model.predict(source)
 return model.experiment_dir
def compare(cfg):
 path=project_path(cfg,"results")/"benchmark_results.csv"
 if not path.exists():raise FileNotFoundError(f"No benchmark results found: {path}")
 frame=pd.read_csv(path);cols=[c for c in ("model_family","implementation","model_name","map50_mask","map50_95_mask","recall_mask","ms_per_image") if c in frame];print(frame[cols].to_string(index=False,na_rep="-"));frame[cols].to_csv(project_path(cfg,"results")/"benchmark_summary.csv",index=False)
