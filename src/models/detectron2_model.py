"""Optional Detectron2 Mask R-CNN adapter; all imports are deliberately lazy."""
from datetime import datetime
from pathlib import Path
import json,time,yaml
from .base_model import BenchmarkModel
from src.config import project_path,public_config
from src.dataset.coco_export import export_coco,dataset_fingerprint

class Detectron2MaskRCNNModel(BenchmarkModel):
 def __init__(self,cfg,experiment_dir=None):
  self.cfg=cfg;self.spec=cfg["models"]["detectree2"];self.experiment_dir=Path(experiment_dir) if experiment_dir else project_path(cfg,"runs")/f"maskrcnn_detectron2_{datetime.now():%Y%m%d_%H%M%S_%f}";self.d2cfg=None;self.model=None;self.training_seconds=None
 @property
 def name(self):return "mask_rcnn_R_50_FPN_3x"
 def _d2(self):
  try:
   from detectron2.config import get_cfg
   from detectron2 import model_zoo
   from detectron2.data.datasets import register_coco_instances
   from detectron2.engine import DefaultTrainer,DefaultPredictor
   from detectron2.evaluation import COCOEvaluator,inference_on_dataset
   from detectron2.data import build_detection_test_loader
   return get_cfg,model_zoo,register_coco_instances,DefaultTrainer,DefaultPredictor,COCOEvaluator,inference_on_dataset,build_detection_test_loader
  except ImportError as e:raise RuntimeError("Mask R-CNN / Detectron2 dependencies are not installed. See README -> Milestone 2 setup. YOLO remains usable.") from e
 def prepare(self,for_training=False):
  export_coco(self.cfg);get_cfg,zoo,register,*_=self._d2();root=project_path(self.cfg,self.cfg["dataset"]["output_dir"]);ann=root/"coco"/"annotations";tag=dataset_fingerprint(root/"dataset_manifest.csv")[:12];self.names={s:f"treecrowns_{tag}_{s}" for s in ("train","val","test")}
  for s,n in self.names.items():
   try:register(n,{},str(ann/f"instances_{s}.json"),"")
   except AssertionError:pass
  self.experiment_dir.mkdir(parents=True,exist_ok=True)
  for d in ("model","logs","predictions"): (self.experiment_dir/d).mkdir(exist_ok=True)
  (self.experiment_dir/"config.yaml").write_text(yaml.safe_dump(public_config(self.cfg),sort_keys=False),encoding="utf8")
  c=get_cfg();key=self.spec["config"];c.merge_from_file(zoo.get_config_file(key));t=self.cfg["detectree2_training"];c.DATASETS.TRAIN=(self.names["train"],);c.DATASETS.TEST=(self.names["val"],);c.DATALOADER.NUM_WORKERS=t["num_workers"];c.SOLVER.IMS_PER_BATCH=t["ims_per_batch"];c.SOLVER.BASE_LR=t["base_lr"];c.SOLVER.MAX_ITER=t["max_iterations"];c.SOLVER.CHECKPOINT_PERIOD=t["checkpoint_period"];c.TEST.EVAL_PERIOD=t["evaluation_period"];c.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=t["roi_batch_size_per_image"];c.MODEL.ROI_HEADS.NUM_CLASSES=len(self.cfg["classes"]);c.OUTPUT_DIR=str(self.experiment_dir/"model");final=Path(c.OUTPUT_DIR)/"model_final.pth";c.MODEL.WEIGHTS=str(final if final.exists() else zoo.get_checkpoint_url(key));c.MODEL.DEVICE="cuda" if __import__("torch").cuda.is_available() else "cpu";c.SEED=self.cfg["training"]["seed"];self.d2cfg=c;return self
 def train(self):
  if self.d2cfg is None:self.prepare(True)
  Trainer=self._d2()[3];start=time.perf_counter()
  try:t=Trainer(self.d2cfg);t.resume_or_load(resume=(Path(self.d2cfg.OUTPUT_DIR)/"last_checkpoint").exists());t.train()
  except RuntimeError as e:
   if "out of memory" in str(e).lower():raise RuntimeError("CUDA OOM: reduce ims_per_batch, image resolution, or roi_batch_size_per_image") from e
   raise
  self.training_seconds=time.perf_counter()-start;self.model=t.model;return t
 def _predictor(self):
  if self.d2cfg is None:self.prepare()
  final=self.experiment_dir/"model"/"model_final.pth"
  if not final.exists():raise FileNotFoundError(f"Checkpoint not found: {final}")
  self.d2cfg.MODEL.WEIGHTS=str(final);self.d2cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=self.cfg["evaluation"]["confidence_threshold"];return self._d2()[4](self.d2cfg)
 def predict(self,source=None):
  import cv2
  p=self._predictor();root=project_path(self.cfg,self.cfg["dataset"]["output_dir"]);data=json.loads((root/"coco"/"annotations"/"instances_test.json").read_text());records=[]
  for im in data["images"]:
   if source and Path(source).resolve()!=Path(im["file_name"]).resolve():continue
   out=p(cv2.imread(im["file_name"]))["instances"].to("cpu")
   for box,score,cid,mask in zip(out.pred_boxes.tensor.tolist(),out.scores.tolist(),out.pred_classes.tolist(),out.pred_masks.numpy()):
    flat=mask.astype("uint8").flatten(order="F");counts=[];last=0;run=0
    for value in flat:
     if value==last:run+=1
     else:counts.append(run);run=1;last=int(value)
    counts.append(run);records.append({"image_id":im["id"],"canonical_image_id":im["canonical_image_id"],"category_id":cid+1,"score":score,"bbox":[box[0],box[1],box[2]-box[0],box[3]-box[1]],"segmentation":{"size":list(mask.shape),"counts":counts}})
  (self.experiment_dir/"predictions"/"predictions.json").write_text(json.dumps(records),encoding="utf8");return records
 def evaluate(self):
  p=self._predictor();*_,Evaluator,infer,loader=self._d2();ev=Evaluator(self.names["test"],tasks=("bbox","segm"),distributed=False,output_dir=str(self.experiment_dir/"evaluation"));start=time.perf_counter();raw=infer(p.model,loader(self.d2cfg,self.names["test"]),ev);elapsed=time.perf_counter()-start
  get=lambda task,key: raw.get(task,{}).get(key)/100 if raw.get(task,{}).get(key) is not None else None
  m={"precision_box":None,"recall_box":None,"map50_box":get("bbox","AP50"),"map75_box":get("bbox","AP75"),"map50_95_box":get("bbox","AP"),"precision_mask":None,"recall_mask":None,"map50_mask":get("segm","AP50"),"map75_mask":get("segm","AP75"),"map50_95_mask":get("segm","AP"),"inference_seconds":elapsed,"metric_source":"Detectron2 COCOEvaluator"};(self.experiment_dir/"metrics.json").write_text(json.dumps(m,indent=2));return m
