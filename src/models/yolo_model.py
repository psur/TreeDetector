from datetime import datetime
from pathlib import Path
import json,random,time,yaml
from src.config import project_path,public_config
from src.utils.reproducibility import resolve_device
from .base_model import BenchmarkModel
class YoloModel(BenchmarkModel):
    def __init__(self,cfg,experiment_dir=None):
        self.cfg=cfg;self.spec=cfg["models"]["yolo"];self.experiment_dir=Path(experiment_dir) if experiment_dir else project_path(cfg,"runs")/f"yolo_seg_{datetime.now():%Y%m%d_%H%M%S_%f}";self.model=None;self.training_seconds=None;self.resume_checkpoint=None
    @property
    def name(self):return Path(self.spec["weights"]).stem
    def prepare(self,for_training=False):
        dataset_yaml=project_path(self.cfg,self.cfg["dataset"]["output_dir"])/"dataset.yaml"
        if not dataset_yaml.is_file():raise RuntimeError(f"Prepared dataset not found: {dataset_yaml}. Run 'python main.py prepare' successfully before training or evaluation")
        try:from ultralytics import YOLO
        except ImportError as exc:raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        if not self.experiment_dir.exists():
            self.experiment_dir.mkdir(parents=True);(self.experiment_dir/"model").mkdir();(self.experiment_dir/"predictions").mkdir()
            with (self.experiment_dir/"config.yaml").open("w",encoding="utf-8") as f:yaml.safe_dump(public_config(self.cfg),f,sort_keys=False)
        weights=self.spec["weights"]
        last_files=sorted(self.experiment_dir.glob("**/weights/last.pt"),key=lambda p:p.stat().st_mtime,reverse=True)
        best_files=sorted(self.experiment_dir.glob("**/weights/best.pt"),key=lambda p:p.stat().st_mtime,reverse=True)
        self.resume_checkpoint=last_files[0] if last_files else None
        best_checkpoint=best_files[0] if best_files else None
        checkpoint=self.resume_checkpoint if for_training and self.resume_checkpoint else best_checkpoint or self.resume_checkpoint
        self.model=YOLO(str(checkpoint or weights));return self
    def train(self):
        if self.model is None:self.prepare(for_training=True)
        from .yolo_trainer import RetryingSegmentationTrainer
        t=self.cfg["training"];start=time.perf_counter()
        if self.resume_checkpoint:
            result=self.model.train(resume=True,trainer=RetryingSegmentationTrainer)
        else:
            result=self.model.train(data=str(project_path(self.cfg,self.cfg["dataset"]["output_dir"])/"dataset.yaml"),epochs=t["epochs"],imgsz=t["imgsz"],batch=t["batch"],patience=t["patience"],workers=t["workers"],seed=t["seed"],pretrained=t.get("pretrained",True),device=resolve_device(t["device"]),project=str(self.experiment_dir),name="training",exist_ok=True,trainer=RetryingSegmentationTrainer)
        self.training_seconds=time.perf_counter()-start;return result
    def predict(self,source=None):
        if self.model is None:self.prepare()
        e=self.cfg["evaluation"]
        if source:
            selected=Path(source)
            if not selected.is_file():raise FileNotFoundError(f"Prediction source image not found: {selected}")
            files=[selected]
        else:
            test_dir=project_path(self.cfg,self.cfg["dataset"]["output_dir"])/"test"/"images";candidates=sorted(p for p in test_dir.iterdir() if p.is_file());count=min(int(e.get("visualization_count",20)),len(candidates));files=random.Random(int(self.cfg["training"]["seed"])).sample(candidates,count)
        return self.model.predict(source=[str(p) for p in files],conf=e["confidence_threshold"],iou=e["iou_threshold"],save=True,project=str(self.experiment_dir),name="predictions",exist_ok=True)
    def evaluate(self):
        if self.model is None:self.prepare()
        start=time.perf_counter();t=self.cfg["training"];r=self.model.val(data=str(project_path(self.cfg,self.cfg["dataset"]["output_dir"])/"dataset.yaml"),split="test",imgsz=t["imgsz"],device=resolve_device(t["device"]),project=str(self.experiment_dir),name="evaluation",exist_ok=True);elapsed=time.perf_counter()-start;box=getattr(r,"box",None);seg=getattr(r,"seg",None)
        val=lambda o,k:float(getattr(o,k)) if o is not None and getattr(o,k,None) is not None else None
        m={"precision_box":val(box,"mp"),"recall_box":val(box,"mr"),"map50_box":val(box,"map50"),"map75_box":None,"map50_95_box":val(box,"map"),"precision_mask":val(seg,"mp"),"recall_mask":val(seg,"mr"),"map50_mask":val(seg,"map50"),"map75_mask":None,"map50_95_mask":val(seg,"map"),"inference_seconds":elapsed,"metric_source":"Ultralytics"};(self.experiment_dir/"metrics.json").write_text(json.dumps(m,indent=2),encoding="utf-8");return m
