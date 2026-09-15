"""Hugging Face Mask2Former benchmark; optional dependencies load on use."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import random
import time

from src.config import project_path, public_config
from .base_model import BenchmarkModel

LOG = logging.getLogger(__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


class Mask2FormerModel(BenchmarkModel):
    def __init__(self, cfg, experiment_dir=None):
        self.cfg = cfg
        self.spec = cfg["models"]["mask2former"]
        runs = project_path(cfg, "runs").resolve()
        self.experiment_dir = (Path(experiment_dir).resolve() if experiment_dir else
                               runs / f"mask2former_{datetime.now():%Y%m%d_%H%M%S_%f}")
        if self.experiment_dir.parent != runs or not self.experiment_dir.name.startswith("mask2former_"):
            raise ValueError("Mask2Former experiments must be runs/mask2former_*; other runs are protected")
        self.model = None
        self.processor = None
        self.training_seconds = None
        self.completed_epochs = 0
        self.root = project_path(cfg, cfg["dataset"]["output_dir"])
        self.seed = int(self.spec["seed"])

    @property
    def name(self):
        return self.spec["model_name"]

    def benchmark_metadata(self):
        return {"model_family": "Mask2Former", "implementation": "Hugging Face Transformers",
                "architecture": "masked-attention transformer instance segmentation", "backbone": "Swin-Tiny",
                "epochs": self.completed_epochs, "image_size": self.spec["image_size"],
                "pretrained_weights": self.name, "seed": self.seed, "optimizer": "AdamW",
                "learning_rate": self.spec["learning_rate"], "batch_size": self.spec["batch_size"],
                "gradient_accumulation_steps": self.spec["gradient_accumulation_steps"],
                "precision": "float32", "checkpoint_revision": self.spec["revision"]}

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.model.parameters())

    def prepare(self, for_training=False, smoke=False):
        import torch
        import yaml
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        from src.dataset.mask2former_dataset import validate_dataset, CocoInstanceDataset, InstanceCollator
        from src.utils.reproducibility import set_seed, runtime_info
        from scipy.optimize import linear_sum_assignment  # fail before model download if missing

        if self.model is not None:
            return self
        if self.cfg["classes"] != {0: "tree", 1: "TreeGroup"}:
            raise ValueError("Expected project classes 0=tree and 1=TreeGroup")
        for key in ("epochs", "batch_size", "gradient_accumulation_steps", "image_size", "patience"):
            if int(self.spec[key]) <= 0:
                raise ValueError(f"Mask2Former {key} must be positive")
        if self.spec["image_size"] % 32:
            raise ValueError("image_size must be divisible by 32")
        if float(self.spec["learning_rate"]) <= 0 or float(self.spec["weight_decay"]) < 0:
            raise ValueError("Invalid learning rate or weight decay")
        if self.spec.get("mixed_precision", False):
            raise ValueError("This benchmark uses float32; mixed precision is not enabled")
        requested = self.spec.get("device", "cuda")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Mask2Former requested CUDA but CUDA is unavailable")
        self.device = torch.device(requested)
        set_seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        audit = validate_dataset(self.root)
        annotation_hashes = {s: hashlib.sha256((self.root / "coco" / "annotations" /
                            f"instances_{s}.json").read_bytes()).hexdigest() for s in ("train", "val", "test")}
        identity = {"backend": "mask2former", "dataset_fingerprint": audit["dataset_fingerprint"],
                    "annotation_sha256": annotation_hashes, "spec": self.spec,
                    "purpose": "smoke" if smoke else "benchmark"}
        marker = self.experiment_dir / "run_identity.json"
        if self.experiment_dir.exists():
            if not marker.is_file():
                raise ValueError("Refusing an existing experiment without Mask2Former identity")
            old = json.loads(marker.read_text())
            if old != identity:
                raise ValueError("Experiment config/dataset/purpose mismatch; use a new run directory")
        elif not for_training and not smoke:
            raise FileNotFoundError("Evaluation/prediction requires --experiment with a trained best checkpoint")
        else:
            self.experiment_dir.mkdir(parents=True, exist_ok=False)
            write_json(marker, identity)
            (self.experiment_dir / "config.yaml").write_text(yaml.safe_dump(public_config(self.cfg), sort_keys=False))
            write_json(self.experiment_dir / "dataset_audit.json", audit)
            versions = {}
            for package in ("torch", "torchvision", "transformers", "pycocotools", "scipy", "numpy", "Pillow", "pandas", "opencv-python"):
                versions[package] = importlib.metadata.version(package)
            write_json(self.experiment_dir / "environment.json", runtime_info() | {"packages": versions,
                       "cuda_runtime": torch.version.cuda, "device_capability": torch.cuda.get_device_capability(self.device)
                       if self.device.type == "cuda" else None, "seed": self.seed})
        best = self.experiment_dir / "best"
        last = self.experiment_dir / "last"
        source = last if for_training and (last / "trainer_state.pt").is_file() else best
        if not source.is_dir():
            if not for_training and not smoke:
                raise FileNotFoundError("No trained best checkpoint in experiment")
            source = self.name
        cache = project_path(self.cfg, ".cache/huggingface")
        options = {"cache_dir": str(cache)}
        if isinstance(source, str):
            options["revision"] = self.spec["revision"]
        self.processor = AutoImageProcessor.from_pretrained(source, **options)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            source, id2label=self.cfg["classes"], label2id={v:k for k,v in self.cfg["classes"].items()},
            ignore_mismatched_sizes=isinstance(source, str), **options).to(self.device)
        if self.model.config.num_labels != 2 or self.model.class_predictor.out_features != 3:
            raise RuntimeError("Classification head must contain two labels plus no-object")
        if max(s["max_instances"] for s in audit["splits"].values()) > self.model.config.num_queries:
            raise ValueError("Instance count exceeds model query capacity")
        self.datasets = {s: CocoInstanceDataset(self.root, s) for s in ("train", "val", "test")}
        self.collator = InstanceCollator(self.processor, int(self.spec["image_size"]))
        if (self.experiment_dir / "training_history.json").exists():
            history = json.loads((self.experiment_dir / "training_history.json").read_text())
            self.completed_epochs = len(history)
            self.training_seconds = sum(r["epoch_seconds"] for r in history)
        return self

    def _inputs(self, batch, targets=True):
        return {key: ([item.to(self.device) for item in value] if isinstance(value, list) else value.to(self.device))
                for key, value in batch["inputs"].items()
                if targets or key not in ("mask_labels", "class_labels")}

    def _loader(self, split, epoch=0):
        import torch
        from torch.utils.data import DataLoader
        generator = torch.Generator().manual_seed(self.seed + epoch)
        return DataLoader(self.datasets[split], batch_size=int(self.spec["batch_size"]) if split == "train" else 1,
                          shuffle=split == "train", num_workers=0, collate_fn=self.collator, generator=generator)

    def _sync(self):
        import torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _evaluate_split(self, split, destination, limit=None):
        import torch
        from src.benchmark.coco_evaluation import prediction_records, evaluate_coco
        self.model.eval()
        records, image_ids = [], []
        forward_seconds = 0.0
        start = time.perf_counter()
        with torch.no_grad():
            for index, batch in enumerate(self._loader(split)):
                if limit is not None and index >= limit:
                    break
                inputs = self._inputs(batch, targets=False)
                self._sync()
                tick = time.perf_counter()
                outputs = self.model(**inputs)
                self._sync()
                forward_seconds += time.perf_counter() - tick
                records.extend(prediction_records(self.processor, outputs, batch["image_ids"], batch["sizes"]))
                image_ids.extend(batch["image_ids"])
        metrics = evaluate_coco(self.datasets[split].annotation_path, records, image_ids)
        metrics.update({"split": split, "inference_seconds": time.perf_counter()-start,
                        "forward_seconds": forward_seconds, "timing_protocol": "synchronized forward only, no warmup; inference_seconds includes loading/postprocessing/evaluation",
                        "evaluation_images": len(image_ids),
                        "model_file_size_mb": sum(p.stat().st_size for p in (self.experiment_dir / "best").glob("*.safetensors"))/2**20
                        if (self.experiment_dir / "best").is_dir() else None})
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination / "predictions_coco.json", records)
        write_json(destination / "metrics.json", metrics)
        write_json(destination / "per_class_metrics.json", metrics["per_class"])
        return metrics

    def _save_checkpoint(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(folder)
        self.processor.save_pretrained(folder)

    def train(self):
        import numpy as np
        import torch
        self.prepare(for_training=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.spec["learning_rate"]),
                                      weight_decay=float(self.spec["weight_decay"]))
        last = self.experiment_dir / "last"
        state_path = last / "trainer_state.pt"
        start_epoch, best_ap, stale, history = 0, -1.0, 0, []
        if state_path.is_file():
            # Only load locally produced optimizer/RNG state from this validated run.
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            start_epoch, best_ap, stale, history = state["epoch"], state["best_ap"], state["stale"], state["history"]
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        accumulation = int(self.spec["gradient_accumulation_steps"])
        for epoch in range(start_epoch, int(self.spec["epochs"])):
            if stale >= int(self.spec["patience"]):
                break
            tick = time.perf_counter()
            self.model.train()
            loader = self._loader("train", epoch)
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for index, batch in enumerate(loader):
                output = self.model(**self._inputs(batch))
                loss = output.loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                # Normalize a short final accumulation group by its actual batch count.
                group_start = (index // accumulation) * accumulation
                divisor = min(accumulation, len(loader) - group_start)
                (loss / divisor).backward()
                loss_total += loss.item()
                if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.spec["max_grad_norm"]), error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            validation = self._evaluate_split("val", self.experiment_dir / f"validation_epoch_{epoch+1:03d}_{datetime.now():%H%M%S_%f}")
            ap = validation["map50_95_mask"]
            if ap is None:
                raise RuntimeError("Validation AP is undefined")
            improved = ap > best_ap
            stale = 0 if improved else stale + 1
            if improved:
                best_ap = ap
                self._save_checkpoint(self.experiment_dir / "best")
                write_json(self.experiment_dir / "best_validation_metrics.json", validation | {"epoch": epoch+1})
            self.completed_epochs = epoch + 1
            history.append({"epoch": epoch+1, "training_loss": loss_total/len(loader), "validation_map50_95_mask": ap,
                            "validation_map50_mask": validation["map50_mask"], "epoch_seconds": time.perf_counter()-tick})
            self.training_seconds = sum(r["epoch_seconds"] for r in history)
            self._save_checkpoint(last)
            torch.save({"epoch": epoch+1, "optimizer": optimizer.state_dict(), "best_ap": best_ap, "stale": stale,
                        "history": history, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []}, state_path)
            write_json(self.experiment_dir / "training_history.json", history)
            LOG.info("Epoch %d: loss %.4f, validation mask AP %.4f", epoch+1, history[-1]["training_loss"], ap)
        # Benchmark must test the validation-selected best weights, never the last epoch implicitly.
        from transformers import Mask2FormerForUniversalSegmentation
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.experiment_dir / "best").to(self.device)
        return self.experiment_dir

    def evaluate(self):
        self.prepare()
        return self._evaluate_split("test", self.experiment_dir / f"test_{datetime.now():%Y%m%d_%H%M%S_%f}")

    def predict(self, source=None):
        import torch
        from PIL import Image
        from src.benchmark.coco_evaluation import prediction_records
        from src.utils.mask_visualization import draw_predictions
        self.prepare()
        self.model.eval()
        output = self.experiment_dir / f"predictions_{datetime.now():%Y%m%d_%H%M%S_%f}"
        output.mkdir(exist_ok=False)
        if source:
            paths = [Path(source)]
        else:
            dataset = self.datasets["test"]
            paths = sorted(dataset.image_path(i) for i in dataset.images)
            paths = random.Random(self.seed).sample(paths, min(len(paths), self.cfg["evaluation"]["visualization_count"]))
        write_json(output / "selected_images.json", [str(p) for p in paths])
        with torch.no_grad():
            for index, path in enumerate(paths):
                with Image.open(path) as image:
                    sample = {"image": image.convert("RGB"), "masks": [], "labels": [],
                              "image_id": index, "size": (image.height, image.width)}
                batch = self.collator([sample])
                predictions = prediction_records(self.processor, self.model(**self._inputs(batch, False)),
                                                 batch["image_ids"], batch["sizes"])
                draw_predictions(path, predictions, output / f"{path.stem}.png", self.cfg["classes"],
                                 self.cfg["evaluation"]["confidence_threshold"])
        return output

    def smoke_test(self):
        import torch
        self.prepare(for_training=True, smoke=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        # Exercise the most crowded training image to avoid a deceptively easy memory test.
        dataset = self.datasets["train"]
        index = max(range(len(dataset)), key=lambda i: len(dataset.annotations[dataset.images[i]["id"]]))
        batch = self.collator([dataset[index]])
        dimensions = {key: [list(t.shape) for t in value] if isinstance(value, list) else list(value.shape)
                      for key, value in batch["inputs"].items()}
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.spec["learning_rate"]),
                                      weight_decay=float(self.spec["weight_decay"]))
        output = self.model(**self._inputs(batch))
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError("Smoke training loss is not finite")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.spec["max_grad_norm"]), error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        validation = self._evaluate_split("val", self.experiment_dir / "smoke_validation", limit=1)
        self._save_checkpoint(self.experiment_dir / "smoke_checkpoint")
        report = {"status": "PASS", "purpose": "one-batch smoke only; not benchmark metrics", "training_loss": loss.item(),
                  "gradient_norm": float(gradient), "training_image_id": dataset.images[index]["id"],
                  "dimensions": dimensions, "dataset": {s:d.summary() for s,d in self.datasets.items()},
                  "validation_images": validation["evaluation_images"], "gpu": torch.cuda.get_device_name(self.device)
                  if self.device.type == "cuda" else "CPU", "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(self.device)/2**30
                  if self.device.type == "cuda" else None, "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(self.device)/2**30
                  if self.device.type == "cuda" else None, "checkpoint": self.name, "revision": self.spec["revision"]}
        write_json(self.experiment_dir / "smoke_test.json", report)
        print(json.dumps(report, indent=2))
        return self.experiment_dir
