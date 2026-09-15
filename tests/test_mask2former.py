"""Mask/metric correctness tests without model downloads or training."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from PIL import Image
try:
    import torch
    from pycocotools import mask as mask_utils
    from transformers import Mask2FormerImageProcessor
    from src.dataset.mask2former_dataset import InstanceCollator, decode_mask, CocoInstanceDataset
    from src.benchmark.coco_evaluation import evaluate_coco, prediction_records
    from src.models.mask2former_model import Mask2FormerModel
    from src.config import load_config
    from src.utils.mask_visualization import draw_predictions
except ImportError as exc:
    raise unittest.SkipTest(f"Optional Mask2Former stack is not installed: {exc}") from exc



class Mask2FormerTests(unittest.TestCase):
    def setUp(self):
        self.processor = Mask2FormerImageProcessor(size={"height": 32, "width": 32})

    def test_overlaps_and_empty_targets_survive_collation(self):
        first = np.zeros((64,64), dtype=np.uint8)
        second = first.copy()
        first[4:40,4:40] = 1
        second[20:60,20:60] = 1
        sample = {"image": Image.new("RGB", (64,64)), "masks": [first,second],
                  "labels": [0,1], "image_id": 1, "size": (64,64)}
        empty = dict(sample, masks=[], labels=[], image_id=2)
        batch = InstanceCollator(self.processor, 32)([sample,empty])
        inputs = batch["inputs"]
        self.assertEqual(list(inputs["pixel_values"].shape), [2,3,32,32])
        self.assertEqual(list(inputs["mask_labels"][0].shape), [2,32,32])
        self.assertTrue((inputs["mask_labels"][0].sum(0)>1).any())
        self.assertEqual(list(inputs["mask_labels"][1].shape), [0,32,32])
        self.assertEqual(inputs["class_labels"][0].tolist(), [0,1])

    def test_polygon_and_rle_decode_agree(self):
        annotation = {"segmentation": [[2,2,20,2,20,20,2,20]]}
        plane = decode_mask(annotation,32,32)
        rle = mask_utils.encode(np.asfortranarray(plane))
        self.assertTrue(np.array_equal(plane,decode_mask({"segmentation":rle},32,32)))

    def test_postprocessing_preserves_overlapping_predictions(self):
        output = SimpleNamespace(class_queries_logits=torch.tensor([[[12.,-12.,-12.],[-12.,12.,-12.]]]),
                                 masks_queries_logits=torch.ones((1,2,8,8))*10)
        records = prediction_records(self.processor,output,[7],[(32,32)])
        self.assertEqual(len(records),2)
        self.assertEqual({r["category_id"] for r in records},{1,2})
        self.assertTrue((mask_utils.decode(records[0]["segmentation"]) & mask_utils.decode(records[1]["segmentation"])).any())

    def test_coco_perfect_empty_and_subset_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plane = np.ones((32,32),dtype=np.uint8)
            rle = mask_utils.encode(np.asfortranarray(plane))
            rle["counts"] = rle["counts"].decode()
            truth = {"images":[{"id":1,"file_name":"a.png","height":32,"width":32},
                               {"id":2,"file_name":"b.png","height":32,"width":32}],
                     "categories":[{"id":1,"name":"tree"},{"id":2,"name":"TreeGroup"}],
                     "annotations":[{"id":1,"image_id":1,"category_id":1,"iscrowd":0,"area":1024,
                                     "bbox":[0,0,32,32],"segmentation":rle}]}
            path = root / "truth.json"
            path.write_text(json.dumps(truth))
            prediction = {"image_id":1,"category_id":1,"score":0.9,"segmentation":rle}
            with contextlib.redirect_stdout(io.StringIO()):
                perfect = evaluate_coco(path,[prediction],[1])
                empty = evaluate_coco(path,[],[1])
            self.assertAlmostEqual(perfect["map50_mask"],1.0)
            self.assertAlmostEqual(perfect["per_class"]["tree"]["map50_95_mask"],1.0)
            self.assertIsNone(perfect["per_class"]["TreeGroup"]["map50_mask"])
            self.assertEqual(empty["map50_mask"],0.0)
            self.assertEqual(perfect["evaluation_image_ids"],[1])
            self.assertIsNone(perfect["precision_mask"])

    def test_visualization_leaves_mask_interior_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB",(128,128),(80,90,100)).save(source)
            plane = np.zeros((128,128),dtype=np.uint8)
            plane[20:110,20:110] = 1
            rle = mask_utils.encode(np.asfortranarray(plane))
            target = root / "output.png"
            draw_predictions(source,[{"category_id":1,"score":0.9,"segmentation":rle}],target,{0:"tree",1:"TreeGroup"})
            with Image.open(target) as result:
                self.assertEqual(result.getpixel((64,64)),(80,90,100))

    def test_training_accumulation_checkpoint_selection_and_resume(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5))
            def forward(self, **inputs):
                return SimpleNamespace(loss=(self.weight-1).square())
            def save_pretrained(self, folder):
                torch.save(self.state_dict(), folder / "tiny.pt")
        class TinyProcessor:
            def save_pretrained(self, folder):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            cfg = load_config("config/config.example.yaml")
            cfg["_project_root"] = temporary
            cfg["models"]["mask2former"].update(epochs=3, patience=2, gradient_accumulation_steps=4)
            model = Mask2FormerModel(cfg)
            model.experiment_dir.mkdir(parents=True)
            model.model, model.processor, model.device = TinyModel(), TinyProcessor(), torch.device("cpu")
            model.prepare = lambda **kwargs: model
            model._loader = lambda *args: [{"inputs": {}} for _ in range(5)]
            scores = iter([0.8,0.7,0.6])
            model._evaluate_split = lambda *args: {"map50_95_mask": next(scores), "map50_mask": 0.9}
            def restore(path):
                restored = TinyModel()
                restored.load_state_dict(torch.load(path / "tiny.pt", weights_only=True))
                return restored
            with patch("transformers.Mask2FormerForUniversalSegmentation.from_pretrained", side_effect=restore):
                model.train()
                best = torch.load(model.experiment_dir / "best" / "tiny.pt", weights_only=True)["weight"]
                self.assertTrue(torch.equal(model.model.weight.detach(), best))
                state = torch.load(model.experiment_dir / "last" / "trainer_state.pt", weights_only=False)
                self.assertEqual(state["epoch"],3)
                self.assertEqual(state["best_ap"],0.8)
                # 5 batches, accumulation 4 => 2 optimizer steps each epoch.
                self.assertEqual(int(next(iter(state["optimizer"]["state"].values()))["step"]),6)
                model.train()  # Completed/early-stopped state resumes without training more.
                self.assertEqual(len(json.loads((model.experiment_dir / "training_history.json").read_text())),3)

    def test_rejects_yolo_experiment_destination(self):
        cfg = load_config("config/config.example.yaml")
        with self.assertRaises(ValueError):
            Mask2FormerModel(cfg, Path(cfg["_project_root"])/"runs"/"yolo_seg_20260912_193929_824528")

if __name__ == "__main__":
    unittest.main()
