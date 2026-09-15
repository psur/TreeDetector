"""Backend cleanup regression checks; runnable without optional pytest."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from main import build_parser
from src.config import load_config
from src.models.registry import get_model
from src.benchmark.runner import append_result

class BackendCleanupTests(unittest.TestCase):
    def test_cli_rejects_unimplemented_backends(self):
        for action in ("train", "evaluate", "benchmark", "predict"):
            self.assertEqual(build_parser().parse_args([action, "--model", "yolo"]).model, "yolo")
            for backend in ("detectree2", "all"):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    build_parser().parse_args([action, "--model", backend])
                self.assertEqual(caught.exception.code, 2)
                with self.assertRaises(ValueError):
                    get_model(backend, {})

    def test_yolo_result_logging_preserves_existing_rows(self):
        cfg = load_config("config/config.example.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg["_project_root"] = str(root)
            dataset = root / cfg["dataset"]["output_dir"]
            dataset.mkdir(parents=True)
            pd.DataFrame([dict(image_id="a", split="test", source_image="a.png", width=10, height=10, number_of_objects=2)]).to_csv(dataset / "dataset_manifest.csv", index=False)
            model = get_model("yolo", cfg, root / "runs" / "test")
            with patch("src.benchmark.runner.runtime_info", return_value={"gpu": "test"}):
                append_result(cfg, model, {"map50_mask": 0.719, "metric_source": "Ultralytics"})
                append_result(cfg, model, {"map50_mask": 0.8, "metric_source": "Ultralytics"})
            rows = pd.read_csv(root / "results" / "benchmark_results.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows.iloc[0].map50_mask, 0.719)
            self.assertEqual(rows.iloc[0].implementation, "Ultralytics")
            self.assertEqual(rows.iloc[0].model_family, "YOLO")
            self.assertEqual(rows.iloc[0].epochs, 100)
            self.assertEqual(rows.iloc[0].image_size, 640)
            self.assertEqual(rows.iloc[0].gt_instances, 2)
            self.assertEqual(json.loads((model.experiment_dir / "experiment.json").read_text())["model_name"], "yolo11m-seg")

if __name__ == "__main__":
    unittest.main()
