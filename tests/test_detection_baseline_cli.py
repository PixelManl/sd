import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


def load_module(module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "detect" / "scripts" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DetectionBaselineCliTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.config_path = self.repo_root / "bolt" / "detect" / "configs" / "baseline.yaml"
        self.train_module = load_module("train_baseline")
        self.eval_module = load_module("eval_baseline")
        self.infer_module = load_module("infer_baseline")

    def test_train_build_job_uses_dataset_yaml_and_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "prepared"
            dataset_root.mkdir(parents=True, exist_ok=True)
            (dataset_root / "dataset.yaml").write_text("path: .\n", encoding="utf-8")

            args = Namespace(
                config=self.config_path,
                dataset_root=dataset_root,
                run_dir=tmp_path / "runs" / "custom-train",
                weights="custom.pt",
                epochs=3,
                batch_size=2,
                imgsz=640,
                device="cpu",
                execute=True,
                dry_run=False,
            )

            config = self.train_module.maybe_load_yaml(args.config)
            job = self.train_module.build_train_job(args, config)
            plan = self.train_module.build_plan(args, config)

            self.assertEqual(job["model"], "custom.pt")
            self.assertEqual(job["run_dir"], args.run_dir.resolve())
            self.assertEqual(job["kwargs"]["data"], str((dataset_root / "dataset.yaml").resolve()))
            self.assertEqual(job["kwargs"]["project"], str(args.run_dir.resolve().parent))
            self.assertEqual(job["kwargs"]["name"], args.run_dir.name)
            self.assertEqual(job["kwargs"]["batch"], 2)
            self.assertEqual(job["kwargs"]["imgsz"], 640)
            self.assertEqual(plan["status"], "execute-requested")
            self.assertEqual(plan["run_dir"], str(args.run_dir.resolve()))

    def test_eval_execute_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "prepared"
            run_dir = tmp_path / "runs" / "eval"
            weights = tmp_path / "weights" / "best.pt"
            dataset_root.mkdir(parents=True, exist_ok=True)
            weights.parent.mkdir(parents=True, exist_ok=True)
            (dataset_root / "dataset.yaml").write_text("path: .\n", encoding="utf-8")
            weights.write_bytes(b"fake-weights")

            args = Namespace(
                config=self.config_path,
                dataset_root=dataset_root,
                run_dir=run_dir,
                weights=weights,
                split="val",
                conf_threshold=0.33,
                iou_threshold=0.55,
                imgsz=640,
                device="cpu",
                execute=True,
                dry_run=False,
            )
            config = self.eval_module.maybe_load_yaml(args.config)
            calls = []

            class FakeYOLO:
                def __init__(self, model):
                    calls.append(("init", model))

                def val(self, **kwargs):
                    calls.append(("val", kwargs))
                    return SimpleNamespace(
                        save_dir=Path(kwargs["project"]) / kwargs["name"],
                        results_dict={"metrics/mAP50(B)": 0.7},
                        speed={"inference": 5.0},
                    )

            self.eval_module.load_yolo_class = lambda: FakeYOLO
            payload = self.eval_module.execute_evaluation(args, config)
            metrics_path = run_dir / "metrics.json"

            self.assertEqual(calls[0], ("init", str(weights.resolve())))
            self.assertEqual(calls[1][0], "val")
            self.assertEqual(calls[1][1]["imgsz"], 640)
            self.assertEqual(calls[1][1]["device"], "cpu")
            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["results_dict"]["metrics/mAP50(B)"], 0.7)
            self.assertEqual(payload["metrics_path"], str(metrics_path.resolve()))
            self.assertEqual(payload["status"], "executed")

    def test_infer_execute_writes_prediction_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "inputs"
            output_dir = tmp_path / "runs" / "infer"
            weights = tmp_path / "weights" / "best.pt"
            input_path.mkdir(parents=True, exist_ok=True)
            (input_path / "sample.jpg").write_bytes(b"fake-image")
            weights.parent.mkdir(parents=True, exist_ok=True)
            weights.write_bytes(b"fake-weights")

            args = Namespace(
                config=self.config_path,
                input=input_path,
                weights=weights,
                output_dir=output_dir,
                conf_threshold=0.4,
                imgsz=768,
                device="cpu",
                execute=True,
                dry_run=False,
            )
            config = self.infer_module.maybe_load_yaml(args.config)
            calls = []

            class FakeYOLO:
                def __init__(self, model):
                    calls.append(("init", model))

                def predict(self, **kwargs):
                    calls.append(("predict", kwargs))
                    return [
                        SimpleNamespace(path=str(input_path / "sample.jpg"), boxes=[1, 2]),
                        SimpleNamespace(path=str(input_path / "other.jpg"), boxes=[]),
                    ]

            self.infer_module.load_yolo_class = lambda: FakeYOLO
            payload = self.infer_module.execute_inference(args, config)
            summary_path = output_dir / "predictions.json"

            self.assertEqual(calls[0], ("init", str(weights.resolve())))
            self.assertEqual(calls[1][0], "predict")
            self.assertEqual(calls[1][1]["imgsz"], 768)
            self.assertEqual(calls[1][1]["device"], "cpu")
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["prediction_count"], 2)
            self.assertEqual(summary["predictions"][0]["box_count"], 2)
            self.assertEqual(payload["output_dir"], str(output_dir.resolve()))
            self.assertEqual(payload["summary_path"], str(summary_path.resolve()))


if __name__ == "__main__":
    unittest.main()
