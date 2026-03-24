import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "detect" / "scripts" / "run_detection_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_detection_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class RunDetectionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.repo_root = Path(__file__).resolve().parents[1]
        self.config_path = self.repo_root / "bolt" / "detect" / "configs" / "baseline.yaml"

    def create_voc_fixture(self, root: Path) -> tuple[Path, Path]:
        images_dir = root / "images"
        annotations_dir = root / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / "sample_a.jpg").write_bytes(b"fake-a")
        (images_dir / "sample_b.jpg").write_bytes(b"fake-b")
        xml_template = """<annotation>
    <filename>{filename}</filename>
    <size><width>400</width><height>300</height><depth>3</depth></size>
    <object>
        <name>{label}</name>
        <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>80</xmax><ymax>120</ymax></bndbox>
    </object>
</annotation>
"""
        write_text(annotations_dir / "sample_a.xml", xml_template.format(filename="sample_a.jpg", label="faultScrew"))
        write_text(annotations_dir / "sample_b.xml", xml_template.format(filename="sample_b.jpg", label="normScrew"))
        return images_dir, annotations_dir

    def test_run_pipeline_dry_run_emits_prepare_and_train_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images_dir, annotations_dir = self.create_voc_fixture(tmp_path)
            args = Namespace(
                images_dir=images_dir,
                annotations=annotations_dir,
                prepared_root=tmp_path / "prepared",
                run_root=tmp_path / "runs",
                config=self.config_path,
                metadata=None,
                group_field="sample_id",
                class_name="missing_fastener",
                include_label=["faultScrew"],
                copy_mode="copy",
                seed=42,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                weights="yolo11n.pt",
                epochs=5,
                batch_size=2,
                imgsz=640,
                device="cpu",
                conf_threshold=0.25,
                iou_threshold=0.5,
                infer_source=None,
                dry_run=True,
            )

            payload = self.module.run_pipeline(args)

            self.assertEqual(payload["status"], "dry-run")
            self.assertEqual(payload["prepare"]["source"]["include_labels"], ["faultScrew"])
            self.assertEqual(payload["prepare"]["resolved_sample_count"], 2)
            self.assertEqual(payload["train"]["task"], "train_baseline")
            self.assertIsNone(payload["eval"])
            self.assertIsNone(payload["infer"])

    def test_pipeline_builders_forward_imgsz_and_device_to_eval_and_infer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = Namespace(
                images_dir=tmp_path / "images",
                annotations=tmp_path / "annotations",
                prepared_root=tmp_path / "prepared",
                run_root=tmp_path / "runs",
                config=self.config_path,
                metadata=None,
                group_field="sample_id",
                class_name="missing_fastener",
                include_label=["faultScrew"],
                copy_mode="copy",
                seed=42,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                weights="yolo11n.pt",
                epochs=5,
                batch_size=2,
                imgsz=640,
                device="0",
                conf_threshold=0.25,
                iou_threshold=0.5,
                infer_source=None,
                dry_run=False,
            )
            weights_path = tmp_path / "best.pt"

            eval_args = self.module.make_eval_args(args, weights_path)
            infer_args = self.module.make_infer_args(args, weights_path)

            self.assertEqual(eval_args.imgsz, 640)
            self.assertEqual(eval_args.device, "0")
            self.assertEqual(infer_args.imgsz, 640)
            self.assertEqual(infer_args.device, "0")


if __name__ == "__main__":
    unittest.main()
