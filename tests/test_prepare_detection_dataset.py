import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "detect" / "scripts" / "prepare_detection_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_detection_dataset", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class PrepareDetectionDatasetTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.repo_root = Path(__file__).resolve().parents[1]
        self.config_path = self.repo_root / "bolt" / "detect" / "configs" / "baseline.yaml"

    def build_args(self, images_dir: Path, annotations: Path, output_dir: Path, **overrides):
        args = Namespace(
            images_dir=images_dir,
            annotations=annotations,
            output_dir=output_dir,
            config=self.config_path,
            class_name="missing_fastener",
            metadata=None,
            group_field="capture_group_id",
            train_ratio=0.5,
            val_ratio=0.5,
            test_ratio=0.0,
            copy_mode="manifest_only",
            seed=7,
            dry_run=True,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def create_voc_fixture(self, root: Path) -> tuple[Path, Path]:
        images_dir = root / "images"
        annotations_dir = root / "annotations"

        image_a = images_dir / "sample_a.jpg"
        image_b = images_dir / "sample_b.jpg"
        image_a.parent.mkdir(parents=True, exist_ok=True)
        image_a.write_bytes(b"fake-image-a")
        image_b.write_bytes(b"fake-image-b")

        xml_template = """<annotation>
    <filename>{filename}</filename>
    <size>
        <width>4000</width>
        <height>3000</height>
        <depth>3</depth>
    </size>
    <object>
        <name>07010032</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>
</annotation>
"""

        write_text(
            annotations_dir / "sample_a.xml",
            xml_template.format(
                filename="sample_a.jpg", xmin=100, ymin=200, xmax=300, ymax=500
            ),
        )
        write_text(
            annotations_dir / "sample_b.xml",
            xml_template.format(
                filename="sample_b.jpg", xmin=1200, ymin=1000, xmax=1500, ymax=1600
            ),
        )
        return images_dir, annotations_dir

    def create_metadata_json(self, path: Path, records: list[dict[str, str]]) -> Path:
        write_text(path, json.dumps(records, ensure_ascii=False, indent=2))
        return path

    def create_coco_fixture(self, root: Path) -> tuple[Path, Path]:
        images_dir = root / "images"
        annotations_path = root / "annotations" / "instances.json"
        image_a = images_dir / "scene_a" / "sample_a.jpg"
        image_b = images_dir / "scene_b" / "sample_b.jpg"
        image_a.parent.mkdir(parents=True, exist_ok=True)
        image_b.parent.mkdir(parents=True, exist_ok=True)
        image_a.write_bytes(b"fake-image-a")
        image_b.write_bytes(b"fake-image-b")

        payload = {
            "images": [
                {"id": 1, "file_name": "scene_a/sample_a.jpg", "width": 4000, "height": 3000},
                {"id": 2, "file_name": "scene_b/sample_b.jpg", "width": 4000, "height": 3000},
            ],
            "annotations": [
                {"id": 11, "image_id": 1, "category_id": 1, "bbox": [100, 200, 200, 300]},
                {"id": 12, "image_id": 2, "category_id": 1, "bbox": [1200, 1000, 300, 600]},
            ],
            "categories": [
                {"id": 1, "name": "bolt_missing"},
            ],
        }
        write_text(annotations_path, json.dumps(payload, ensure_ascii=False, indent=2))
        return images_dir, annotations_path

    def test_build_plan_supports_pascal_voc_annotation_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images_dir, annotations_dir = self.create_voc_fixture(tmp_path)
            args = self.build_args(images_dir, annotations_dir, tmp_path / "prepared")

            self.module.validate_args(args)
            plan = self.module.build_plan(args, config=None)

            self.assertEqual(plan["annotation_summary"]["format"], "pascal_voc_xml_dir")
            self.assertEqual(plan["annotation_summary"]["image_count"], 2)
            self.assertEqual(plan["annotation_summary"]["annotation_count"], 2)
            self.assertEqual(plan["annotation_summary"]["category_names"], ["missing_fastener"])
            self.assertEqual(plan["splits"], {"train": 1, "val": 1, "test": 0})

    def test_materialize_copy_mode_exports_yolo_labels_from_pascal_voc(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images_dir, annotations_dir = self.create_voc_fixture(tmp_path)
            output_dir = tmp_path / "prepared"
            args = self.build_args(
                images_dir,
                annotations_dir,
                output_dir,
                copy_mode="copy",
                dry_run=False,
            )

            self.module.validate_args(args)
            config = self.module.maybe_load_yaml(args.config)
            plan = self.module.build_plan(args, config)
            self.module.materialize_dataset(plan, args)

            manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["annotation_summary"]["format"], "pascal_voc_xml_dir")

            exported_images = sorted((output_dir / "images" / "train").glob("*.jpg")) + sorted(
                (output_dir / "images" / "val").glob("*.jpg")
            )
            exported_labels = sorted((output_dir / "labels" / "train").glob("*.txt")) + sorted(
                (output_dir / "labels" / "val").glob("*.txt")
            )

            self.assertEqual(len(exported_images), 2)
            self.assertEqual(len(exported_labels), 2)

            first_label = exported_labels[0].read_text(encoding="utf-8").strip()
            self.assertTrue(first_label.startswith("0 "))

    def test_group_split_keeps_same_capture_group_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images_dir, annotations_dir = self.create_voc_fixture(tmp_path)
            metadata_path = self.create_metadata_json(
                tmp_path / "metadata.json",
                [
                    {
                        "sample_id": "sample_a",
                        "scene_id": "scene-1",
                        "capture_group_id": "cap-1",
                    },
                    {
                        "sample_id": "sample_b",
                        "scene_id": "scene-2",
                        "capture_group_id": "cap-1",
                    },
                ],
            )
            output_dir = tmp_path / "prepared"
            args = self.build_args(
                images_dir,
                annotations_dir,
                output_dir,
                metadata=metadata_path,
                train_ratio=0.5,
                val_ratio=0.5,
                test_ratio=0.0,
                copy_mode="manifest_only",
                dry_run=False,
            )

            self.module.validate_args(args)
            config = self.module.maybe_load_yaml(args.config)
            plan = self.module.build_plan(args, config)
            self.module.materialize_dataset(plan, args)

            train_rows = json.loads((output_dir / "manifests" / "train.json").read_text(encoding="utf-8"))
            val_rows = json.loads((output_dir / "manifests" / "val.json").read_text(encoding="utf-8"))

            self.assertEqual(len(train_rows), 2)
            self.assertEqual(len(val_rows), 0)
            self.assertEqual({row["group_key"] for row in train_rows}, {"cap-1"})

    def test_materialize_copy_mode_exports_yolo_labels_from_coco_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images_dir, annotations_path = self.create_coco_fixture(tmp_path)
            output_dir = tmp_path / "prepared"
            args = self.build_args(
                images_dir,
                annotations_path,
                output_dir,
                copy_mode="copy",
                dry_run=False,
            )

            self.module.validate_args(args)
            config = self.module.maybe_load_yaml(args.config)
            plan = self.module.build_plan(args, config)
            self.module.materialize_dataset(plan, args)

            manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["annotation_summary"]["format"], "coco_json")

            exported_images = sorted((output_dir / "images" / "train").glob("*.jpg")) + sorted(
                (output_dir / "images" / "val").glob("*.jpg")
            )
            exported_labels = sorted((output_dir / "labels" / "train").glob("*.txt")) + sorted(
                (output_dir / "labels" / "val").glob("*.txt")
            )

            self.assertEqual(len(exported_images), 2)
            self.assertEqual(len(exported_labels), 2)
            self.assertIn("missing_fastener", (output_dir / "dataset.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
