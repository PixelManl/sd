import importlib.util
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "generate" / "scripts" / "run_powerpaint_v2_batch.py"
    spec = importlib.util.spec_from_file_location("run_powerpaint_v2_batch", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class RunPowerPaintV2BatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def create_voc_fixture(self, root: Path) -> tuple[Path, Path]:
        image_path = root / "images" / "sample.jpg"
        annotation_path = root / "annotations" / "sample.xml"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 128), color=(120, 130, 140)).save(image_path)
        write_text(
            annotation_path,
            """<annotation>
    <filename>sample.jpg</filename>
    <size><width>128</width><height>128</height><depth>3</depth></size>
    <object>
        <name>missing_fastener</name>
        <bndbox><xmin>10</xmin><ymin>12</ymin><xmax>30</xmax><ymax>36</ymax></bndbox>
    </object>
    <object>
        <name>missing_fastener</name>
        <bndbox><xmin>48</xmin><ymin>52</ymin><xmax>80</xmax><ymax>92</ymax></bndbox>
    </object>
</annotation>
""",
        )
        return image_path, annotation_path

    def test_run_batch_dry_run_supports_list_manifest_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path, annotation_path = self.create_voc_fixture(root)
            manifest_path = root / "manifest.json"
            output_dir = root / "outputs"
            write_text(
                manifest_path,
                json.dumps(
                    [
                        {
                            "image_path": str(image_path),
                            "annotation_format": "voc",
                            "annotation_path": str(annotation_path),
                            "target_id": "target-001",
                            "output_stem": "sample_step_001",
                            "bbox": [10, 12, 30, 36],
                            "class_name": "missing_fastener",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            summary = self.module.run_batch(
                manifest_path=manifest_path,
                output_dir=output_dir,
                backend_mode="placeholder-copy",
                dry_run=True,
            )

            self.assertEqual(summary["status"], "dry-run")
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["records"][0]["status"], "dry-run")
            self.assertTrue((output_dir / "manifest_results.json").exists())

    def test_run_batch_reuses_latest_outputs_for_same_image_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path, annotation_path = self.create_voc_fixture(root)
            manifest_path = root / "manifest.json"
            output_dir = root / "outputs"
            write_text(
                manifest_path,
                json.dumps(
                    {
                        "records": [
                            {
                                "image_path": str(image_path),
                                "annotation_format": "voc",
                                "annotation_path": str(annotation_path),
                                "target_id": "target-001",
                                "output_stem": "sample_step_001",
                                "bbox": [10, 12, 30, 36],
                                "class_name": "missing_fastener",
                            },
                            {
                                "image_path": str(image_path),
                                "annotation_format": "voc",
                                "annotation_path": str(annotation_path),
                                "target_id": "target-002",
                                "output_stem": "sample_step_002",
                                "bbox": [48, 52, 80, 92],
                                "class_name": "missing_fastener",
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            summary = self.module.run_batch(
                manifest_path=manifest_path,
                output_dir=output_dir,
                backend_mode="placeholder-copy",
                dry_run=False,
            )

            self.assertEqual(summary["status"], "completed", msg=json.dumps(summary, ensure_ascii=False, indent=2))
            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(summary["records"][0]["status"], "completed")
            self.assertEqual(summary["records"][1]["status"], "completed")
            self.assertEqual(
                summary["records"][1]["annotation_before"],
                summary["records"][0]["annotation_after"],
            )
            self.assertTrue(Path(summary["records"][0]["edited_image"]).exists())
            self.assertTrue(Path(summary["records"][1]["edited_image"]).exists())

            final_annotation = Path(summary["records"][1]["annotation_after"])
            objects = ET.parse(final_annotation).getroot().findall("object")
            self.assertEqual(len(objects), 0)

    def test_run_batch_supports_real_backend_smoke_when_enabled(self) -> None:
        if os.environ.get("POWERPAINT_SMOKE") != "1":
            self.skipTest("POWERPAINT_SMOKE is not enabled")

        checkpoint_dir = os.environ.get("POWERPAINT_CHECKPOINT_DIR", "").strip()
        conda_prefix = os.environ.get("POWERPAINT_CONDA_PREFIX", "").strip()
        if not checkpoint_dir or not conda_prefix:
            self.skipTest("POWERPAINT_CHECKPOINT_DIR or POWERPAINT_CONDA_PREFIX missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "images" / "sample.jpg"
            annotation_path = root / "annotations" / "sample.xml"
            manifest_path = root / "manifest.json"
            output_dir = root / "outputs"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), color=(160, 170, 180)).save(image_path)
            write_text(
                annotation_path,
                """<annotation>
    <filename>sample.jpg</filename>
    <size><width>128</width><height>128</height><depth>3</depth></size>
    <object>
        <name>missing_fastener</name>
        <bndbox><xmin>42</xmin><ymin>44</ymin><xmax>86</xmax><ymax>88</ymax></bndbox>
    </object>
</annotation>
""",
            )
            write_text(
                manifest_path,
                json.dumps(
                    {
                        "records": [
                            {
                                "image_path": str(image_path),
                                "annotation_format": "voc",
                                "annotation_path": str(annotation_path),
                                "target_id": "target-001",
                                "output_stem": "sample_step_001",
                                "bbox": [42, 44, 86, 88],
                                "class_name": "missing_fastener",
                                "powerpaint_checkpoint_dir": checkpoint_dir,
                                "powerpaint_conda_prefix": conda_prefix,
                                "mask_box": [42, 44, 86, 88],
                                "steps": 4,
                                "guidance_scale": 7.5,
                                "seed": 123,
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            summary = self.module.run_batch(
                manifest_path=manifest_path,
                output_dir=output_dir,
                backend_mode="powerpaint-v2-1-offline",
                dry_run=False,
            )

            self.assertEqual(summary["status"], "completed", msg=json.dumps(summary, ensure_ascii=False, indent=2))
            self.assertEqual(summary["records"][0]["status"], "completed")
            self.assertTrue(Path(summary["records"][0]["edited_image"]).exists())
            objects = ET.parse(Path(summary["records"][0]["annotation_after"])).getroot().findall("object")
            self.assertEqual(len(objects), 0)

    def test_run_batch_preserves_mask_path_for_backend_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path, annotation_path = self.create_voc_fixture(root)
            mask_path = root / "masks" / "sample-mask.png"
            manifest_path = root / "manifest.json"
            output_dir = root / "outputs"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (128, 128), color=0).save(mask_path)
            write_text(
                manifest_path,
                json.dumps(
                    {
                        "records": [
                            {
                                "image_path": str(image_path),
                                "annotation_format": "voc",
                                "annotation_path": str(annotation_path),
                                "target_id": "target-mask-001",
                                "output_stem": "sample_step_mask_001",
                                "bbox": [10, 12, 30, 36],
                                "class_name": "missing_fastener",
                                "mask_path": str(mask_path),
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            def fake_backend(*, source_image: Path, output_image: Path, record: dict, backend_mode: str) -> dict:
                self.assertEqual(record["mask_path"], str(mask_path.resolve()))
                output_image.parent.mkdir(parents=True, exist_ok=True)
                Image.open(source_image).save(output_image)
                return {
                    "backend_mode": backend_mode,
                    "source_image": str(source_image.resolve()),
                    "edited_image": str(output_image.resolve()),
                }

            with mock.patch.object(self.module, "run_backend", side_effect=fake_backend):
                summary = self.module.run_batch(
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                    backend_mode="placeholder-copy",
                    dry_run=False,
                )

            self.assertEqual(summary["status"], "completed", msg=json.dumps(summary, ensure_ascii=False, indent=2))
            self.assertEqual(summary["records"][0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
