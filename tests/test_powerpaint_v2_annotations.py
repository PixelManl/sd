import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "generate" / "powerpaint_v2_annotations.py"
    spec = importlib.util.spec_from_file_location("powerpaint_v2_annotations", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class PowerPaintV2AnnotationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_rewrite_annotation_removes_matching_voc_object_without_overwriting_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.xml"
            output_path = root / "out" / "edited.xml"
            write_text(
                source_path,
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

            summary = self.module.rewrite_annotation(
                annotation_format="voc",
                source_path=source_path,
                output_path=output_path,
                target={
                    "target_id": "sample-target-001",
                    "bbox": [10, 12, 30, 36],
                    "class_name": "missing_fastener",
                },
            )

            self.assertEqual(summary["removed_count"], 1)
            self.assertEqual(summary["target_id"], "sample-target-001")
            self.assertTrue(output_path.exists())

            source_root = ET.parse(source_path).getroot()
            output_root = ET.parse(output_path).getroot()
            self.assertEqual(len(source_root.findall("object")), 2)
            self.assertEqual(len(output_root.findall("object")), 1)
            remaining = output_root.findall("object")[0]
            self.assertEqual(remaining.findtext("name"), "missing_fastener")
            self.assertEqual(
                [
                    int(remaining.find("bndbox").findtext("xmin", "0")),
                    int(remaining.find("bndbox").findtext("ymin", "0")),
                    int(remaining.find("bndbox").findtext("xmax", "0")),
                    int(remaining.find("bndbox").findtext("ymax", "0")),
                ],
                [48, 52, 80, 92],
            )

    def test_rewrite_annotation_removes_matching_coco_annotation_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "instances.json"
            output_path = root / "out" / "instances.edited.json"
            payload = {
                "images": [
                    {"id": 1, "file_name": "sample.jpg", "width": 128, "height": 128},
                ],
                "annotations": [
                    {"id": 101, "image_id": 1, "category_id": 1, "bbox": [10, 12, 20, 24]},
                    {"id": 102, "image_id": 1, "category_id": 1, "bbox": [48, 52, 32, 40]},
                ],
                "categories": [
                    {"id": 1, "name": "missing_fastener"},
                ],
            }
            write_text(source_path, json.dumps(payload, ensure_ascii=False, indent=2))

            summary = self.module.rewrite_annotation(
                annotation_format="coco",
                source_path=source_path,
                output_path=output_path,
                target={
                    "target_id": "sample-target-002",
                    "annotation_id": 101,
                    "image_id": 1,
                },
            )

            self.assertEqual(summary["removed_count"], 1)
            self.assertTrue(output_path.exists())

            source_payload = json.loads(source_path.read_text(encoding="utf-8"))
            output_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(source_payload["annotations"]), 2)
            self.assertEqual(len(output_payload["annotations"]), 1)
            self.assertEqual(output_payload["annotations"][0]["id"], 102)
            self.assertEqual(len(output_payload["images"]), 1)
            self.assertEqual(len(output_payload["categories"]), 1)


if __name__ == "__main__":
    unittest.main()
