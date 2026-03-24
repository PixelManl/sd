import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "package" / "scripts" / "materialize_submission_scaffold.py"
    spec = importlib.util.spec_from_file_location("materialize_submission_scaffold", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MaterializeSubmissionScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def create_healthy_fixture(self, root: Path) -> Path:
        healthy_root = root / "good_bolt_assets"
        for relative in (
            Path("incoming/images"),
            Path("manual_labels/healthy_labelme_json"),
            Path("sam2/metadata"),
            Path("sam2/core_masks"),
            Path("sam2/edit_masks"),
            Path("sam2/overlays"),
            Path("exports/healthy_roi_bank"),
            Path("exports/qa_lists"),
        ):
            (healthy_root / relative).mkdir(parents=True, exist_ok=True)

        (healthy_root / "incoming" / "images" / "sample_a.jpg").write_bytes(b"fake")
        (healthy_root / "incoming" / "images" / "sample_b.png").write_bytes(b"fake")
        (healthy_root / "manual_labels" / "healthy_labelme_json" / "sample_a.json").write_text(
            '{"shapes":[{"label":"GG"},{"label":"GN"}]}',
            encoding="utf-8",
        )
        (healthy_root / "sam2" / "metadata" / "asset_001.json").write_text("{}", encoding="utf-8")
        for relative in (
            Path("sam2/core_masks/asset_001.png"),
            Path("sam2/edit_masks/asset_001.png"),
            Path("sam2/overlays/asset_001.png"),
            Path("exports/healthy_roi_bank/asset_001.png"),
            Path("exports/qa_lists/usable.txt"),
        ):
            (healthy_root / relative).write_bytes(b"fake")
        return healthy_root

    def test_build_inventory_counts_assets_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            healthy_root = self.create_healthy_fixture(tmp_path)

            inventory = self.module.build_inventory(healthy_root)

            self.assertEqual(inventory["incoming_image_count"], 2)
            self.assertEqual(inventory["manual_label_file_count"], 1)
            self.assertEqual(inventory["manual_label_box_counts"]["GG"], 1)
            self.assertEqual(inventory["manual_label_box_counts"]["GN"], 1)
            self.assertEqual(inventory["sam2_asset_counts"]["metadata"], 1)
            self.assertEqual(inventory["healthy_roi_bank_count"], 1)
            self.assertEqual(inventory["qa_list_count"], 1)

    def test_materialize_scaffold_creates_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            healthy_root = self.create_healthy_fixture(tmp_path)
            build_root = tmp_path / "build"
            tag = "2026-03-24-upload-round-01"
            build_dir = build_root / tag

            layout = self.module.materialize_scaffold(build_dir)
            inventory = self.module.build_inventory(healthy_root)
            inventory_json = build_dir / "checks" / "asset_inventory.json"
            inventory_md = build_dir / "checks" / "asset_inventory.md"
            upload_manifest = build_dir / "checks" / "upload_manifest.csv"
            checklist_md = build_dir / "checks" / "submission_checklist.md"

            inventory_json.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.module.write_markdown(inventory_md, inventory, "team", tag)
            self.module.write_upload_manifest(upload_manifest, inventory)
            self.module.write_submission_checklist(checklist_md, "team")

            self.assertIn("model_pkg", layout)
            self.assertTrue((build_dir / "dataset_pkg" / "DataFiles").exists())
            self.assertTrue((build_dir / "model_pkg" / "code").exists())
            self.assertTrue(inventory_json.exists())
            self.assertTrue(inventory_md.exists())
            self.assertTrue(upload_manifest.exists())
            self.assertTrue(checklist_md.exists())


if __name__ == "__main__":
    unittest.main()
