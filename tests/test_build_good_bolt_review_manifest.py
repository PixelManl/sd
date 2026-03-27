import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "scripts" / "build_good_bolt_review_manifest.py"
    spec = importlib.util.spec_from_file_location("build_good_bolt_review_manifest", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildGoodBoltReviewManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.original_repo_root = self.module.REPO_ROOT

    def tearDown(self) -> None:
        self.module.REPO_ROOT = self.original_repo_root

    def create_workspace(self, root: Path) -> Path:
        workspace = root / "good_bolt_assets"
        (workspace / "incoming" / "images").mkdir(parents=True, exist_ok=True)
        (workspace / "sam2" / "metadata").mkdir(parents=True, exist_ok=True)
        (workspace / "sam2" / "core_masks").mkdir(parents=True, exist_ok=True)
        (workspace / "sam2" / "edit_masks").mkdir(parents=True, exist_ok=True)
        (workspace / "sam2" / "overlays").mkdir(parents=True, exist_ok=True)
        (workspace / "manual_labels" / "healthy_labelme_json").mkdir(parents=True, exist_ok=True)
        return workspace

    def test_build_review_manifest_resolves_local_paths_and_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = self.create_workspace(Path(tmpdir))
            image_path = workspace / "incoming" / "images" / "sample_a.jpg"
            image_path.write_bytes(b"fake-image")
            labelme_path = workspace / "manual_labels" / "healthy_labelme_json" / "sample_a.json"
            labelme_path.write_text("{}", encoding="utf-8")
            core_mask_path = workspace / "sam2" / "core_masks" / "asset_001.png"
            edit_mask_path = workspace / "sam2" / "edit_masks" / "asset_001.png"
            overlay_path = workspace / "sam2" / "overlays" / "asset_001.png"
            core_mask_path.write_bytes(b"core")
            edit_mask_path.write_bytes(b"edit")
            overlay_path.write_bytes(b"overlay")

            metadata = {
                "asset_id": "asset_001",
                "roi_id": "sample_a-manual-001",
                "qa_state": "draft",
                "target_class": "healthy_bolt",
                "image_name": "sample_a.jpg",
                "source_image": "/root/sd-main/data/bolt_parallel/good_bolt_assets/incoming/images/sample_a.jpg",
                "box_xyxy": [10, 20, 30, 40],
                "core_mask_path": "/root/sd-main/data/bolt_parallel/good_bolt_assets/sam2/core_masks/asset_001.png",
                "edit_mask_path": "/root/sd-main/data/bolt_parallel/good_bolt_assets/sam2/edit_masks/asset_001.png",
                "overlay_path": "/root/sd-main/data/bolt_parallel/good_bolt_assets/sam2/overlays/asset_001.png",
            }
            (workspace / "sam2" / "metadata" / "asset_001.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            summary = self.module.build_review_manifest(workspace)

            manifest_path = workspace / "manifests" / "good_bolt_assets_manifest.jsonl"
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["manifest_path"], str(manifest_path.resolve()))
            self.assertTrue(manifest_path.exists())

            lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["asset_id"], "asset_001")
            self.assertEqual(record["source_image"], str(image_path.resolve()))
            self.assertEqual(record["labelme_json_path"], str(labelme_path.resolve()))
            self.assertEqual(record["core_mask_path"], str(core_mask_path.resolve()))
            self.assertEqual(record["edit_mask_path"], str(edit_mask_path.resolve()))
            self.assertEqual(record["overlay_path"], str(overlay_path.resolve()))

    def test_build_review_manifest_keeps_missing_labelme_path_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = self.create_workspace(Path(tmpdir))
            image_path = workspace / "incoming" / "images" / "sample_b.jpg"
            image_path.write_bytes(b"fake-image")
            (workspace / "sam2" / "core_masks" / "asset_002.png").write_bytes(b"core")
            (workspace / "sam2" / "edit_masks" / "asset_002.png").write_bytes(b"edit")
            (workspace / "sam2" / "overlays" / "asset_002.png").write_bytes(b"overlay")
            metadata = {
                "asset_id": "asset_002",
                "roi_id": "sample_b-manual-001",
                "qa_state": "draft",
                "target_class": "healthy_bolt",
                "image_name": "sample_b.jpg",
                "source_image": "/remote/sample_b.jpg",
                "box_xyxy": [1, 2, 3, 4],
                "core_mask_path": "/remote/asset_002_core.png",
                "edit_mask_path": "/remote/asset_002_edit.png",
                "overlay_path": "/remote/asset_002_overlay.png",
            }
            (workspace / "sam2" / "metadata" / "asset_002.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            summary = self.module.build_review_manifest(workspace)

            lines = [
                json.loads(line)
                for line in (workspace / "manifests" / "good_bolt_assets_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(lines[0]["labelme_json_path"], "")
            self.assertEqual(lines[0]["source_image"], str(image_path.resolve()))

    def test_build_review_manifest_falls_back_to_repo_local_image_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            workspace = self.create_workspace(temp_root)
            self.module.REPO_ROOT = temp_root
            mirror_image = (
                temp_root
                / "data"
                / "bolt_parallel"
                / "good_bolt_assets"
                / "incoming"
                / "images"
                / "sample_c.jpg"
            )
            mirror_image.parent.mkdir(parents=True, exist_ok=True)
            mirror_image.write_bytes(b"fake-image")
            (workspace / "sam2" / "core_masks" / "asset_003.png").write_bytes(b"core")
            (workspace / "sam2" / "edit_masks" / "asset_003.png").write_bytes(b"edit")
            (workspace / "sam2" / "overlays" / "asset_003.png").write_bytes(b"overlay")
            metadata = {
                "asset_id": "asset_003",
                "roi_id": "sample_c-manual-001",
                "qa_state": "draft",
                "target_class": "healthy_bolt",
                "image_name": "sample_c.jpg",
                "source_image": "/remote/sample_c.jpg",
                "box_xyxy": [5, 6, 7, 8],
                "core_mask_path": "/remote/asset_003_core.png",
                "edit_mask_path": "/remote/asset_003_edit.png",
                "overlay_path": "/remote/asset_003_overlay.png",
            }
            (workspace / "sam2" / "metadata" / "asset_003.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            summary = self.module.build_review_manifest(workspace)

            lines = [
                json.loads(line)
                for line in (workspace / "manifests" / "good_bolt_assets_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(lines[0]["source_image"], str(mirror_image.resolve()))


if __name__ == "__main__":
    unittest.main()
