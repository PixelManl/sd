import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "generate" / "scripts" / "prepare_lora_dataset_manifest.py"
    spec = importlib.util.spec_from_file_location("prepare_lora_dataset_manifest", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareLoraDatasetManifestTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_build_record_creates_local_caption_for_nut_semantics(self):
        record = self.module.build_record(
            image_path=Path("images/sample_a.png"),
            caption_template=(
                "close-up utility hardware ROI, threaded stud with exactly one weathered gray steel hex nut, "
                "nut seated tightly against metal plate"
            ),
            split="train",
        )

        self.assertEqual(record["image_relpath"], "images/sample_a.png")
        self.assertEqual(record["caption_txt_relpath"], "captions/sample_a.txt")
        self.assertEqual(record["split"], "train")
        self.assertIn("exactly one weathered gray steel hex nut", record["caption"])

    def test_materialize_manifest_writes_jsonl_and_caption_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / "sample_a.png").write_bytes(b"fake")
            (images_dir / "sample_b.png").write_bytes(b"fake")

            output_dir = root / "prepared"
            records = self.module.materialize_dataset_manifest(
                image_paths=[images_dir / "sample_a.png", images_dir / "sample_b.png"],
                output_dir=output_dir,
                caption_template=(
                    "close-up utility hardware ROI, threaded stud with exactly one weathered gray steel hex nut"
                ),
                train_ratio=0.5,
                seed=7,
            )

            manifest_path = output_dir / "manifests" / "dataset.jsonl"
            metadata_path = output_dir / "metadata.jsonl"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(metadata_path.exists())
            lines = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            metadata_lines = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(metadata_lines), 2)
            self.assertIn("file_name", metadata_lines[0])
            self.assertIn("text", metadata_lines[0])
            self.assertTrue((output_dir / "captions" / "sample_a.txt").exists())
            self.assertTrue((output_dir / "captions" / "sample_b.txt").exists())
            self.assertEqual(len(records), 2)
            self.assertEqual(sorted({item["split"] for item in records}), ["train", "val"])


if __name__ == "__main__":
    unittest.main()
