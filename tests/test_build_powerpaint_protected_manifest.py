import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "generate" / "scripts" / "build_powerpaint_protected_manifest.py"
    spec = importlib.util.spec_from_file_location("build_powerpaint_protected_manifest", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildPowerPaintProtectedManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_build_protected_manifest_writes_protect_and_paste_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "roi512_d0"
            (source_dir / "crops").mkdir(parents=True, exist_ok=True)
            (source_dir / "masks").mkdir(parents=True, exist_ok=True)
            (source_dir / "annotations").mkdir(parents=True, exist_ok=True)

            crop_path = source_dir / "crops" / "sample-crop.png"
            mask_path = source_dir / "masks" / "sample-mask.png"
            annotation_path = source_dir / "annotations" / "sample.xml"
            Image.new("RGB", (64, 64), color=(180, 180, 180)).save(crop_path)
            mask = Image.new("L", (64, 64), color=0)
            for x in range(20, 44):
                for y in range(24, 40):
                    mask.putpixel((x, y), 255)
            mask.save(mask_path)
            annotation_path.write_text(
                """<annotation>
  <filename>sample-crop.png</filename>
  <size><width>64</width><height>64</height><depth>3</depth></size>
  <object>
    <name>healthy_nut</name>
    <bndbox><xmin>20</xmin><ymin>24</ymin><xmax>44</xmax><ymax>40</ymax></bndbox>
  </object>
</annotation>
""",
                encoding="utf-8",
            )

            manifest_path = source_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "task": "roi512_d0",
                        "records": [
                            {
                                "image_path": str(crop_path),
                                "annotation_format": "voc",
                                "annotation_path": str(annotation_path),
                                "target_id": "sample-roi512_d0",
                                "output_stem": "sample-roi512_d0-out",
                                "bbox": [20, 24, 44, 40],
                                "class_name": "healthy_nut",
                                "mask_path": str(mask_path),
                                "powerpaint_checkpoint_dir": "/tmp/checkpoints",
                                "powerpaint_conda_prefix": "/tmp/venv",
                                "powerpaint_repo_dir": "/tmp/repo",
                                "steps": 8,
                                "guidance_scale": 7.5,
                                "seed": 123,
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            output_dir = root / "protected"
            summary = self.module.build_protected_manifest(
                source_manifest_path=manifest_path,
                output_dir=output_dir,
                variant_name="protect_v1",
                seam_px=2,
                context_ring_px=12,
                keep_hard_length_scale=1.2,
                keep_hard_width_scale=0.2,
            )

            self.assertEqual(summary["record_count"], 1)
            out_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            record = out_manifest["records"][0]
            self.assertTrue(Path(record["protect_mask_path"]).exists())
            self.assertTrue(Path(record["paste_mask_path"]).exists())
            self.assertEqual(record["strict_paste_seam_px"], 2)
            self.assertEqual(record["context_ring_px"], 12)
            self.assertEqual(record["keep_hard_length_scale"], 1.2)
            self.assertEqual(record["keep_hard_width_scale"], 0.2)


if __name__ == "__main__":
    unittest.main()
