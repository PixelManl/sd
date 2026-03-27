import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "scripts" / "export_gimp_xcf_masks.py"
    spec = importlib.util.spec_from_file_location("export_gimp_xcf_masks", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExportGimpXcfMasksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def create_workspace(self, root: Path) -> tuple[Path, Path]:
        xcf_dir = root / "xcf"
        output_dir = root / "masks"
        xcf_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return xcf_dir, output_dir

    def test_collect_export_jobs_pairs_xcf_to_mask_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xcf_dir, output_dir = self.create_workspace(Path(tmpdir))
            (xcf_dir / "asset_001.xcf").write_bytes(b"xcf")

            jobs = self.module.collect_export_jobs(
                xcf_dir=xcf_dir,
                output_dir=output_dir,
                layer_name="mask",
                overwrite=False,
                output_suffix="_mask",
            )

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["xcf_name"], "asset_001.xcf")
            self.assertEqual(jobs[0]["layer_name"], "mask")
            self.assertEqual(Path(jobs[0]["output_path"]), output_dir / "asset_001_mask.png")

    def test_collect_export_jobs_rejects_existing_output_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xcf_dir, output_dir = self.create_workspace(Path(tmpdir))
            (xcf_dir / "asset_002.xcf").write_bytes(b"xcf")
            (output_dir / "asset_002_mask.png").write_bytes(b"png")

            with self.assertRaises(ValueError):
                self.module.collect_export_jobs(
                    xcf_dir=xcf_dir,
                    output_dir=output_dir,
                    layer_name="mask",
                    overwrite=False,
                    output_suffix="_mask",
                )

    def test_write_batch_scheme_contains_layer_name_and_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xcf_dir, output_dir = self.create_workspace(Path(tmpdir))
            xcf_path = xcf_dir / "asset_003.xcf"
            xcf_path.write_bytes(b"xcf")
            jobs = self.module.collect_export_jobs(
                xcf_dir=xcf_dir,
                output_dir=output_dir,
                layer_name="mask",
                overwrite=True,
                output_suffix="_mask",
            )

            scheme_path = self.module.write_batch_scheme(
                output_dir=output_dir,
                jobs=jobs,
                layer_name="mask",
            )

            content = scheme_path.read_text(encoding="utf-8")
            self.assertIn("mask", content)
            self.assertIn("asset_003_mask.png", content)
            self.assertIn("sd-find-layer-by-name", content)
            self.assertIn("gimp-layer-set-mode export-layer 0", content)
            self.assertIn("gimp-layer-set-opacity export-layer 100.0", content)

    def test_postprocess_exported_masks_converts_rgba_to_binary_grayscale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            export_path = root / "asset_004_mask.png"
            image = Image.new("RGBA", (2, 2), color=(0, 0, 0, 0))
            image.putpixel((0, 0), (255, 255, 255, 255))
            image.putpixel((1, 0), (120, 120, 120, 255))
            image.putpixel((0, 1), (10, 10, 10, 255))
            image.putpixel((1, 1), (255, 0, 0, 255))
            image.save(export_path)

            self.module.postprocess_exported_masks([export_path], threshold=127)

            with Image.open(export_path) as processed:
                self.assertEqual(processed.mode, "L")
                self.assertEqual(processed.getpixel((0, 0)), 255)
                self.assertEqual(processed.getpixel((1, 0)), 0)
                self.assertEqual(processed.getpixel((0, 1)), 0)
                self.assertEqual(processed.getpixel((1, 1)), 0)


if __name__ == "__main__":
    unittest.main()
