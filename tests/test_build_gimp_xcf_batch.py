import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "scripts" / "build_gimp_xcf_batch.py"
    spec = importlib.util.spec_from_file_location("build_gimp_xcf_batch", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildGimpXcfBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def create_workspace(self, root: Path) -> tuple[Path, Path, Path]:
        image_dir = root / "images"
        mask_dir = root / "masks"
        output_dir = root / "xcf_out"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return image_dir, mask_dir, output_dir

    def test_collect_batch_jobs_pairs_images_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir, output_dir = self.create_workspace(Path(tmpdir))
            Image.new("RGB", (48, 32), color=(10, 20, 30)).save(image_dir / "sample_a.jpg")
            Image.new("L", (48, 32), color=255).save(mask_dir / "sample_a_mask.png")

            jobs = self.module.collect_batch_jobs(
                image_dir=image_dir,
                mask_dir=mask_dir,
                output_dir=output_dir,
                mask_suffix="_mask",
                overwrite=False,
                resize_mask_to_image=False,
            )

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["image_name"], "sample_a.jpg")
            self.assertEqual(jobs[0]["mask_name"], "sample_a_mask.png")
            self.assertEqual(jobs[0]["width"], 48)
            self.assertEqual(jobs[0]["height"], 32)
            self.assertEqual(Path(jobs[0]["xcf_path"]), output_dir / "sample_a.xcf")

    def test_collect_batch_jobs_can_resize_mask_into_staging_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir, output_dir = self.create_workspace(Path(tmpdir))
            Image.new("RGB", (64, 40), color=(10, 20, 30)).save(image_dir / "sample_b.jpg")
            Image.new("L", (16, 10), color=255).save(mask_dir / "sample_b_mask.png")

            jobs = self.module.collect_batch_jobs(
                image_dir=image_dir,
                mask_dir=mask_dir,
                output_dir=output_dir,
                mask_suffix="_mask",
                overwrite=False,
                resize_mask_to_image=True,
            )

            self.assertEqual(len(jobs), 1)
            staged_mask_path = Path(jobs[0]["mask_path"])
            self.assertTrue(staged_mask_path.exists())
            self.assertNotEqual(staged_mask_path, (mask_dir / "sample_b_mask.png").resolve())
            with Image.open(staged_mask_path) as staged_mask:
                self.assertEqual(staged_mask.size, (64, 40))

    def test_collect_batch_jobs_rejects_missing_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir, output_dir = self.create_workspace(Path(tmpdir))
            Image.new("RGB", (32, 32), color=(10, 20, 30)).save(image_dir / "sample_c.jpg")

            with self.assertRaises(ValueError):
                self.module.collect_batch_jobs(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    output_dir=output_dir,
                    mask_suffix="_mask",
                    overwrite=False,
                    resize_mask_to_image=False,
                )

    def test_write_batch_scheme_contains_mode_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir, output_dir = self.create_workspace(Path(tmpdir))
            image_path = image_dir / "sample_d.jpg"
            mask_path = mask_dir / "sample_d_mask.png"
            Image.new("RGB", (48, 32), color=(10, 20, 30)).save(image_path)
            Image.new("L", (48, 32), color=255).save(mask_path)

            jobs = self.module.collect_batch_jobs(
                image_dir=image_dir,
                mask_dir=mask_dir,
                output_dir=output_dir,
                mask_suffix="_mask",
                overwrite=False,
                resize_mask_to_image=False,
            )
            scheme_path = self.module.write_batch_scheme(
                output_dir=output_dir,
                jobs=jobs,
                background_layer_name="background",
                mask_layer_name="mask",
                background_opacity=100.0,
                background_mode_symbol=self.module.resolve_mode_symbol("lighten"),
                mask_opacity=55.0,
                mask_mode_symbol=self.module.resolve_mode_symbol("lighten"),
            )

            content = scheme_path.read_text(encoding="utf-8")
            self.assertIn("gimp-layer-set-mode background-layer 10", content)
            self.assertIn("gimp-layer-set-mode mask-layer 10", content)
            self.assertIn("background", content)
            self.assertIn("mask", content)
            self.assertIn("sample_d.xcf", content)

    def test_resolve_mode_symbol_supports_lighten(self) -> None:
        self.assertEqual(self.module.resolve_mode_symbol("lighten"), "10")


if __name__ == "__main__":
    unittest.main()
