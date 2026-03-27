from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bolt.generate.scripts.run_first10_missing_patch_batch import main as batch_main


class RunFirst10MissingPatchBatchTests(unittest.TestCase):
    def test_batch_runner_invokes_single_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            output_dir = root / "out"
            single_runner = root / "single.py"
            single_runner.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "out = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'preview_same_image_missing_patch.png').write_text('ok', encoding='utf-8')\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "asset_id": "sample-001",
                                "image_path": str(root / "img.jpg"),
                                "target_mask_path": str(root / "mask.png"),
                                "donor_box": [1, 2, 3, 4],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "img.jpg").write_bytes(b"img")
            (root / "mask.png").write_bytes(b"mask")

            import sys

            argv = sys.argv
            try:
                sys.argv = [
                    "run_first10_missing_patch_batch.py",
                    "--manifest-path",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--runner-python",
                    sys.executable,
                    "--single-runner",
                    str(single_runner),
                ]
                rc = batch_main()
            finally:
                sys.argv = argv

            self.assertEqual(rc, 0)
            summary = json.loads((output_dir / "run_results.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertTrue((output_dir / "sample-001" / "preview_same_image_missing_patch.png").exists())


if __name__ == "__main__":
    unittest.main()
