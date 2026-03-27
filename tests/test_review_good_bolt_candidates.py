import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "scripts" / "review_good_bolt_candidates.py"
    spec = importlib.util.spec_from_file_location("review_good_bolt_candidates", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReviewGoodBoltCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def create_workspace(self, root: Path) -> Path:
        workspace = root / "good_bolt_assets"
        (workspace / "incoming" / "images").mkdir(parents=True, exist_ok=True)
        (workspace / "dino" / "boxes_json").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (120, 120, 120)).save(
            workspace / "incoming" / "images" / "sample_a.jpg"
        )
        Image.new("RGB", (32, 32), (90, 90, 90)).save(
            workspace / "incoming" / "images" / "sample_b.png"
        )
        (workspace / "dino" / "boxes_json" / "sample_a.json").write_text(
            json.dumps(
                {
                    "image": "sample_a.jpg",
                    "boxes": [
                        {"box_xyxy": [4, 5, 18, 24], "score": 0.93, "label": "healthy_bolt"}
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return workspace

    def test_build_review_summary_reports_box_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.create_workspace(Path(tmp))

            summary = self.module.build_review_summary(workspace)

            self.assertEqual(summary["image_count"], 2)
            self.assertEqual(summary["with_boxes"], 1)
            self.assertEqual(summary["missing_boxes"], 1)
            self.assertEqual(summary["records"][0]["image_name"], "sample_a.jpg")


if __name__ == "__main__":
    unittest.main()
