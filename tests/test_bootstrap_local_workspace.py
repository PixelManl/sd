import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "scripts" / "bootstrap_local_workspace.py"
    spec = importlib.util.spec_from_file_location("bootstrap_local_workspace", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BootstrapLocalWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_build_layout_contains_expected_round_merge_dir(self):
        layout = self.module.build_layout(Path("data/bolt"), "2026-03-27")
        self.assertIn("detect/merged_20260327/images", layout["detect_flow"])
        self.assertIn("generate/sdxl/accepted/images", layout["sdxl_flow"])

    def test_materialize_layout_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = self.module.build_layout(root, "2026-03-27")
            created = self.module.materialize_layout(root, layout, dry_run=False)

            self.assertTrue((root / "generate" / "sdxl" / "accepted" / "images").exists())
            self.assertTrue((root / "detect" / "merged_20260327" / "annotations").exists())
            self.assertEqual(len(created), sum(len(paths) for paths in layout.values()))


if __name__ == "__main__":
    unittest.main()
