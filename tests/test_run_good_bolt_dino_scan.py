import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "scripts" / "run_good_bolt_dino_scan.py"
    spec = importlib.util.spec_from_file_location("run_good_bolt_dino_scan", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_backend_module(path: Path) -> None:
    path.write_text(
        """
def predictor(source_image, asset_context):
    name = asset_context["image_name"]
    if name == "sample_a.jpg":
        return [
            {"box_xyxy": [4, 5, 18, 24], "score": 0.93, "label": "healthy_bolt"},
            {"box_xyxy": [1, 1, 8, 8], "score": 0.21, "label": "healthy_bolt"},
        ]
    return []
""".strip(),
        encoding="utf-8",
    )


class RunGoodBoltDinoScanTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def create_workspace(self, root: Path) -> tuple[Path, Path]:
        workspace = root / "good_bolt_assets"
        images_dir = root / "server_defect_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        (workspace / "dino" / "boxes_json").mkdir(parents=True, exist_ok=True)

        Image.new("RGB", (32, 32), (120, 120, 120)).save(images_dir / "sample_a.jpg")
        Image.new("RGB", (24, 24), (90, 90, 90)).save(images_dir / "sample_b.png")
        return workspace, images_dir

    def test_build_scan_plan_supports_external_image_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace, images_dir = self.create_workspace(Path(tmp))

            plan = self.module.build_scan_plan(workspace=workspace, input_dir=images_dir)

            self.assertEqual(plan["image_count"], 2)
            self.assertEqual(plan["records"][0]["image_name"], "sample_a.jpg")
            self.assertEqual(plan["records"][0]["source_image"], str(images_dir / "sample_a.jpg"))

    def test_execute_scan_plan_writes_boxes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace, images_dir = self.create_workspace(tmp_path)
            backend_path = tmp_path / "good_bolt_dino_backend.py"
            write_backend_module(backend_path)

            plan = self.module.build_scan_plan(workspace=workspace, input_dir=images_dir)
            backend = self.module.resolve_backend(f"{backend_path}:predictor")
            summary = self.module.execute_scan_plan(
                plan,
                workspace,
                backend,
                min_score=0.5,
                top_k=5,
                target_mode="all",
            )

            self.assertEqual(summary["generated_json_count"], 2)

            sample_a_path = workspace / "dino" / "boxes_json" / "sample_a.json"
            payload = json.loads(sample_a_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["image"], "sample_a.jpg")
            self.assertEqual(payload["source_image"], str(images_dir / "sample_a.jpg"))
            self.assertEqual(len(payload["boxes"]), 1)
            self.assertEqual(payload["boxes"][0]["box_xyxy"], [4, 5, 18, 24])
            self.assertEqual(payload["boxes"][0]["label"], "healthy_bolt")

            sample_b_path = workspace / "dino" / "boxes_json" / "sample_b.json"
            payload_b = json.loads(sample_b_path.read_text(encoding="utf-8"))
            self.assertEqual(payload_b["boxes"], [])

    def test_build_grounding_postprocess_kwargs_supports_threshold_alias(self):
        class LegacyProcessor:
            def post_process_grounded_object_detection(
                self,
                outputs,
                input_ids=None,
                threshold=0.25,
                text_threshold=0.25,
                target_sizes=None,
            ):
                return []

        kwargs = self.module.build_grounding_postprocess_kwargs(
            LegacyProcessor(),
            box_threshold=0.31,
            text_threshold=0.19,
            target_sizes=[(32, 32)],
        )

        self.assertEqual(kwargs["threshold"], 0.31)
        self.assertNotIn("box_threshold", kwargs)

    def test_build_grounding_postprocess_kwargs_supports_box_threshold(self):
        class NewProcessor:
            def post_process_grounded_object_detection(
                self,
                outputs,
                input_ids=None,
                box_threshold=0.25,
                text_threshold=0.25,
                target_sizes=None,
            ):
                return []

        kwargs = self.module.build_grounding_postprocess_kwargs(
            NewProcessor(),
            box_threshold=0.31,
            text_threshold=0.19,
            target_sizes=[(32, 32)],
        )

        self.assertEqual(kwargs["box_threshold"], 0.31)
        self.assertNotIn("threshold", kwargs)

    def test_filter_vertical_candidates_removes_large_sideview_components(self):
        boxes = [
            {
                "box_xyxy": [502, 14, 1884, 2990],
                "score": 0.28,
                "label": "healthy_bolt",
                "matched_label": "vertical hanging threaded stud threaded rod end",
            },
            {
                "box_xyxy": [2541, 450, 2733, 626],
                "score": 0.23,
                "label": "healthy_bolt",
                "matched_label": "downward bolt screw thread",
            },
            {
                "box_xyxy": [2476, 12, 3267, 727],
                "score": 0.25,
                "label": "healthy_bolt",
                "matched_label": "fastener side view",
            },
        ]

        filtered = self.module.filter_vertical_candidates(
            boxes,
            image_size=(4000, 3000),
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["matched_label"], "downward bolt screw thread")

    def test_filter_vertical_candidates_keeps_small_threaded_targets(self):
        boxes = [
            {
                "box_xyxy": [1211, 1818, 1376, 1988],
                "score": 0.29,
                "label": "healthy_bolt",
                "matched_label": "downward bolt thread",
            },
            {
                "box_xyxy": [1333, 2002, 1494, 2165],
                "score": 0.27,
                "label": "healthy_bolt",
                "matched_label": "downward bolt thread",
            },
        ]

        filtered = self.module.filter_vertical_candidates(
            boxes,
            image_size=(4000, 3000),
        )

        self.assertEqual(len(filtered), 2)

    def test_score_vertical_stud_shape_prefers_hanging_rod_over_round_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rod_path = tmp_path / "rod.png"
            round_path = tmp_path / "round.png"

            rod = Image.new("L", (120, 160), 245)
            for x in range(45, 75):
                for y in range(20, 45):
                    rod.putpixel((x, y), 80)
            for y in range(45, 145):
                band_value = 55 if ((y - 45) // 7) % 2 == 0 else 20
                for x in range(52, 68):
                    rod.putpixel((x, y), band_value)
            rod.save(rod_path)

            circle = Image.new("L", (120, 160), 245)
            for x in range(30, 90):
                for y in range(40, 100):
                    if (x - 60) ** 2 + (y - 70) ** 2 <= 28 ** 2:
                        circle.putpixel((x, y), 40)
            circle.save(round_path)

            rod_score = self.module.score_vertical_stud_shape(
                rod_path,
                [20, 10, 100, 150],
            )
            round_score = self.module.score_vertical_stud_shape(
                round_path,
                [20, 10, 100, 150],
            )

            self.assertGreater(rod_score["shape_score"], round_score["shape_score"])
            self.assertTrue(rod_score["passes_shape_gate"])
            self.assertFalse(round_score["passes_shape_gate"])

    def test_score_vertical_stud_shape_rejects_smooth_vertical_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            threaded_path = tmp_path / "threaded.png"
            smooth_path = tmp_path / "smooth.png"

            threaded = Image.new("L", (160, 200), 245)
            for x in range(58, 102):
                for y in range(28, 44):
                    threaded.putpixel((x, y), 90)
            for y in range(44, 170):
                band_value = 55 if ((y - 44) // 8) % 2 == 0 else 20
                for x in range(68, 92):
                    threaded.putpixel((x, y), band_value)
            threaded.save(threaded_path)

            smooth = Image.new("L", (160, 200), 245)
            for x in range(63, 97):
                for y in range(28, 170):
                    smooth_value = 120 if 74 <= x < 86 else 170
                    smooth.putpixel((x, y), smooth_value)
            smooth.save(smooth_path)

            threaded_score = self.module.score_vertical_stud_shape(
                threaded_path,
                [30, 20, 130, 180],
            )
            smooth_score = self.module.score_vertical_stud_shape(
                smooth_path,
                [30, 20, 130, 180],
            )

            self.assertTrue(threaded_score["passes_shape_gate"])
            self.assertFalse(smooth_score["passes_shape_gate"])
            self.assertGreater(
                threaded_score["thread_edge_ratio"],
                smooth_score["thread_edge_ratio"],
            )

    def test_score_vertical_stud_shape_rejects_near_square_round_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            round_path = tmp_path / "round_textured.png"

            image = Image.new("L", (160, 160), 220)
            center_x = 80
            center_y = 60
            radius = 34
            for x in range(20, 140):
                for y in range(10, 150):
                    dx = x - center_x
                    dy = y - center_y
                    if dx * dx + dy * dy <= radius * radius:
                        image.putpixel((x, y), 85)
                    if 50 <= x <= 110 and 90 <= y <= 148:
                        band_value = 45 if ((y - 90) // 4) % 2 == 0 else 100
                        image.putpixel((x, y), band_value)
            image.save(round_path)

            shape = self.module.score_vertical_stud_shape(
                round_path,
                [20, 10, 140, 150],
            )

            self.assertFalse(shape["passes_shape_gate"])

    def test_score_vertical_stud_shape_keeps_short_hanging_screw(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            short_path = tmp_path / "short_hanging.png"

            image = Image.new("L", (120, 120), 240)
            for x in range(25, 89):
                for y in range(10, 24):
                    image.putpixel((x, y), 100)
            for x in range(35, 79):
                for y in range(24, 42):
                    image.putpixel((x, y), 70)
            for y in range(42, 78):
                band_value = 35 if ((y - 42) // 5) % 2 == 0 else 80
                for x in range(53, 67):
                    image.putpixel((x, y), band_value)
            image.save(short_path)

            shape = self.module.score_vertical_stud_shape(
                short_path,
                [25, 10, 95, 80],
            )

            self.assertTrue(shape["passes_shape_gate"])

    def test_filter_vertical_candidates_accepts_plain_screw_label_when_shape_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "short_hanging.png"

            image = Image.new("L", (120, 120), 240)
            for x in range(25, 89):
                for y in range(10, 24):
                    image.putpixel((x, y), 100)
            for x in range(35, 79):
                for y in range(24, 42):
                    image.putpixel((x, y), 70)
            for y in range(42, 78):
                band_value = 35 if ((y - 42) // 5) % 2 == 0 else 80
                for x in range(53, 67):
                    image.putpixel((x, y), band_value)
            image.save(image_path)

            filtered = self.module.filter_vertical_candidates(
                [
                    {
                        "box_xyxy": [25, 10, 95, 80],
                        "score": 0.31,
                        "label": "healthy_bolt",
                        "matched_label": "bolt screw",
                    }
                ],
                image_size=(4000, 3000),
                source_image=image_path,
            )

            self.assertEqual(len(filtered), 1)


if __name__ == "__main__":
    unittest.main()
