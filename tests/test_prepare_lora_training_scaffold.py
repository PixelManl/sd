import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolt" / "generate" / "scripts" / "prepare_lora_training_scaffold.py"
    spec = importlib.util.spec_from_file_location("prepare_lora_training_scaffold", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareLoraTrainingScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_build_training_command_contains_accelerate_and_dataset_inputs(self):
        plan = self.module.build_training_plan(
            dataset_root=Path("data/bolt/generate/lora/nut_semantic/source"),
            output_root=Path("data/bolt/generate/lora/nut_semantic/checkpoints/run_a"),
            pretrained_model_name_or_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            instance_prompt="close-up utility hardware ROI, threaded stud with exactly one weathered gray steel hex nut",
            resolution=1024,
            train_batch_size=1,
            learning_rate=1e-4,
            max_train_steps=1200,
            rank=16,
            seed=42,
        )

        command = plan["command"]
        self.assertEqual(command[0], "python")
        self.assertIn("-m", command)
        self.assertIn("accelerate.commands.launch", command)
        self.assertIn("--pretrained_model_name_or_path", command)
        self.assertIn("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", command)
        self.assertIn("--train_data_dir", command)
        self.assertIn("data/bolt/generate/lora/nut_semantic/source", command)
        self.assertIn("--output_dir", command)
        self.assertIn("--caption_column", command)
        self.assertIn("text", command)

    def test_materialize_training_scaffold_writes_plan_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            manifests_dir = dataset_root / "manifests"
            dataset_root.mkdir(parents=True, exist_ok=True)
            manifests_dir.mkdir(parents=True, exist_ok=True)
            (manifests_dir / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
            (dataset_root / "metadata.jsonl").write_text('{"file_name":"images/sample_a.png","text":"nut"}\n', encoding="utf-8")

            output_root = root / "checkpoints" / "run_a"
            payload = self.module.materialize_training_scaffold(
                dataset_root=dataset_root,
                output_root=output_root,
                pretrained_model_name_or_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                instance_prompt="close-up utility hardware ROI, threaded stud with exactly one weathered gray steel hex nut",
                resolution=1024,
                train_batch_size=1,
                learning_rate=1e-4,
                max_train_steps=1200,
                rank=16,
                seed=42,
            )

            self.assertTrue((output_root / "plan" / "training_plan.json").exists())
            self.assertTrue((output_root / "plan" / "run_train_lora.bat").exists())
            self.assertTrue((output_root / "plan" / "run_train_lora.sh").exists())
            plan = json.loads((output_root / "plan" / "training_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["task"], "prepare_lora_training_scaffold")
            self.assertEqual(payload["output_root"], str(output_root.resolve()))


if __name__ == "__main__":
    unittest.main()
