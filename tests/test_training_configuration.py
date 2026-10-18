import unittest
from pathlib import Path

from training import build_checkpoint_path
from utils import (
    DataSplitType,
    LRSchedulingType,
    ModelArchitecture,
    SampleAmplificationMode,
)


class TrainingConfigurationTests(unittest.TestCase):
    def test_sample_amplification_is_part_of_checkpoint_identity(self):
        path = build_checkpoint_path(
            Path("outputs/checkpoints"),
            architecture=ModelArchitecture.CNN_TRANSFORMER_CSA,
            transformer_heads=4,
            transformer_expansion_factor=4,
            transformer_dropout=0.1,
            model_variant="outer_residual",
            run_name="full",
            patch_size=15,
            epochs=150,
            feature_channels=128,
            scheduling_type=LRSchedulingType.FIXED,
            split_seed=42,
            split_type=DataSplitType.RANDOM,
            training_seed=100,
            sample_amplification_mode=SampleAmplificationMode.CTA_NET,
        )

        self.assertTrue(path.name.endswith("_sa-cta_net.pt"))

    def test_ste_pruning_percentage_is_part_of_checkpoint_identity(self):
        common = {
            "architecture": (
                ModelArchitecture.CNN_TRANSFORMER_CSA_BAND_PRUNING
            ),
            "transformer_heads": 4,
            "transformer_expansion_factor": 4,
            "transformer_dropout": 0.1,
            "model_variant": "outer_residual",
            "run_name": "full",
            "patch_size": 15,
            "epochs": 150,
            "feature_channels": 128,
            "scheduling_type": LRSchedulingType.FIXED,
            "split_seed": 42,
            "split_type": DataSplitType.RANDOM,
            "training_seed": 100,
            "sample_amplification_mode": SampleAmplificationMode.CTA_NET,
        }
        five_percent_path = build_checkpoint_path(
            Path("outputs/checkpoints"),
            pruning_percentage=5.0,
            **common,
        )
        ten_percent_path = build_checkpoint_path(
            Path("outputs/checkpoints"),
            pruning_percentage=10.0,
            **common,
        )

        self.assertIn("_pr5pct", five_percent_path.name)
        self.assertIn("_pr10pct", ten_percent_path.name)
        self.assertNotEqual(five_percent_path, ten_percent_path)


if __name__ == "__main__":
    unittest.main()
