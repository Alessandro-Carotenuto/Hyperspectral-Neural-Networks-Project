import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from models import (
    CNNTransformerCSABandPruningClassifier,
    CNNTransformerCSAClassifier,
)


class AdaptivePruningDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = CNNTransformerCSABandPruningClassifier(
            input_channels=6,
            feature_channels=8,
            num_classes=3,
            patch_size=5,
            transformer_heads=2,
            transformer_dropout=0.5,
        )
        self.model.train()
        self.patches = torch.randn(4, 6, 5, 5)

    def test_gate_pass_disables_dropout_without_blocking_gradients(self):
        with torch.no_grad():
            self.model.band_gate_head[-1].weight.normal_()

        transformer_batch_norms = [
            module
            for module in self.model.ct_block.transformer_branch.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        ]
        batch_counts_before = [
            module.num_batches_tracked.clone()
            for module in transformer_batch_norms
        ]

        rng_before = torch.random.get_rng_state()
        gates = self.model.predict_band_gates(self.patches)
        rng_after = torch.random.get_rng_state()

        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertTrue(self.model.training)
        self.assertTrue(
            all(
                module.training
                for module in self.model.ct_block.transformer_branch.modules()
                if isinstance(module, torch.nn.Dropout)
            )
        )
        self.assertTrue(
            all(
                torch.equal(before, module.num_batches_tracked)
                for before, module in zip(
                    batch_counts_before, transformer_batch_norms
                )
            )
        )

        gates.sum().backward()
        self.assertIsNotNone(self.model.band_gate_head[-1].weight.grad)
        self.assertIsNotNone(
            self.model.ct_block.transformer_branch
            .self_attention.qkv_projection.weight.grad
        )

    def test_classification_pass_keeps_dropout_active(self):
        self.model.predict_band_gates = lambda patches: torch.ones(
            patches.shape[0],
            self.model.input_channels,
            device=patches.device,
            dtype=patches.dtype,
        )

        torch.manual_seed(101)
        first_logits = self.model(self.patches)
        torch.manual_seed(202)
        second_logits = self.model(self.patches)

        self.assertFalse(torch.allclose(first_logits, second_logits))

    def test_all_one_gates_match_baseline_during_training(self):
        torch.manual_seed(31)
        baseline = CNNTransformerCSAClassifier(
            input_channels=6,
            feature_channels=8,
            num_classes=3,
            patch_size=5,
            transformer_heads=2,
            transformer_dropout=0.5,
        )
        self.model.load_state_dict(baseline.state_dict(), strict=False)
        baseline.train()
        self.model.train()

        predict_gates = self.model.predict_band_gates

        def predict_gates_then_return_ones(patches):
            predict_gates(patches)
            return torch.ones(
                patches.shape[0],
                self.model.input_channels,
                device=patches.device,
                dtype=patches.dtype,
            )

        self.model.predict_band_gates = predict_gates_then_return_ones
        targets = torch.tensor([0, 1, 2, 1])
        criterion = torch.nn.CrossEntropyLoss()

        torch.manual_seed(43)
        baseline_logits = baseline(self.patches)
        torch.manual_seed(43)
        adaptive_logits = self.model(self.patches)
        self.assertTrue(torch.equal(baseline_logits, adaptive_logits))

        baseline.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        criterion(baseline_logits, targets).backward()
        (
            criterion(adaptive_logits, targets)
            + self.model.pruning_regularization(self.patches)
        ).backward()

        adaptive_parameters = dict(self.model.named_parameters())
        for name, parameter in baseline.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.equal(parameter.grad, adaptive_parameters[name].grad),
                    msg=f"Gradient mismatch for {name}",
                )

    def test_global_mask_prunes_configured_bottom_percentage(self):
        self.model.pruning_percentage = 33.0
        gate_values = torch.tensor([0.8, 0.1, 0.7, 0.2, 0.9, 0.3])
        self.model.predict_band_gates = lambda patches: gate_values.to(
            patches.device
        ).expand(patches.shape[0], -1)
        loader = DataLoader(
            TensorDataset(
                torch.randn(4, 6, 5, 5),
                torch.zeros(4, dtype=torch.long),
            ),
            batch_size=2,
        )

        selected, mean_gates = self.model.estimate_global_band_mask(
            loader, torch.device("cpu")
        )

        self.assertEqual(selected.tolist(), [0, 2, 4, 5])
        self.assertTrue(torch.allclose(mean_gates, gate_values))

    def test_default_five_percent_prunes_five_of_103_bands(self):
        model = CNNTransformerCSABandPruningClassifier(
            input_channels=103,
            feature_channels=8,
            num_classes=3,
            patch_size=5,
            transformer_heads=2,
        )
        gate_values = torch.arange(103, dtype=torch.float32)
        model.predict_band_gates = lambda patches: gate_values.to(
            patches.device
        ).expand(patches.shape[0], -1)
        loader = DataLoader(
            TensorDataset(
                torch.randn(2, 103, 5, 5),
                torch.zeros(2, dtype=torch.long),
            ),
            batch_size=2,
        )

        selected, _ = model.estimate_global_band_mask(
            loader, torch.device("cpu")
        )

        self.assertEqual(len(selected), 98)
        self.assertEqual(selected.tolist(), list(range(5, 103)))


if __name__ == "__main__":
    unittest.main()
