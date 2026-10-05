import json
import unittest

import torch
from torch.utils.data import TensorDataset

from data import (
    amplify_training_dataset,
    create_run_datasets,
    rotate_patch_nearest,
)
from utils import SampleAmplificationMode, SplitName


class SampleAmplificationTests(unittest.TestCase):
    def setUp(self):
        patches = torch.arange(
            6 * 2 * 5 * 5, dtype=torch.float32
        ).reshape(6, 2, 5, 5) / 100.0
        targets = torch.tensor([0, 0, 0, 1, 1, 1])
        self.dataset = TensorDataset(patches, targets)

    def test_disabled_mode_returns_original_dataset(self):
        result = amplify_training_dataset(
            self.dataset,
            mode=SampleAmplificationMode.DISABLED,
            seed=7,
        )
        self.assertIs(result, self.dataset)

    def test_cta_net_mode_builds_four_parallel_pools(self):
        result = amplify_training_dataset(
            self.dataset,
            mode=SampleAmplificationMode.CTA_NET,
            seed=7,
        )

        self.assertEqual(len(result), 22)
        self.assertEqual(
            result.amplification_metadata["linear_samples"], 4
        )
        json.dumps(result.amplification_metadata)
        expected_targets = [0, 0, 0, 1, 1, 1]
        self.assertEqual(result.targets[:6].tolist(), expected_targets)
        self.assertEqual(result.targets[6:12].tolist(), expected_targets)
        self.assertEqual(result.targets[12:18].tolist(), expected_targets)
        self.assertEqual(result.targets[18:].tolist(), [0, 0, 1, 1])

        originals = torch.stack([self.dataset[index][0] for index in range(6)])
        self.assertTrue(torch.equal(result.patches[:6], originals))
        self.assertTrue(
            torch.equal(
                result.patches[6:12, :, 1:4, 1:4],
                originals[:, :, 1:4, 1:4],
            )
        )
        self.assertFalse(torch.equal(result.patches[6:12], originals))

        anchors = result.amplification_metadata[
            "anchor_indices_by_zero_based_class"
        ]
        linear_offset = 18
        for class_id in (0, 1):
            class_indices = torch.nonzero(
                torch.tensor(expected_targets) == class_id,
                as_tuple=True,
            )[0].tolist()
            anchor_index = anchors[class_id]
            for sample_index in class_indices:
                if sample_index == anchor_index:
                    continue
                self.assertTrue(
                    torch.equal(
                        result.patches[linear_offset],
                        originals[sample_index] + originals[anchor_index],
                    )
                )
                linear_offset += 1

    def test_same_seed_reproduces_all_augmented_samples(self):
        first = amplify_training_dataset(
            self.dataset,
            mode=SampleAmplificationMode.CTA_NET,
            seed=91,
        )
        second = amplify_training_dataset(
            self.dataset,
            mode=SampleAmplificationMode.CTA_NET,
            seed=91,
        )
        different = amplify_training_dataset(
            self.dataset,
            mode=SampleAmplificationMode.CTA_NET,
            seed=92,
        )

        self.assertTrue(torch.equal(first.patches, second.patches))
        self.assertFalse(torch.equal(first.patches, different.patches))

    def test_rotation_uses_the_same_spatial_grid_for_every_band(self):
        first_band = torch.arange(1, 26, dtype=torch.float32).reshape(5, 5)
        patch = torch.stack((first_band, first_band * 10))

        rotated = rotate_patch_nearest(patch, 37.0)

        self.assertTrue(torch.equal(rotated[1], rotated[0] * 10))

    def test_only_training_dataset_is_replaced(self):
        datasets = {
            SplitName.TRAIN.value: self.dataset,
            SplitName.VALIDATION.value: object(),
            SplitName.TEST.value: object(),
        }
        result = create_run_datasets(
            datasets,
            sample_amplification_mode=SampleAmplificationMode.CTA_NET,
            training_seed=7,
        )

        self.assertIs(
            result[SplitName.VALIDATION.value],
            datasets[SplitName.VALIDATION.value],
        )
        self.assertIs(
            result[SplitName.TEST.value], datasets[SplitName.TEST.value]
        )
        self.assertIsNot(result[SplitName.TRAIN.value], self.dataset)


if __name__ == "__main__":
    unittest.main()
