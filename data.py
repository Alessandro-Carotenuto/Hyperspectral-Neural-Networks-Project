import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from utils import (
    DataSplitType,
    InputNormalization,
    SampleAmplificationMode,
    SplitName,
)


def build_forbidden_test_center_mask(image_shape, train_coordinates, radius):
    """Mark test centers whose patches would violate the split policy."""
    forbidden_mask = np.zeros(image_shape, dtype=bool)
    image_height, image_width = image_shape

    for row, column in train_coordinates:
        row_start = max(0, int(row) - radius)
        row_end = min(image_height, int(row) + radius + 1)
        column_start = max(0, int(column) - radius)
        column_end = min(image_width, int(column) + radius + 1)
        forbidden_mask[
            row_start:row_end, column_start:column_end
        ] = True

    return forbidden_mask


def coordinate_set(coordinates):
    """Convert an ``(N, 2)`` coordinate array to hashable pairs."""
    return {tuple(coordinate) for coordinate in coordinates.tolist()}


def build_split_table(coordinates_by_label, labeled_class_ids):
    """Build a ``row, column, label`` table ordered by class."""
    rows = []

    for label in labeled_class_ids:
        coordinates = coordinates_by_label[label]
        label_column = np.full((len(coordinates), 1), label)
        rows.append(np.column_stack((coordinates, label_column)))

    return np.vstack(rows)


def normalize_per_band_min_max(cube):
    """Normalize every spectral band independently to the [0, 1] range."""
    cube_float = cube.astype(np.float32)
    band_min = cube_float.min(axis=(0, 1), keepdims=True)
    band_max = cube_float.max(axis=(0, 1), keepdims=True)
    band_range = band_max - band_min

    constant_band_indices = np.flatnonzero(band_range.reshape(-1) == 0)
    if len(constant_band_indices) > 0:
        raise ValueError(
            "Constant spectral bands found: "
            f"{constant_band_indices.tolist()}"
        )

    normalized_cube = (cube_float - band_min) / band_range
    return normalized_cube, band_min, band_max


def normalize_per_band_z_score(cube):
    """Standardize every spectral band to zero mean and unit variance."""
    cube_float = cube.astype(np.float32)
    band_mean = cube_float.mean(axis=(0, 1), keepdims=True)
    band_standard_deviation = cube_float.std(
        axis=(0, 1), keepdims=True
    )

    constant_band_indices = np.flatnonzero(
        band_standard_deviation.reshape(-1) == 0
    )
    if len(constant_band_indices) > 0:
        raise ValueError(
            "Constant spectral bands found: "
            f"{constant_band_indices.tolist()}"
        )

    normalized_cube = (
        cube_float - band_mean
    ) / band_standard_deviation
    return normalized_cube, band_mean, band_standard_deviation


def normalize_per_band(cube, normalization):
    """Apply the selected per-band normalization without changing raw data."""
    normalization = InputNormalization(normalization)
    if normalization == InputNormalization.MIN_MAX:
        return normalize_per_band_min_max(cube)[0]
    return normalize_per_band_z_score(cube)[0]


def reflect_pad_spatially(cube, patch_radius):
    """Reflect-pad only the two spatial axes of an HSI cube."""
    return np.pad(
        cube,
        (
            (patch_radius, patch_radius),
            (patch_radius, patch_radius),
            (0, 0),
        ),
        mode="reflect",
    )


def extract_patch(padded_cube, row, column, patch_size):
    """Extract one HWC patch using coordinates from the unpadded image."""
    patch_size = int(patch_size)
    patch = padded_cube[
        row:row + patch_size,
        column:column + patch_size,
        :,
    ]
    expected_shape = (patch_size, patch_size, padded_cube.shape[2])
    if patch.shape != expected_shape:
        raise ValueError(
            f"Invalid patch shape {patch.shape} at {(row, column)}"
        )

    return patch


class HyperspectralPatchDataset(Dataset):
    """Extract HSI patches on demand from a pre-padded cube."""

    def __init__(self, padded_hsi_cube, split_table, patch_size, num_classes):
        self.padded_hsi_cube = padded_hsi_cube
        self.coordinates = split_table[:, :2].astype(np.int64)
        self.targets = split_table[:, 2].astype(np.int64) - 1
        self.patch_size = int(patch_size)

        assert self.coordinates.shape == (len(split_table), 2)
        assert self.targets.min() >= 0
        assert self.targets.max() < num_classes

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        row, column = self.coordinates[index]
        patch = extract_patch(
            self.padded_hsi_cube,
            row,
            column,
            self.patch_size,
        )
        patch = np.ascontiguousarray(
            patch.transpose(2, 0, 1),
            dtype=np.float32,
        )

        patch_tensor = torch.from_numpy(patch)
        target_tensor = torch.tensor(
            self.targets[index],
            dtype=torch.long,
        )
        return patch_tensor, target_tensor


class MaterializedPatchDataset(Dataset):
    """Store a finite patch set produced for one training run."""

    def __init__(self, patches, targets, amplification_metadata):
        self.patches = patches
        self.targets = targets.numpy()
        self.amplification_metadata = amplification_metadata

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.patches[index], torch.tensor(
            self.targets[index], dtype=torch.long
        )


def rotate_patch_nearest(patch, angle_degrees):
    """Rotate all bands around the patch center with zero fill."""
    angle_radians = torch.deg2rad(
        torch.tensor(angle_degrees, dtype=patch.dtype)
    )
    cosine = torch.cos(angle_radians)
    sine = torch.sin(angle_radians)
    affine = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0]],
        dtype=patch.dtype,
    ).unsqueeze(0)
    batched_patch = patch.unsqueeze(0)
    grid = functional.affine_grid(
        affine, batched_patch.shape, align_corners=False
    )
    return functional.grid_sample(
        batched_patch,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0)


def amplify_training_dataset(
    dataset,
    *,
    mode,
    seed,
    central_region_size=3,
    gaussian_noise_standard_deviation=0.04,
):
    """Materialize the paper-faithful CTA-Net training augmentation."""
    mode = SampleAmplificationMode(mode)
    if mode == SampleAmplificationMode.DISABLED:
        return dataset

    central_region_size = int(central_region_size)
    noise_standard_deviation = float(
        gaussian_noise_standard_deviation
    )
    if central_region_size < 1 or central_region_size % 2 == 0:
        raise ValueError("SA central region size must be a positive odd number")
    if noise_standard_deviation < 0:
        raise ValueError("SA Gaussian noise standard deviation cannot be negative")

    original_patches = []
    original_targets = []
    for patch, target in dataset:
        original_patches.append(patch.detach().clone())
        original_targets.append(int(target))
    if not original_patches:
        raise ValueError("Cannot amplify an empty training dataset")

    originals = torch.stack(original_patches)
    targets = torch.tensor(original_targets, dtype=torch.long)
    patch_height, patch_width = originals.shape[-2:]
    if patch_height != patch_width:
        raise ValueError("SA requires square patches")
    if central_region_size > patch_height:
        raise ValueError("SA central region cannot exceed the patch size")

    generator = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(
        originals.shape,
        generator=generator,
        dtype=originals.dtype,
    ) * noise_standard_deviation
    center_start = (patch_height - central_region_size) // 2
    center_end = center_start + central_region_size
    noise[:, :, center_start:center_end, center_start:center_end] = 0
    noisy_patches = originals + noise

    angles = (
        torch.rand(len(originals), generator=generator) * 360.0 - 180.0
    )
    rotated_patches = torch.stack([
        rotate_patch_nearest(patch, float(angle))
        for patch, angle in zip(originals, angles)
    ])

    linear_patches = []
    linear_targets = []
    anchor_indices = {}
    for class_id in torch.unique(targets, sorted=True).tolist():
        class_indices = torch.nonzero(
            targets == class_id, as_tuple=True
        )[0]
        if len(class_indices) < 2:
            raise ValueError(
                "CTA-Net linear SA requires at least two samples per class"
            )
        anchor_offset = int(
            torch.randint(
                len(class_indices), (1,), generator=generator
            ).item()
        )
        anchor_index = int(class_indices[anchor_offset])
        anchor_indices[int(class_id)] = anchor_index
        for sample_index in class_indices.tolist():
            if sample_index == anchor_index:
                continue
            linear_patches.append(
                originals[sample_index] + originals[anchor_index]
            )
            linear_targets.append(class_id)

    amplified_patches = torch.cat(
        (
            originals,
            noisy_patches,
            rotated_patches,
            torch.stack(linear_patches),
        )
    )
    amplified_targets = torch.cat(
        (
            targets,
            targets,
            targets,
            torch.tensor(linear_targets, dtype=torch.long),
        )
    )
    metadata = {
        "mode": mode.value,
        "seed": int(seed),
        "central_region_size": central_region_size,
        "gaussian_noise_standard_deviation": noise_standard_deviation,
        "rotation_degrees": [-180.0, 180.0],
        "rotation_interpolation": "nearest",
        "rotation_fill": 0.0,
        "linear_mode": "direct_sum",
        "clip_after_transform": False,
        "original_samples": len(originals),
        "noise_samples": len(noisy_patches),
        "rotation_samples": len(rotated_patches),
        "linear_samples": len(linear_patches),
        "total_samples": len(amplified_targets),
        "anchor_indices_by_zero_based_class": anchor_indices,
        "rotation_angles_degrees": angles.tolist(),
    }
    return MaterializedPatchDataset(
        amplified_patches, amplified_targets, metadata
    )


def create_run_datasets(
    datasets,
    *,
    sample_amplification_mode,
    training_seed,
    central_region_size=3,
    gaussian_noise_standard_deviation=0.04,
):
    """Create per-run datasets while preserving validation and test."""
    run_datasets = dict(datasets)
    run_datasets[SplitName.TRAIN.value] = amplify_training_dataset(
        datasets[SplitName.TRAIN.value],
        mode=sample_amplification_mode,
        seed=training_seed,
        central_region_size=central_region_size,
        gaussian_noise_standard_deviation=(
            gaussian_noise_standard_deviation
        ),
    )
    return run_datasets


def collect_coordinates_by_class(ground_truth, labeled_class_ids):
    """Collect the center coordinates for every labeled class."""
    return {
        label: np.argwhere(ground_truth == label)
        for label in labeled_class_ids
    }


def create_few_shot_split(
    ground_truth,
    labeled_class_ids,
    *,
    seed,
    train_samples_per_class,
    validation_samples_per_class,
    split_type,
    patch_radius,
):
    """Create train, validation, test and excluded coordinate tables."""
    coordinates_by_class = collect_coordinates_by_class(
        ground_truth, labeled_class_ids
    )
    rng = np.random.default_rng(seed)
    train_coordinates_by_class = {}
    validation_coordinates_by_class = {}
    test_coordinates_by_class = {}
    excluded_coordinates_by_class = {}
    validation_end = (
        train_samples_per_class + validation_samples_per_class
    )

    for label, coordinates in coordinates_by_class.items():
        shuffled_coordinates = rng.permutation(coordinates)
        train_coordinates_by_class[label] = shuffled_coordinates[
            :train_samples_per_class
        ]
        validation_coordinates_by_class[label] = shuffled_coordinates[
            train_samples_per_class:validation_end
        ]
        test_coordinates_by_class[label] = shuffled_coordinates[
            validation_end:
        ]

    all_train_coordinates = np.vstack(
        list(train_coordinates_by_class.values())
    )
    if split_type == DataSplitType.RANDOM:
        forbidden_test_center_mask = np.zeros(
            ground_truth.shape, dtype=bool
        )
    elif split_type == DataSplitType.RANDOM_NO_CENTER_OVERLAP:
        forbidden_test_center_mask = build_forbidden_test_center_mask(
            ground_truth.shape,
            all_train_coordinates,
            patch_radius,
        )
    elif split_type == DataSplitType.RANDOM_NO_OVERLAP:
        forbidden_test_center_mask = build_forbidden_test_center_mask(
            ground_truth.shape,
            all_train_coordinates,
            2 * patch_radius,
        )
    else:
        raise ValueError(f"Unsupported data split: {split_type}")

    for label in labeled_class_ids:
        candidate_coordinates = test_coordinates_by_class[label]
        is_excluded = forbidden_test_center_mask[
            candidate_coordinates[:, 0], candidate_coordinates[:, 1]
        ]
        test_coordinates_by_class[label] = candidate_coordinates[
            ~is_excluded
        ]
        excluded_coordinates_by_class[label] = candidate_coordinates[
            is_excluded
        ]

    return {
        SplitName.TRAIN.value: build_split_table(
            train_coordinates_by_class, labeled_class_ids
        ),
        SplitName.VALIDATION.value: build_split_table(
            validation_coordinates_by_class, labeled_class_ids
        ),
        SplitName.TEST.value: build_split_table(
            test_coordinates_by_class, labeled_class_ids
        ),
        SplitName.EXCLUDED.value: build_split_table(
            excluded_coordinates_by_class, labeled_class_ids
        ),
    }


def split_counts_by_class(split_tables, labeled_class_ids):
    """Return sample counts for every split and original class label."""
    counts = {}
    for label in labeled_class_ids:
        counts[label] = {
            split_name: int(np.count_nonzero(table[:, 2] == label))
            for split_name, table in split_tables.items()
        }
    return counts


def verify_few_shot_split(
    split_tables,
    ground_truth,
    labeled_class_ids,
    *,
    train_samples_per_class,
    validation_samples_per_class,
    split_type,
    patch_radius,
    class_names=None,
):
    """Verify coverage, labels, counts and the selected overlap policy."""
    coordinates_by_class = collect_coordinates_by_class(
        ground_truth, labeled_class_ids
    )
    all_split_sets = {
        split_name: coordinate_set(table[:, :2])
        for split_name, table in split_tables.items()
    }

    split_names = tuple(split_tables)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1:]:
            assert all_split_sets[left_name].isdisjoint(
                all_split_sets[right_name]
            )

    total_labeled_coordinates = sum(
        len(coordinates) for coordinates in coordinates_by_class.values()
    )
    assert sum(len(table) for table in split_tables.values()) == (
        total_labeled_coordinates
    )
    combined_coordinates = set().union(*all_split_sets.values())
    assert len(combined_coordinates) == total_labeled_coordinates
    assert len(split_tables[SplitName.TRAIN.value]) == (
        len(labeled_class_ids) * train_samples_per_class
    )
    assert len(split_tables[SplitName.VALIDATION.value]) == (
        len(labeled_class_ids) * validation_samples_per_class
    )

    counts = split_counts_by_class(split_tables, labeled_class_ids)
    for label in labeled_class_ids:
        class_name = (
            class_names[label] if class_names is not None else str(label)
        )
        if counts[label][SplitName.TEST.value] == 0:
            raise ValueError(
                f"{split_type.value} leaves no test samples for "
                f"class {label} ({class_name}). Choose another seed "
                "or a less restrictive split."
            )
        assert (
            counts[label][SplitName.TRAIN.value]
            == train_samples_per_class
        )
        assert (
            counts[label][SplitName.VALIDATION.value]
            == validation_samples_per_class
        )

        original_coordinates = coordinate_set(
            coordinates_by_class[label]
        )
        class_split_coordinates = set()
        for table in split_tables.values():
            class_rows = table[table[:, 2] == label, :2]
            class_split_coordinates.update(coordinate_set(class_rows))
        assert class_split_coordinates == original_coordinates

    train_coordinates = split_tables[SplitName.TRAIN.value][:, :2]
    if split_type == DataSplitType.RANDOM:
        forbidden_radius = None
    elif split_type == DataSplitType.RANDOM_NO_CENTER_OVERLAP:
        forbidden_radius = patch_radius
    elif split_type == DataSplitType.RANDOM_NO_OVERLAP:
        forbidden_radius = 2 * patch_radius
    else:
        raise ValueError(f"Unsupported data split: {split_type}")

    if forbidden_radius is not None:
        forbidden_mask = build_forbidden_test_center_mask(
            ground_truth.shape,
            train_coordinates,
            forbidden_radius,
        )
        test_coordinates = split_tables[SplitName.TEST.value][:, :2]
        excluded_coordinates = split_tables[
            SplitName.EXCLUDED.value
        ][:, :2]
        assert not forbidden_mask[
            test_coordinates[:, 0], test_coordinates[:, 1]
        ].any()
        assert forbidden_mask[
            excluded_coordinates[:, 0], excluded_coordinates[:, 1]
        ].all()
    else:
        assert len(split_tables[SplitName.EXCLUDED.value]) == 0

    for table in split_tables.values():
        rows = table[:, 0]
        columns = table[:, 1]
        labels = table[:, 2]
        assert np.array_equal(ground_truth[rows, columns], labels)

    return counts


def split_file_suffix(split_type):
    """Return the filename suffix used by one split policy."""
    return "" if split_type == DataSplitType.RANDOM else f"_{split_type.value}"


def save_and_verify_split(
    split_tables,
    ground_truth,
    split_directory,
    split_type,
):
    """Save split CSV files, reload them and verify exact equality."""
    split_directory.mkdir(parents=True, exist_ok=True)
    suffix = split_file_suffix(split_type)
    loaded_split_tables = {}

    for split_name, expected_table in split_tables.items():
        path = split_directory / f"{split_name}{suffix}.csv"
        np.savetxt(
            path,
            expected_table,
            fmt="%d",
            delimiter=",",
            header="row,column,label",
            comments="",
        )
        print(f"Saved {len(expected_table)} samples to {path}")

        if len(expected_table) == 0:
            loaded_table = np.empty((0, 3), dtype=np.int64)
        else:
            loaded_table = np.atleast_2d(
                np.loadtxt(
                    path,
                    delimiter=",",
                    skiprows=1,
                    dtype=np.int64,
                )
            )
        assert loaded_table.shape == expected_table.shape
        assert np.array_equal(loaded_table, expected_table)

        rows = loaded_table[:, 0]
        columns = loaded_table[:, 1]
        labels = loaded_table[:, 2]
        assert np.array_equal(ground_truth[rows, columns], labels)
        loaded_split_tables[split_name] = loaded_table
        print(f"Verified {path.name}: {len(loaded_table)} samples")

    return loaded_split_tables


def count_padding_requirements(split_tables, image_shape, patch_radius):
    """Count samples whose patches touch an image border."""
    image_height, image_width = image_shape
    counts = {}
    for split_name, split_table in split_tables.items():
        rows = split_table[:, 0]
        columns = split_table[:, 1]
        requires_padding = (
            (rows < patch_radius)
            | (rows >= image_height - patch_radius)
            | (columns < patch_radius)
            | (columns >= image_width - patch_radius)
        )
        counts[split_name] = int(np.count_nonzero(requires_padding))
    return counts


def create_patch_datasets(
    padded_cube,
    split_tables,
    patch_size,
    num_classes,
):
    """Create the train, validation and test patch datasets."""
    return {
        split_name: HyperspectralPatchDataset(
            padded_cube,
            split_tables[split_name],
            patch_size,
            num_classes,
        )
        for split_name in (
            SplitName.TRAIN.value,
            SplitName.VALIDATION.value,
            SplitName.TEST.value,
        )
    }


def create_dataloaders(
    datasets,
    *,
    batch_size,
    num_workers,
    training_seed,
):
    """Create deterministic train, validation and test DataLoaders."""
    train_generator = torch.Generator().manual_seed(training_seed)
    return {
        SplitName.TRAIN.value: DataLoader(
            datasets[SplitName.TRAIN.value],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=train_generator,
        ),
        SplitName.VALIDATION.value: DataLoader(
            datasets[SplitName.VALIDATION.value],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        SplitName.TEST.value: DataLoader(
            datasets[SplitName.TEST.value],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
