import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from data import HyperspectralPatchDataset
from utils import PaviaClass, SplitName


@torch.no_grad()
def predict_classes(
    model,
    loader,
    device,
    *,
    progress_every=None,
    progress_label="Evaluation",
):
    """Predict a loader and optionally print periodic batch progress."""
    model.eval()
    all_targets = []
    all_predictions = []
    processed_samples = 0
    total_batches = len(loader)
    total_samples = len(loader.dataset)

    for batch_index, (patches, targets) in enumerate(loader, start=1):
        logits = model(patches.to(device))
        predictions = logits.argmax(dim=1).cpu()

        all_targets.append(targets.cpu())
        all_predictions.append(predictions)
        processed_samples += targets.size(0)

        should_report = (
            progress_every is not None
            and (
                batch_index == 1
                or batch_index % progress_every == 0
                or batch_index == total_batches
            )
        )
        if should_report:
            print(
                f"{progress_label}: batch {batch_index}/{total_batches} | "
                f"samples {processed_samples}/{total_samples}"
            )

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()
    return targets, predictions


def calculate_classification_metrics(targets, predictions, num_classes):
    """Calculate the metrics used by the paper."""
    class_indices = np.arange(num_classes)
    confusion = confusion_matrix(
        targets,
        predictions,
        labels=class_indices,
    )
    per_class_accuracy = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(num_classes, dtype=float),
        where=confusion.sum(axis=1) != 0,
    )

    return {
        "overall_accuracy": float(accuracy_score(targets, predictions)),
        "average_accuracy": float(per_class_accuracy.mean()),
        "kappa": float(cohen_kappa_score(targets, predictions)),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion,
    }


def aggregate_classification_metrics(run_metrics):
    """Aggregate classification metrics across independent training runs."""
    if not run_metrics:
        raise ValueError("At least one run is required for aggregation")

    scalar_names = ("overall_accuracy", "average_accuracy", "kappa")
    run_count = len(run_metrics)
    ddof = 1 if run_count > 1 else 0
    aggregate = {"num_runs": run_count}

    for metric_name in scalar_names:
        values = np.asarray(
            [metrics[metric_name] for metrics in run_metrics],
            dtype=float,
        )
        aggregate[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=ddof)),
        }

    per_class_values = np.stack(
        [metrics["per_class_accuracy"] for metrics in run_metrics]
    )
    aggregate["per_class_accuracy"] = {
        "mean": per_class_values.mean(axis=0),
        "std": per_class_values.std(axis=0, ddof=ddof),
    }
    aggregate["confusion_matrix_sum"] = np.stack(
        [metrics["confusion_matrix"] for metrics in run_metrics]
    ).sum(axis=0)
    return aggregate


def _json_compatible(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def save_evaluation_report(output_path, report):
    """Save configuration and metrics as a human-readable JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_compatible(report), indent=2),
        encoding="utf-8",
    )
    return output_path


def build_evaluation_report(
    configuration,
    checkpoint_path,
    evaluation_result,
    class_names,
):
    """Assemble a serializable report from one evaluation result."""
    checkpoint = evaluation_result["checkpoint"]
    return {
        "configuration": configuration,
        "checkpoint": {
            "path": checkpoint_path,
            "best_epoch": checkpoint["epoch"],
            "validation_loss": checkpoint["validation_loss"],
        },
        "test": {
            "samples": len(evaluation_result["targets"]),
            "class_names": class_names,
            "metrics": evaluation_result["metrics"],
        },
    }


def evaluate_checkpoint(
    model,
    loader,
    checkpoint_path,
    device,
    num_classes,
    *,
    progress_every=None,
    progress_label="Test evaluation",
):
    """Load a checkpoint, predict a dataset and calculate all metrics."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    targets, predictions = predict_classes(
        model,
        loader,
        device,
        progress_every=progress_every,
        progress_label=progress_label,
    )
    metrics = calculate_classification_metrics(
        targets, predictions, num_classes
    )
    return {
        "checkpoint": checkpoint,
        "targets": targets,
        "predictions": predictions,
        "metrics": metrics,
    }


def build_classification_map(
    model,
    device,
    ground_truth,
    padded_cube,
    split_tables,
    test_predictions,
    *,
    patch_size,
    num_classes,
    batch_size,
    num_workers,
):
    """Predict non-test labeled pixels and build the complete map."""
    non_test_table = np.vstack(
        [
            split_tables[SplitName.TRAIN.value],
            split_tables[SplitName.VALIDATION.value],
            split_tables[SplitName.EXCLUDED.value],
        ]
    )
    non_test_dataset = HyperspectralPatchDataset(
        padded_cube,
        non_test_table,
        patch_size,
        num_classes,
    )
    non_test_loader = DataLoader(
        non_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    _, non_test_predictions = predict_classes(
        model, non_test_loader, device
    )

    classification_map = np.zeros_like(
        ground_truth, dtype=np.uint8
    )
    test_table = split_tables[SplitName.TEST.value]
    test_rows = test_table[:, 0]
    test_columns = test_table[:, 1]
    classification_map[test_rows, test_columns] = test_predictions + 1

    non_test_rows = non_test_table[:, 0]
    non_test_columns = non_test_table[:, 1]
    classification_map[
        non_test_rows, non_test_columns
    ] = non_test_predictions + 1

    labeled_mask = ground_truth != PaviaClass.UNLABELED
    assert classification_map.shape == ground_truth.shape
    assert np.all(classification_map[~labeled_mask] == 0)
    assert np.all(classification_map[labeled_mask] != 0)
    return classification_map
