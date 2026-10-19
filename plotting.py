import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from utils import SplitName


def plot_spectral_bands_and_ground_truth(
    cube,
    ground_truth,
    bands_to_show,
    num_classes,
):
    """Plot selected HSI bands beside the ground truth."""
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    for axis, band_index in zip(axes[:3], bands_to_show):
        axis.imshow(cube[:, :, band_index], cmap="gray")
        axis.set_title(f"Band {band_index}")
        axis.axis("off")

    image = axes[3].imshow(
        ground_truth,
        cmap="tab10",
        vmin=0,
        vmax=num_classes,
    )
    axes[3].set_title("Ground Truth")
    axes[3].axis("off")
    plt.colorbar(image, ax=axes[3], fraction=0.046)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels, counts, class_names):
    """Plot the number of labeled pixels per class."""
    labeled_mask = labels != 0
    labeled_labels = labels[labeled_mask]
    labeled_counts = counts[labeled_mask]
    labeled_names = [class_names[label] for label in labeled_labels]

    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(labeled_names, labeled_counts, color="steelblue")
    axis.set_title("Pavia University Labeled Class Distribution")
    axis.set_xlabel("Number of labeled pixels")
    axis.set_ylabel("Class")
    axis.invert_yaxis()
    axis.bar_label(bars, padding=4)
    plt.tight_layout()
    plt.show()


def plot_mean_spectral_signatures(
    cube,
    ground_truth,
    labeled_class_ids,
    class_names,
):
    """Plot the mean spectral signature of every labeled class."""
    band_indices = np.arange(1, cube.shape[2] + 1)
    figure, axis = plt.subplots(figsize=(12, 7))

    for label in labeled_class_ids:
        class_pixels = cube[ground_truth == label]
        axis.plot(
            band_indices,
            class_pixels.mean(axis=0),
            label=class_names[label],
        )

    axis.set_title("Mean Spectral Signature per Class")
    axis.set_xlabel("Spectral band")
    axis.set_ylabel("Mean pixel value")
    axis.grid(alpha=0.25)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_training_curves(history, run_name, model_name):
    """Plot loss and accuracy histories."""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(
        epochs, history["validation_loss"], label="Validation"
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(
        epochs,
        history["validation_accuracy"],
        label="Validation",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle(f"{model_name} training curves ({run_name} run)")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    confusion,
    display_names,
    model_name,
    *,
    normalized=False,
):
    """Plot a raw or row-normalized confusion matrix."""
    matrix = confusion
    values_format = None
    title = f"{model_name} Test Confusion Matrix"
    if normalized:
        row_totals = confusion.sum(axis=1, keepdims=True)
        matrix = np.divide(
            confusion,
            row_totals,
            out=np.zeros_like(confusion, dtype=float),
            where=row_totals != 0,
        )
        values_format = ".2f"
        title = f"{model_name} Normalized Test Confusion Matrix"

    figure, axis = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=display_names,
    ).plot(
        ax=axis,
        cmap="Blues",
        values_format=values_format,
        xticks_rotation=45,
        colorbar=False,
    )
    axis.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_pruning_mask(mean_band_gates, selected_band_indices, run_label):
    """Show the global binary mask and mean learned gate for every band."""
    mean_band_gates = np.asarray(mean_band_gates, dtype=float).reshape(-1)
    selected_band_indices = np.asarray(
        selected_band_indices, dtype=int
    ).reshape(-1)
    selected_mask = np.zeros(mean_band_gates.size, dtype=bool)
    selected_mask[selected_band_indices] = True
    band_numbers = np.arange(1, mean_band_gates.size + 1)
    colors = np.where(selected_mask, "seagreen", "lightgray")
    retained_count = int(selected_mask.sum())

    figure, axes = plt.subplots(
        2, 1, figsize=(14, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    axes[0].bar(band_numbers, selected_mask.astype(int), color=colors)
    axes[0].set_yticks([0, 1], labels=["Dropped", "Kept"])
    axes[0].set_ylim(-0.1, 1.2)
    axes[0].set_title(
        f"Adaptive spectral pruning mask — {run_label} "
        f"({retained_count}/{mean_band_gates.size} bands kept)"
    )
    axes[0].grid(axis="x", alpha=0.15)

    axes[1].plot(
        band_numbers, mean_band_gates, color="steelblue", linewidth=1.5
    )
    axes[1].scatter(
        band_numbers[selected_mask], mean_band_gates[selected_mask],
        color="seagreen", s=18, label="Kept",
    )
    axes[1].scatter(
        band_numbers[~selected_mask], mean_band_gates[~selected_mask],
        color="gray", s=18, label="Dropped",
    )
    axes[1].axhline(0.5, color="darkorange", linestyle="--", label="0.5 threshold")
    axes[1].set_xlabel("Original spectral band")
    axes[1].set_ylabel("Mean gate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlim(0.5, mean_band_gates.size + 0.5)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_classification_map(
    ground_truth,
    classification_map,
    class_names,
    num_classes,
    model_name,
    split_tables,
):
    """Plot ground truth, predictions and errors outside validation/test."""
    colors = ["black"] + list(
        plt.get_cmap("tab10").colors[:num_classes]
    )
    color_map = ListedColormap(colors)
    error_map = np.zeros_like(ground_truth, dtype=np.uint8)
    ignored_table = np.vstack(
        [
            split_tables[SplitName.TEST.value],
            split_tables[SplitName.VALIDATION.value],
        ]
    )
    labeled_mask = ground_truth != 0
    error_map[labeled_mask] = (
        classification_map[labeled_mask] == ground_truth[labeled_mask]
    ).astype(np.uint8) + 1
    error_map[ignored_table[:, 0], ignored_table[:, 1]] = 0
    error_color_map = ListedColormap(["black", "green", "red"])

    figure, axes = plt.subplots(1, 3, figsize=(20, 9))

    for axis, image, title in zip(
        axes[:2],
        [ground_truth, classification_map],
        ["Ground Truth", f"{model_name} Classification Map"],
    ):
        displayed_image = axis.imshow(
            image,
            cmap=color_map,
            vmin=-0.5,
            vmax=num_classes + 0.5,
        )
        axis.set_title(title)
        axis.axis("off")

    colorbar = figure.colorbar(
        displayed_image,
        ax=axes[:2],
        ticks=np.arange(num_classes + 1),
        fraction=0.035,
        pad=0.02,
    )
    colorbar.ax.set_yticklabels(
        [class_names[label] for label in range(num_classes + 1)]
    )
    axes[2].imshow(
        error_map,
        cmap=error_color_map,
        vmin=0,
        vmax=2,
    )
    axes[2].set_title("Train/excluded errors")
    axes[2].axis("off")
    figure.legend(
        handles=[
            Patch(color="green", label="Correct"),
            Patch(color="red", label="Incorrect"),
            Patch(color="black", label="Validation/test/unlabeled"),
        ],
        loc="lower center",
        ncol=3,
    )
    figure.suptitle(
        f"Pavia University: Ground Truth, {model_name} Prediction and Errors"
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    plt.show()
