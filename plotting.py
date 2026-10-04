import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


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


def plot_training_curves(history, run_name):
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

    figure.suptitle(f"CNN-only training curves ({run_name} run)")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    confusion,
    display_names,
    *,
    normalized=False,
):
    """Plot a raw or row-normalized confusion matrix."""
    matrix = confusion
    values_format = None
    title = "CNN-Only Test Confusion Matrix"
    if normalized:
        row_totals = confusion.sum(axis=1, keepdims=True)
        matrix = np.divide(
            confusion,
            row_totals,
            out=np.zeros_like(confusion, dtype=float),
            where=row_totals != 0,
        )
        values_format = ".2f"
        title = "CNN-Only Normalized Test Confusion Matrix"

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


def plot_classification_map(
    ground_truth,
    classification_map,
    class_names,
    num_classes,
):
    """Plot ground truth and the predicted classification map."""
    colors = ["black"] + list(
        plt.get_cmap("tab10").colors[:num_classes]
    )
    color_map = ListedColormap(colors)
    figure, axes = plt.subplots(1, 2, figsize=(14, 9))

    for axis, image, title in zip(
        axes,
        [ground_truth, classification_map],
        ["Ground Truth", "CNN-Only Classification Map"],
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
        ax=axes,
        ticks=np.arange(num_classes + 1),
        fraction=0.035,
        pad=0.02,
    )
    colorbar.ax.set_yticklabels(
        [class_names[label] for label in range(num_classes + 1)]
    )
    figure.suptitle("Pavia University: Ground Truth vs Prediction")
    plt.show()
