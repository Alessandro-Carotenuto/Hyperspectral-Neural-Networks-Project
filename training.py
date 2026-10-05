import torch
import torch.nn as nn

from utils import LRSchedulingType, ModelArchitecture


def build_lr_scheduler(optimizer, scheduling_type, total_epochs):
    """Create the configured scheduler, or ``None`` for a fixed LR."""
    if scheduling_type == LRSchedulingType.FIXED:
        return None
    if scheduling_type == LRSchedulingType.COSINEANNEALING:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs
        )
    if scheduling_type == LRSchedulingType.REDUCELRONPLATEAU:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2
        )
    raise ValueError(f"Unsupported LR scheduling: {scheduling_type}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return sample-weighted loss and accuracy."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for patches, targets in loader:
        patches = patches.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(patches)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples
    return epoch_loss, epoch_accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate sample-weighted loss and accuracy."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for patches, targets in loader:
        patches = patches.to(device)
        targets = targets.to(device)

        logits = model(patches)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy


def configure_reproducibility(seed):
    """Configure deterministic PyTorch behavior and return the device."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_checkpoint_path(
    checkpoint_directory,
    *,
    architecture,
    transformer_heads,
    transformer_expansion_factor,
    transformer_dropout,
    transformer_ffn_residual_scale=0.5,
    model_variant,
    run_name,
    patch_size,
    epochs,
    feature_channels,
    scheduling_type,
    split_seed,
    split_type,
    training_seed,
):
    """Build a checkpoint path containing the experiment identity."""
    architecture_name = architecture.value
    if model_variant:
        architecture_name = f"{architecture_name}_{model_variant}"
    if architecture.uses_transformer:
        dropout_name = str(transformer_dropout).replace(".", "p")
        architecture_name = (
            f"{architecture_name}_h{transformer_heads}_"
            f"x{transformer_expansion_factor}_d{dropout_name}"
        )
        if transformer_ffn_residual_scale != 0.5:
            residual_scale_name = str(
                transformer_ffn_residual_scale
            ).replace(".", "p")
            architecture_name = (
                f"{architecture_name}_s{residual_scale_name}"
            )
    filename = (
        f"{architecture_name}_{run_name}_p{int(patch_size)}_e{epochs}_"
        f"f{feature_channels}_lr{scheduling_type.value}_"
        f"split{split_seed}_{split_type.value}_"
        f"train{training_seed}.pt"
    )
    return checkpoint_directory / filename


def run_small_batch_overfit_check(
    model,
    patches,
    targets,
    *,
    learning_rate,
    min_steps,
    max_steps,
    target_loss,
):
    """Verify that a model can memorize one deliberately tiny batch."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )
    initial_loss = None

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, targets)

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()
        if (
            step >= min_steps
            and accuracy == 1.0
            and loss.item() < target_loss
        ):
            break

    model.eval()
    with torch.no_grad():
        final_logits = model(patches)
        final_loss = criterion(final_logits, targets).item()
        final_predictions = final_logits.argmax(dim=1)
        final_accuracy = (
            (final_predictions == targets).float().mean().item()
        )

    assert final_loss < initial_loss
    assert final_loss < target_loss
    assert final_accuracy == 1.0
    return {
        "steps": step,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
    }


def fit_model(
    model,
    train_loader,
    validation_loader,
    device,
    *,
    epochs,
    learning_rate,
    scheduling_type,
    checkpoint_path,
    checkpoint_metadata,
    log_every=10,
):
    """Train a model, save the best checkpoint and return its history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )
    scheduler = build_lr_scheduler(
        optimizer, scheduling_type, epochs
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }
    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        current_learning_rate = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["learning_rate"].append(current_learning_rate)

        if scheduler is not None:
            if scheduling_type == LRSchedulingType.REDUCELRONPLATEAU:
                scheduler.step(validation_loss)
            else:
                scheduler.step()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict()
                        if scheduler is not None
                        else None
                    ),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "history": history,
                    **checkpoint_metadata,
                },
                checkpoint_path,
            )

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train loss {train_loss:.4f}, "
                f"acc {train_accuracy:.4f} | "
                f"val loss {validation_loss:.4f}, "
                f"acc {validation_accuracy:.4f} | "
                f"lr {current_learning_rate:.2e}"
            )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "checkpoint_path": checkpoint_path,
    }
