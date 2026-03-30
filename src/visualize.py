import matplotlib.pyplot as plt
from pathlib import Path


def visualize_training(history: dict, model_name: str, save_dir: str = "plots") -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(model_name, fontsize=14, fontweight="bold")

    ax_loss.plot(epochs, history["train_loss"], label="Train loss")
    ax_loss.plot(epochs, history["val_loss"],   label="Val loss", linestyle="--")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(epochs, history["train_acc"], label="Train acc")
    ax_acc.plot(epochs, history["val_acc"],   label="Val acc", linestyle="--")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)

    plt.tight_layout()

    save_path = Path(save_dir) / f"{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wykres zapisany: {save_path}")


