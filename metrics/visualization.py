import os
import matplotlib.pyplot as plt
import numpy as np


def save_plots(metrics, title, epoch, folder="results"):
    os.makedirs(folder, exist_ok=True)
    plt.figure()

    # Convert a list of tensors to a NumPy array
    metrics_np = np.array(
        [
            metric.detach().numpy() if metric.requires_grad else metric.numpy()
            for metric in metrics
        ]
    )

    plt.plot(metrics_np)
    plt.title(f"{title} Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.savefig(os.path.join(folder, f"{title.lower()}_epoch_{epoch}.png"))
    plt.close()


import matplotlib.pyplot as plt


def save_metrics_plot(
    train_metrics, val_metrics, title, xlabel, ylabel, folder="metrics_plots"
):
    """
    Saves a plot comparing training and validation metrics over epochs.
    Handles a single epoch case by using scatter plot if not enough data for line plot.
    """
    os.makedirs(folder, exist_ok=True)
    plt.figure()
    epochs = range(len(train_metrics))

    if len(train_metrics) > 1:
        plt.plot(epochs, train_metrics, label="Train")
        plt.plot(epochs, val_metrics, label="Validation")
    else:
        plt.scatter(epochs, train_metrics, label="Train")
        plt.scatter(epochs, val_metrics, label="Validation")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(epochs)  # Ensure that each epoch is marked clearly
    plt.legend()
    plt.savefig(os.path.join(folder, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()
