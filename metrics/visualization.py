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
    train_metrics, val_metrics, title, xlabel, ylabel, epoch, folder="results"
):
    """
    Saves a plot comparing training and validation metrics over epochs.

    Args:
    train_metrics (list): A list of metric values from training data.
    val_metrics (list): A list of metric values from validation data.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    epoch (int): The current epoch, used for naming the file.
    folder (str): The directory to save the plot.
    """
    os.makedirs(folder, exist_ok=True)
    plt.figure()
    plt.plot(range(len(train_metrics)), train_metrics, label="Train")
    plt.plot(range(len(val_metrics)), val_metrics, label="Validation")
    plt.title(f"{title} Over Epochs")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(
        os.path.join(folder, f"{title.lower().replace(' ', '_')}_epoch_{epoch}.png")
    )
    plt.close()
