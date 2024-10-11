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


import matplotlib.pyplot as plt
import numpy as np
import os


def save_metrics_plot(
    train_metrics,
    val_metrics,
    title,
    xlabel,
    ylabel,
    num_epochs,
    folder="results",
    step=2,
):
    """
    Saves a plot comparing training and validation metrics over epochs.

    Args:
    train_metrics (list): A list of metric values from training data.
    val_metrics (list): A list of metric values from validation data.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    num_epochs (int): The total number of epochs.
    folder (str): The directory to save the plot.
    step (int): The step between x-axis ticks.
    """
    os.makedirs(folder, exist_ok=True)
    plt.figure()
    plt.plot(np.arange(1, len(train_metrics) + 1), train_metrics, label="Train")
    plt.plot(np.arange(1, len(val_metrics) + 1), val_metrics, label="Validation")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set x-axis ticks
    plt.xticks(np.arange(1, num_epochs + 1, step))

    plt.legend()
    plt.savefig(os.path.join(folder, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()
