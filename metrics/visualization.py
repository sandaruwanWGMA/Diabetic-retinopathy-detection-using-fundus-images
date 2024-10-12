import os
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib.pyplot as plt
import numpy as np
import os


def save_plots(metrics, title, num_epochs, folder="results", step=2):
    """
    Saves a plot of given metrics over the epochs with customizable x-axis ticks.

    Args:
    metrics (list): A list containing the metric to be plotted. Each metric can be a float or a tensor.
    title (str): The title for the plot and the y-axis.
    num_epochs (int): The total number of epochs which will be used to determine x-axis labels.
    folder (str): The folder where to save the plot.
    step (int): The interval at which to place x-axis ticks.
    """
    os.makedirs(folder, exist_ok=True)
    plt.figure()

    # Check each metric and convert it appropriately
    metrics_np = np.array([m.item() if hasattr(m, "item") else m for m in metrics])

    # Generate epoch indices based on the actual number of metrics provided
    epochs = np.arange(1, len(metrics) + 1)

    plt.plot(epochs, metrics_np, marker="o")  # Optional: Add marker for visibility
    plt.title(f"{title} Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(title)

    # Setting the ticks on the x-axis
    plt.xticks(np.arange(1, len(metrics) + 1, step))

    # Save the plot
    plt.savefig(os.path.join(folder, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()


def save_metrics_plot(
    train_metrics,
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
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set x-axis ticks
    plt.xticks(np.arange(1, num_epochs + 1, step))

    plt.legend()
    plt.savefig(os.path.join(folder, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()
