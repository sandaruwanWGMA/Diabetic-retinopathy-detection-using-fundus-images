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
