import os
import matplotlib.pyplot as plt


def save_plots(metrics, title, epoch, folder="results"):
    os.makedirs(folder, exist_ok=True)
    plt.figure()

    # Ensure the tensor is detached and converted to a NumPy array
    if metrics.requires_grad:
        metrics = metrics.detach().numpy()
    else:
        metrics = metrics.numpy()

    plt.plot(metrics)
    plt.title(f"{title} Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.savefig(os.path.join(folder, f"{title.lower()}_epoch_{epoch}.png"))
    plt.close()
