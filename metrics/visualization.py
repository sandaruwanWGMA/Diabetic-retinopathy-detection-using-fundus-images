import matplotlib.pyplot as plt
import os


def save_plots(metrics, title, epoch, folder="results"):
    os.makedirs(folder, exist_ok=True)
    plt.figure()
    plt.plot(metrics)
    plt.title(f"{title} Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.savefig(os.path.join(folder, f"{title.lower()}_epoch_{epoch}.png"))
    plt.close()
