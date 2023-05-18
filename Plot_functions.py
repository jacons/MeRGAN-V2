from matplotlib import pyplot as plt
from torch import Tensor, arange, no_grad

from Generator import Generator

"""
def plot_mnist_eval(source: Tensor):
    epochs, num_classes, img_size = source.size(0), source.size(1), source.size(2)

    f, axs = plt.subplots(ncols=num_classes, nrows=epochs, figsize=(7, max(int(0.7 * epochs), 1)))
    f.patch.set_facecolor('black')

    if epochs == 1:
        for c in range(num_classes):
            axs[c].imshow(-source[0, c], cmap="binary")
            axs[c].axis("off")

    elif epochs > 1:
        for e in range(epochs):
            y_offset = (axs[e, 0].get_position().y0 - axs[e, 0].get_position().y1) / 2 + axs[e, 0].get_position().y1
            f.text(0.006, y_offset, "Epochs = " + str(e + 1), fontsize=10, color="white")

            for c in range(num_classes):
                axs[e, c].imshow(-source[e, c], cmap="binary")
                axs[e, c].axis("off")
    plt.show()
"""


def generate_classes(g: Generator, num_classes: int, rows: int, device: str):
    rows = max(rows, 2)

    f, axs = plt.subplots(ncols=num_classes, nrows=rows, figsize=(7, int(0.7 * rows)))
    f.patch.set_facecolor('black')
    labels = arange(0, 10, device=device)

    for c in range(num_classes):
        x_offset = (axs[0, c].get_position().x0 - axs[0, c].get_position().x1) / 2 + axs[0, c].get_position().x1
        f.text(x_offset - 0.025, 0.92, "t = " + str(c), fontsize=10, color="white")

    with no_grad():
        for e in range(rows):
            y_offset = (axs[e, 0].get_position().y0 - axs[e, 0].get_position().y1) / 2 + axs[e, 0].get_position().y1
            f.text(0.006, y_offset, "Gen = " + str(e + 1), fontsize=10, color="white")

            images = g(labels).cpu()
            for c in range(num_classes):

                if images.size(2) == 1:
                    axs[e, c].imshow(-images[c, 0], cmap="binary")
                else:
                    axs[e, c].imshow(-images[c])

                axs[e, c].axis("off")
    plt.show()


def save_grid(images: Tensor, rows: int, name_file: str, num_classes: int = 10):
    f, axs = plt.subplots(ncols=num_classes, nrows=rows, figsize=(7, int(0.7 * rows)))
    f.patch.set_facecolor('black')

    for c in range(num_classes):
        x_offset = (axs[0, c].get_position().x0 - axs[0, c].get_position().x1) / 2 + axs[0, c].get_position().x1
        f.text(x_offset - 0.025, 0.92, "t = " + str(c), fontsize=10, color="white")

    for e in range(rows):
        y_offset = (axs[e, 0].get_position().y0 - axs[e, 0].get_position().y1) / 2 + axs[e, 0].get_position().y1
        f.text(0.006, y_offset, "Gen = " + str(e + 1), fontsize=10, color="white")

        for c in range(num_classes):
            if images.size(2) == 1:
                axs[e, c].imshow(-images[c, 0], cmap="binary")
            else:
                axs[e, c].imshow(-images[c])
            axs[e, c].axis("off")

    plt.savefig(name_file)


def plot_history(history: Tensor):
    fig = plt.figure(figsize=(20, 12))
    sub_figs = fig.subfigures(1, 2)

    ax_img = sub_figs[0].subplots(nrows=2)
    ax_img[0].plot(history[0], label="Discriminator loss", color="blue", alpha=0.8)
    ax_img[0].grid()
    ax_img[0].legend()
    ax_img[0].set_xlabel("Updates")
    ax_img[0].set_ylabel("Loss value")
    ax_img[0].set_title("Discriminator loss")

    ax_img[1].plot(history[1], label="Generator loss", color="green", alpha=0.8)
    ax_img[1].grid()
    ax_img[1].legend()
    ax_img[1].set_xlabel("Updates")
    ax_img[1].set_ylabel("Loss value")
    ax_img[1].set_title("Generator loss")

    ax_c = sub_figs[1].subplots()
    ax_c.plot(history[2], label="Accuracy", color="orange", alpha=0.8)
    ax_c.grid()
    ax_c.legend()
    ax_c.set_xlabel("Updates")
    ax_c.set_ylabel("Accuracy")
    ax_c.set_title("Accuracy")

    plt.tight_layout()
    plt.show()
