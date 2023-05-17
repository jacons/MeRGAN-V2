import os
import shutil

import torch
from matplotlib import pyplot as plt
from torch import Tensor, eq, arange, cat
from torch.nn.init import normal_, constant_
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from Generator import Generator


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        normal_(m.weight.data, 0.0, 0.02)

    elif class_name.find("BatchNorm2d") != -1:
        normal_(m.weight.data, 1.0, 0.02)
        constant_(m.bias.data, 0.0)


def compute_acc(predicted: Tensor, labels: Tensor):
    """
    Compute the accuracy of model
    :param predicted: label predicted by discriminator
    :param labels:  true label
    :return:
    """
    correct = eq(predicted.argmax(dim=1), labels).sum().item()
    return float(correct) / float(labels.size(0))


class ExperienceDataset(Dataset):

    def __init__(self, image: Tensor, target: Tensor, device: str = "cpu"):
        self.image, self.target = image, target
        self.device = device

    def __len__(self):
        return self.image.size(0)

    def __getitem__(self, idx):
        return self.image[idx].to(self.device), self.target[idx].to(self.device)


def generate_mnist_dataset():
    if not os.path.exists("../single_digit"):
        print("Dataset don't found...")

        os.makedirs("../single_digit")

        # Downloading the MNIST digits
        transformations = Compose(
            [Resize((32, 32)), ToTensor(), Normalize([0.5], [0.5])])
        mnist_data = MNIST('temp_mnist', download=True, train=True, transform=transformations)

        print("Preprocessing numbers...")
        x, y = next(iter(DataLoader(mnist_data, shuffle=False, batch_size=mnist_data.data.size(0))))

        for n in arange(0, 10):
            print("Saving number:", n.item())
            idx = torch.where(y == n)[0]
            torch.save([x[idx], y[idx]], "../single_digit/num_" + str(n.item()) + ".pt")

        shutil.rmtree("temp_mnist")
    else:
        print("Dataset found...")


def custom_mnist(experiences: list[list[int]]) -> tuple[list, Tensor, Tensor]:
    generate_mnist_dataset()

    for t_ in experiences:
        img_x, img_y = None, None

        for n in t_:
            num_x, num_y = torch.load("../single_digit/num_" + str(n) + ".pt")

            img_x = num_x if img_x is None else cat([img_x, num_x])
            img_y = num_y if img_y is None else cat([img_y, num_y])

        yield t_, img_x, img_y


def plot_mnist_eval(source: Tensor):
    epochs, num_classes, img_size = source.size(0), source.size(1), source.size(2)

    if epochs == 1:
        f, axs = plt.subplots(ncols=num_classes, nrows=epochs, figsize=(7, 1))
        f.patch.set_facecolor('black')
        for c in arange(num_classes):
            axs[c].imshow(-source[0, c], cmap="binary")
            axs[c].axis("off")

    elif epochs > 1:
        f, axs = plt.subplots(ncols=num_classes, nrows=epochs, figsize=(7, int(0.7 * epochs)))
        f.patch.set_facecolor('black')
        for e in arange(epochs):
            y_offset = (axs[e, 0].get_position().y0 - axs[e, 0].get_position().y1) / 2 + axs[e, 0].get_position().y1
            f.text(0.006, y_offset, "Epochs = " + str(e.item() + 1), fontsize=10, color="white")
            for c in arange(num_classes):
                axs[e, c].imshow(-source[e, c], cmap="binary")
                axs[e, c].axis("off")
    plt.show()


def generate_classes(g: Generator, num_classes: int, rows: int, device: str):
    f, axs = plt.subplots(ncols=num_classes, nrows=rows, figsize=(7, int(0.7 * rows)))
    f.patch.set_facecolor('black')
    labels = arange(0, 10, device=device)

    for c in arange(num_classes):
        x_offset = (axs[0, c].get_position().x0 - axs[0, c].get_position().x1) / 2 + axs[0, c].get_position().x1
        f.text(x_offset - 0.025, 0.92, "t = " + str(c.item()), fontsize=10, color="white")

    with torch.no_grad():
        for e in arange(rows):
            images = g(labels).squeeze().cpu()
            y_offset = (axs[e, 0].get_position().y0 - axs[e, 0].get_position().y1) / 2 + axs[e, 0].get_position().y1
            f.text(0.006, y_offset, "Gen = " + str(e.item() + 1), fontsize=10, color="white")
            for c in arange(num_classes):
                axs[e, c].imshow(-images[c], cmap="binary")
                axs[e, c].axis("off")
    plt.show()


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

    plt.show()
