import os
import shutil

import torch
from torch import Tensor, eq, cat
from torch.nn.init import normal_, constant_
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class ExperienceDataset(Dataset):

    def __init__(self, image: Tensor, target: Tensor, device: str = "cpu"):
        self.image, self.target = image, target
        self.device = device

    def __len__(self):
        return self.image.size(0)

    def __getitem__(self, idx):
        return self.image[idx].to(self.device), self.target[idx].to(self.device)


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        normal_(m.weight.data, 0.0, 0.02)

    elif class_name.find("BatchNorm2d") != -1:
        normal_(m.weight.data, 1.0, 0.02)
        constant_(m.bias.data, 0.0)


def compute_acc(predicted: Tensor, labels: Tensor):
    """
    Compute the accuracy of model both for real and fake images
    :param predicted: label predicted by discriminator
    :param labels:  true label
    :return:
    """
    correct = eq(predicted.argmax(dim=1), labels).sum().item()
    return float(correct) / float(labels.size(0))


def generate_mnist_dataset(dataset_path: str, temp_path: str = "temp_mnist"):
    if not os.path.exists(dataset_path):
        print("Dataset not found...")
        os.makedirs(dataset_path)

        # Downloading the MNIST digits
        transformations = Compose([Resize((32, 32)), ToTensor(), Normalize([0.5], [0.5])])
        mnist_data = MNIST(temp_path, download=True, train=True, transform=transformations)

        # a straightforward trick to apply all the transformation
        print("Preprocessing numbers...")
        dataloader = DataLoader(mnist_data, shuffle=False, batch_size=mnist_data.data.size(0))
        x, y = next(iter(dataloader))

        # we save all preprocessed digits into separate files
        for n in range(10):
            print("Saving number:", n)
            idx = torch.where(y == n)[0]
            torch.save([x[idx], y[idx]], os.path.join(dataset_path, f"num_{n}.pt"))

        shutil.rmtree(temp_path)
    else:
        print("Dataset found...")


def custom_mnist(experiences: list[list[int]], dataset_path: str = "../single_digit") -> tuple[list, Tensor, Tensor]:

    # check if the dataset is ready
    generate_mnist_dataset(dataset_path)

    for t_ in experiences:  # iterate each experience
        img_x, img_y = None, None

        for n in t_:  # for each experience concatenate the numbers
            num_x, num_y = torch.load(f"{dataset_path}/num_{n}.pt")

            img_x = num_x if img_x is None else cat([img_x, num_x])
            img_y = num_y if img_y is None else cat([img_y, num_y])

        yield t_, img_x, img_y
