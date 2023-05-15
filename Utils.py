import torch
from matplotlib import pyplot as plt
from torch import Tensor, eq, mean, rand
from torch.nn.init import normal_, constant_
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage


def weights_init_normal(m):
    class_name = m.__class__.__name__

    if class_name.find("Conv") != -1:
        normal_(m.weight.data, 0.0, 0.02)

    elif class_name.find("BatchNorm") != -1:
        normal_(m.weight.data, 1.0, 0.02)
        constant_(m.bias.data, 0)


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
    """
    Custom dataset
    """

    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y
        self.transformations = Compose([ToPILImage(),
                                        Resize((28, 28)),
                                        ToTensor(),
                                        Normalize([0.5], [0.5])])

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.transformations(self.x[idx]), self.y[idx]


def custom_mnist(experiences: list[list[int]], device: str = "cpu") -> tuple[Tensor, Tensor, Tensor]:
    # Downloading the MNIST digits
    mnist_data = MNIST('../datasets', download=True, train=True)

    img_x, img_y = mnist_data.data, mnist_data.targets

    for t_ in experiences:
        ids = []
        for n in t_:
            ids.extend(torch.where(mnist_data.targets == n)[0].tolist())

        yield t_, img_x[ids], img_y[ids]


def plot_mnist_eval(source: Tensor, b: int = 2):
    epochs, num_classes, img_size = source.size(0), source.size(1), source.size(2)
    plt.figure(facecolor='white')
    f, axs = plt.subplots(ncols=num_classes, nrows=epochs, figsize=(7, int(0.7 * epochs)))
    for e in range(epochs):
        for c in range(num_classes):
            bordered = torch.zeros(img_size + b * 2, img_size + b * 2).fill_(0.8)
            bordered[b:-b, b:-b] = source[e, c]
            axs[e, c].imshow(bordered)
            axs[e, c].axis("off")
    plt.show()


def gradient_penalty(discriminator, real: Tensor, fake: Tensor, device: str = "cpu") -> Tensor:
    batch_size = real.size(0)
    alpha = rand((batch_size, 1, 1, 1)).repeat(1, 1, 28, 28).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores, _ = discriminator(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(batch_size, -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = mean((gradient_norm - 1) ** 2)
    return 10 * gp


