from torch import Tensor
from torch.nn import Module, Sequential, LeakyReLU, Conv2d, BatchNorm2d, Linear, Sigmoid, Softmax, Flatten


class Discriminator(Module):
    def __init__(self, classes: int = 10, in_channels: int = 28, img_size=28):
        super(Discriminator, self).__init__()

        self.conv_blocks = Sequential(
            Conv2d(1, in_channels, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(in_channels),
            LeakyReLU(0.2),

            Conv2d(in_channels, in_channels * 2, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(in_channels * 2),
            LeakyReLU(0.2),

            Conv2d(in_channels * 2, in_channels * 4, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(in_channels * 4),
            LeakyReLU(0.2),

            Flatten()
        )

        self.discr = Sequential(
            Linear(in_channels * 4 * (img_size // 8) ** 2, 1),
            Sigmoid()
        )
        self.classifier = Sequential(
            Linear(in_channels * 4 * (img_size // 8) ** 2, classes),
            Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_blocks(x)

        return self.discr(x), self.classifier(x)
