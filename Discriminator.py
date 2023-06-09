from torch import Tensor
from torch.nn import Module, Sequential, LeakyReLU, Conv2d, BatchNorm2d, Linear, Sigmoid, Softmax, Dropout2d


class Discriminator(Module):
    def __init__(self, channels: int, classes: int = 10):
        super(Discriminator, self).__init__()

        self.conv_blocks = Sequential(

            Conv2d(channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            LeakyReLU(negative_slope=0.2, inplace=True),
            Dropout2d(p=0.25, inplace=False),

            Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            LeakyReLU(negative_slope=0.2, inplace=True),
            Dropout2d(p=0.25, inplace=False),
            BatchNorm2d(32, eps=0.8),

            Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            LeakyReLU(negative_slope=0.2, inplace=True),
            Dropout2d(p=0.25, inplace=False),
            BatchNorm2d(64, eps=0.8),

            Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            LeakyReLU(negative_slope=0.2, inplace=True),
            Dropout2d(p=0.25, inplace=False),
            BatchNorm2d(128, eps=0.8)
        )

        # Output layers
        self.discr = Sequential(
            Linear(512, 1),
            Sigmoid())

        self.classifier = Sequential(
            Linear(512, classes),
            Softmax(dim=1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)

        return self.discr(x), self.classifier(x)
