from torch import Tensor, randn, cat
from torch.nn import Linear, ConvTranspose2d, BatchNorm2d, ReLU, Tanh, Module, Sequential, Embedding


class Generator(Module):
    def __init__(self, z_dim: int = 100, num_classes: int = 10, embedding_dim: int = 10):
        super(Generator, self).__init__()

        self.input_dim = z_dim + embedding_dim
        self.hidden_dim = 7 * 7 * 256
        self.z_dim = z_dim

        self.embedding = Embedding(num_classes, embedding_dim)
        self.fc = Linear(self.input_dim, self.hidden_dim)

        self.block = Sequential(
            BatchNorm2d(256),
            ReLU(True),

            ConvTranspose2d(256, 128, kernel_size=5, padding=2, output_padding=0, bias=False),
            BatchNorm2d(128),
            ReLU(True),

            ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(True),

            ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            Tanh()
        )

    def forward(self, label: Tensor, z: Tensor = None) -> Tensor:
        batch_size = label.size(0)

        # If not provided, we generate the random noise
        if z is None:
            z = randn((batch_size, self.z_dim), device=label.device)

        # concat with the conditional label
        input_ = cat([z, self.embedding(label)], dim=1)

        x = self.fc(input_)
        x = self.block(x.view(x.size(0), 256, 7, 7))

        return x
