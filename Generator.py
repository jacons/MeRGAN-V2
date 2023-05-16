from torch import Tensor, mul, normal
from torch.nn import Linear, BatchNorm2d, Tanh, Module, Sequential, Embedding, Upsample, Conv2d, \
    LeakyReLU


class Generator(Module):
    def __init__(self, num_classes: int = 10, embedding_dim: int = 100, img_size: int = 32):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(num_classes, embedding_dim)
        self.fc = Linear(embedding_dim, 128 * self.init_size ** 2)

        self.conv_blocks = Sequential(
            BatchNorm2d(128),

            Upsample(scale_factor=2),
            Conv2d(128, 128, 3, stride=1, padding=1),
            BatchNorm2d(128, 0.8),
            LeakyReLU(0.2, inplace=True),

            Upsample(scale_factor=2),
            Conv2d(128, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, 0.8),
            LeakyReLU(0.2, inplace=True),

            Conv2d(64, 1, 3, stride=1, padding=1),
            Tanh(),
        )

    def forward(self, labels: Tensor, z: Tensor = None) -> Tensor:
        batch_size = labels.size(0)

        # If not provided, we generate the random noise
        if z is None:
            z = normal(0, 1, (batch_size, self.embedding_dim), device=labels.device)

        # concat with the conditional label
        input_ = mul(self.embedding(labels), z)

        out = self.fc(input_)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        return self.conv_blocks(out)
