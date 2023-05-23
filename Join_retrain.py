from torch import Tensor, zeros, full, cat, no_grad
from torch.utils.data import DataLoader

from Generator import Generator
from Utils import ExperienceDataset


class Join_retrain:
    def __init__(self, generator: Generator, batch_size: int, buff_img: int, channels: int,
                 img_size: int, device: str = "cpu"):
        """
        Manage the join retraining in an online setting
        """
        self.g = generator
        self.img_size = img_size
        self.batch_size = batch_size
        self.buff_img = buff_img
        self.device = device
        self.channels = channels

    def create_buffer(self, id_exp: int, past_classes: Tensor,
                      source: tuple[Tensor, Tensor]) -> DataLoader:

        real_image, real_label = source
        device = self.device

        if id_exp == 0:  # No previous experience (first experience)
            return DataLoader(ExperienceDataset(real_image, real_label, device),
                              shuffle=True,
                              batch_size=self.batch_size)

        elif id_exp > 0:  # generating buffer replay

            # Define the number of images to generate, we allocate a fixed number of slots for each number of class
            # encountered
            img_to_create = self.buff_img * past_classes.size(0)
            gen_buffer = zeros((img_to_create, self.channels, self.img_size, self.img_size))

            self.g.eval()
            with no_grad():
                count = 0
                for i in past_classes:  # for each class encountered

                    # since the buffer may have high dimension, we generate image in batch fashion
                    to_generate = self.buff_img
                    while to_generate > 0:
                        batch_size = min(256, to_generate)
                        gen_label = full((batch_size,), i, device=device)
                        gen_buffer[count:count + batch_size] = self.g(gen_label).cpu()

                        count += batch_size
                        to_generate -= batch_size
            self.g.train()

            # In the end, we concat the replay generated and the current batch of image (new classes)
            custom_x = cat((real_image, gen_buffer), dim=0)
            custom_y = cat(
                (real_label, past_classes.repeat_interleave(self.buff_img)),
                dim=0)

            return DataLoader(ExperienceDataset(custom_x, custom_y, device),
                              shuffle=True,
                              batch_size=self.batch_size)
