import copy
from typing import Dict

from torch import arange, ones, zeros, randint, cat, tensor, stack, normal, Tensor, no_grad
from torch.nn import BCELoss, CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from Discriminator import Discriminator
from Generator import Generator
from Join_retrain import Join_replay
from Utils import weights_init_normal, ExperienceDataset, compute_acc


class Trainer:
    def __init__(self, config: Dict, generator: Generator = None, discriminator: Discriminator = None):

        # Retrieve the parameters
        self.device, self.n_epochs = config["device"], config["n_epochs"]
        self.img_size, self.embedding_dim = config["img_size"], config["embedding"]

        # Create a fixed noise to look how the generator improves
        # the construction of the image during each epoch
        self.eval_noise = normal(0, 1, (config["num_classes"], config["embedding"]), device=self.device)
        self.eval_label = arange(0, config["num_classes"], device=self.device)

        self.eval_progress: list[Tensor] = []

        # Define the generator and discriminator if they are not provided
        if generator is None or discriminator is None:

            self.generator = Generator(
                num_classes=config["num_classes"],
                embedding_dim=config["embedding"],
            ).to(self.device)

            self.discriminator = Discriminator(
                classes=config["num_classes"],
            ).to(self.device)

            self.discriminator.apply(weights_init_normal)
            self.generator.apply(weights_init_normal)
        else:
            self.generator = generator.to(self.device)
            self.discriminator = discriminator.to(self.device)

        # Loss functions
        self.adversarial_loss = BCELoss().to(self.device)
        self.auxiliary_loss = CrossEntropyLoss().to(self.device)

        self.optimizer_g = Adam(self.generator.parameters(), lr=config["lr_g"], betas=(0.5, 0.999))
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=config["lr_d"], betas=(0.5, 0.999))

    def fit_classic(self, experiences, batch_size: int, epch_expr: bool) -> Tensor:

        device, n_epochs = self.device, self.n_epochs
        history, usable_num = [], None

        for idx, (numbers, x, y) in enumerate(experiences):
            # Number that can be generated, because the model have seen
            usable_num = tensor(numbers) if usable_num is None else cat((usable_num, tensor(numbers)))
            print("Experience -- ", idx + 1, "numbers", usable_num.tolist())

            loader = DataLoader(ExperienceDataset(x, y, device), shuffle=True, batch_size=batch_size)
            for epoch in arange(0, n_epochs):
                for real_image, real_label in tqdm(loader):
                    batch_size = real_image.size(0)

                    valid, fake = ones((batch_size, 1), device=device), zeros((batch_size, 1), device=device)

                    # ---- Generator ----
                    self.optimizer_g.zero_grad()

                    gen_label = usable_num[randint(0, len(usable_num), size=(batch_size,))].to(device)
                    fake_img = self.generator(gen_label)

                    dis_output, aux_output = self.discriminator(fake_img)
                    errG = 0.5 * (
                            self.adversarial_loss(dis_output, valid) +
                            self.auxiliary_loss(aux_output, gen_label))

                    errG.backward()
                    self.optimizer_g.step()

                    # ---- Discriminator ----
                    self.optimizer_d.zero_grad()

                    dis_real, aux_real = self.discriminator(real_image)
                    dis_fake, aux_fake = self.discriminator(fake_img.detach())

                    errD = 0.25 * (
                            self.adversarial_loss(dis_real, valid) +
                            self.adversarial_loss(dis_fake, fake) +
                            self.auxiliary_loss(aux_real, real_label) +
                            self.auxiliary_loss(aux_fake, gen_label)
                    )

                    d_acc = compute_acc(
                        cat([aux_real, aux_fake], dim=0),
                        cat([real_label, gen_label], dim=0)
                    )

                    errD.backward()
                    self.optimizer_d.step()

                    history.append(tensor([errD.item(), errG.item(), d_acc]))

                print("[%d/%d] Loss_D: %.4f Loss_G: %.4f Acc %.6f"
                      % (epoch + 1, n_epochs, history[-1][0], history[-1][1],
                         history[-1][2]))

                if epch_expr:  # eval at each epoch
                    with no_grad():
                        self.eval_progress.append(self.generator(self.eval_label, self.eval_noise).squeeze())

            if not epch_expr:  # eval at each experience
                with no_grad():
                    self.eval_progress.append(self.generator(self.eval_label, self.eval_noise).squeeze())

        self.eval_progress = stack(self.eval_progress).detach().cpu()

        return stack(history).T

    def fit_bufferReplay(self, experiences, buff_img: int, batch_size: int) -> Tensor:

        device, n_epochs = self.device, self.n_epochs
        history, usable_num = [], None

        jr = Join_replay(generator=self.generator,
                         batch_size=batch_size,
                         buff_img=buff_img,
                         img_size=self.img_size,
                         device=device)

        for idx, (classes, x, y) in enumerate(experiences):

            loader = jr.create_buffer(idx, usable_num, (x, y))

            # Number that can be generated, because the model have seen
            new_classes = tensor(classes)
            usable_num = new_classes if usable_num is None else cat((usable_num, new_classes))
            print("Experience -- ", idx + 1, "numbers", usable_num.tolist())

            for epoch in arange(0, n_epochs):
                for real_image, real_label in tqdm(loader):
                    batch_size = real_image.size(0)

                    valid, fake = ones((batch_size, 1), device=device), zeros((batch_size, 1), device=device)

                    # ---- Generator ----
                    self.optimizer_g.zero_grad()

                    gen_label = usable_num[randint(0, len(usable_num), size=(batch_size,))].to(device)
                    fake_img = self.generator(gen_label)

                    dis_output, aux_output = self.discriminator(fake_img)
                    errG = 0.5 * (
                            self.adversarial_loss(dis_output, valid) +
                            self.auxiliary_loss(aux_output, gen_label))

                    errG.backward()
                    self.optimizer_g.step()

                    # ---- Discriminator ----
                    self.optimizer_d.zero_grad()

                    dis_real, aux_real = self.discriminator(real_image)
                    dis_fake, aux_fake = self.discriminator(fake_img.detach())

                    errD = 0.25 * (
                            self.adversarial_loss(dis_real, valid) +
                            self.adversarial_loss(dis_fake, fake) +
                            self.auxiliary_loss(aux_real, real_label) +
                            self.auxiliary_loss(aux_fake, gen_label)
                    )

                    d_acc = compute_acc(
                        cat([aux_real, aux_fake], dim=0),
                        cat([real_label, gen_label], dim=0)
                    )

                    errD.backward()
                    self.optimizer_d.step()

                    history.append(tensor([errD.item(), errG.item(), d_acc]))

                print("[%d/%d] Loss_D: %.4f Loss_G: %.4f Acc %.6f"
                      % (epoch + 1, n_epochs, history[-1][0], history[-1][1],
                         history[-1][2]))

            with no_grad():
                self.eval_progress.append(self.generator(self.eval_label, self.eval_noise).squeeze())

        self.eval_progress = stack(self.eval_progress).detach().cpu()
        return stack(history).T

    def fit_replay_alignment(self, experiences, batch_size_: int, lmb_ra: float = 1e-3):

        device, n_epochs = self.device, self.n_epochs
        history, usable_num = [], None

        prev_gen = None
        alignment_loss = MSELoss().to(self.device)

        for idx, (classes, x, y) in enumerate(experiences):

            # Number that can be generated, because the model have seen
            new_classes = tensor(classes)
            usable_num = new_classes if usable_num is None else cat((usable_num, new_classes))
            print("Experience -- ", idx + 1, "numbers", usable_num.tolist())

            loader = DataLoader(ExperienceDataset(x, y, device), shuffle=True, batch_size=batch_size_)
            for epoch in arange(0, n_epochs):
                for real_image, real_label in tqdm(loader):
                    batch_size = real_image.size(0)

                    valid, fake = ones((batch_size, 1), device=device), zeros((batch_size, 1), device=device)

                    # ---- Generator ----
                    self.optimizer_g.zero_grad()

                    gen_label = usable_num[randint(0, len(usable_num), size=(batch_size,))].to(device)
                    fake_img = self.generator(gen_label)

                    dis_output, aux_output = self.discriminator(fake_img)

                    # replay alignment
                    align_loss = 0
                    if prev_gen is not None:
                        z = normal(0, 1, (batch_size_, self.embedding_dim), device=device)
                        gen_label = usable_num[randint(0, len(usable_num), size=(batch_size_,))].to(device)
                        align_loss = lmb_ra * alignment_loss(self.generator(gen_label, z),
                                                             prev_gen(gen_label, z))

                    errG = 0.5 * (
                            self.adversarial_loss(dis_output, valid) +
                            self.auxiliary_loss(aux_output, gen_label) +
                            align_loss)

                    errG.backward()
                    self.optimizer_g.step()

                    # ---- Discriminator ----
                    self.optimizer_d.zero_grad()

                    dis_real, aux_real = self.discriminator(real_image)
                    dis_fake, aux_fake = self.discriminator(fake_img.detach())

                    errD = 0.25 * (
                            self.adversarial_loss(dis_real, valid) +
                            self.adversarial_loss(dis_fake, fake) +
                            self.auxiliary_loss(aux_real, real_label) +
                            self.auxiliary_loss(aux_fake, gen_label)
                    )

                    d_acc = compute_acc(
                        cat([aux_real, aux_fake], dim=0),
                        cat([real_label, gen_label], dim=0)
                    )

                    errD.backward()
                    self.optimizer_d.step()

                    history.append(tensor([errD.item(), errG.item(), d_acc]))

                print("[%d/%d] Loss_D: %.4f Loss_G: %.4f Acc %.6f"
                      % (epoch + 1, n_epochs, history[-1][0], history[-1][1],
                         history[-1][2]))

            prev_gen = copy.deepcopy(self.generator)
            with no_grad():
                self.eval_progress.append(self.generator(self.eval_label, self.eval_noise).squeeze())

        self.eval_progress = stack(self.eval_progress).detach().cpu()
        return stack(history).T
