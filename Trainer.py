from typing import Dict

import torch
from torch import randn, arange, ones, zeros, randint, cat, tensor, stack
from torch.nn import BCELoss, NLLLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from Discriminator import Discriminator
from Generator import Generator
from Utils import weights_init_normal, compute_acc, ExperienceDataset, gradient_penalty2


class Trainer:
    def __init__(self, config: Dict, generator: Generator = None, discriminator: Discriminator = None):
        # Retrieve the parameters
        self.device = config["device"]
        self.n_epochs = config["n_epochs"]
        num_classes = config["num_classes"]
        # the latent dimension is composed by the one-hot encoding of the number of class + the noise dimension
        z_dim = config["z_dim"]

        # Create a fixed noise to look how the generator improves the construction of the
        # image during each epoch
        self.eval_noise = randn((num_classes, z_dim), device=self.device)
        self.eval_label = arange(0, num_classes, device=self.device)

        self.eval_progress = []  # empty((config["n_epochs"], num_classes, config["img_size"], config["img_size"]))

        # Define the generator and discriminator if they are not provided
        if generator is None or discriminator is None:

            self.generator = Generator(
                z_dim=config["z_dim"],
                num_classes=num_classes,
            ).to(self.device)

            self.discriminator = Discriminator(
                classes=num_classes,
                img_size=config["img_size"]
            ).to(self.device)

            self.discriminator.apply(weights_init_normal)
            self.generator.apply(weights_init_normal)
        else:
            self.generator = generator
            self.discriminator = discriminator

        # Loss functions
        self.adversarial_loss = BCELoss().to(self.device)
        self.auxiliary_loss = CrossEntropyLoss().to(self.device)

        self.optimizer_g = Adam(self.generator.parameters(), lr=config["lr_g"], betas=(0.5, 0.999))
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=config["lr_d"], betas=(0.5, 0.999))

    def fit_classic(self, experiences, batch_size: int, epch_expr: bool):

        device, n_epochs = self.device, self.n_epochs
        history = []

        usable_num = None  # Number that can be generated, because the model have seen

        for idx, (numbers, x, y) in enumerate(experiences):

            usable_num = tensor(numbers) if usable_num is None else cat((usable_num, tensor(numbers)))

            print("Experience -- ", idx + 1, "numbers", usable_num)

            loader = DataLoader(ExperienceDataset(x, y),
                                shuffle=True, batch_size=batch_size)

            for epoch in arange(0, n_epochs):
                for real_image, real_label in tqdm(loader):
                    real_image, real_label = real_image.to(device), real_label.to(device)
                    batch_size = real_image.size(0)

                    valid, fake = ones((batch_size, 1), device=device), zeros((batch_size, 1), device=device)

                    # ###########################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    # ##########################
                    self.optimizer_d.zero_grad()

                    dis_output, aux_output = self.discriminator(real_image)

                    errD_real = 0.5 * (
                            self.adversarial_loss(dis_output, valid) +
                            self.auxiliary_loss(aux_output, real_label))

                    errD_real.backward()

                    accuracy_real = compute_acc(aux_output, real_label)

                    gen_label = usable_num[randint(0, len(usable_num), size=(batch_size,))].to(device)

                    fake_img = self.generator(gen_label)

                    dis_output, aux_output = self.discriminator(fake_img.detach())
                    errD_fake = 0.5 * (
                            self.adversarial_loss(dis_output, fake) +
                            self.auxiliary_loss(aux_output, gen_label))

                    errD_fake.backward()
                    errD = errD_real + errD_fake
                    self.optimizer_d.step()

                    # Clip weights of discriminator
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-0.1, 0.1)

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################

                    self.optimizer_g.zero_grad()
                    dis_output, aux_output = self.discriminator(fake_img)
                    errG = 0.5 * (
                            self.adversarial_loss(dis_output, valid) +
                            self.auxiliary_loss(aux_output, gen_label))

                    errG.backward()
                    self.optimizer_g.step()

                    accuracy_fake = compute_acc(aux_output, gen_label)

                    history.append(tensor([errD.item(), errG.item(), accuracy_real, accuracy_fake]))

                print("[%d/%d] Loss_D: %.4f Loss_G: %.4f Acc real: %.6f Acc fake %.6f"
                      % (epoch + 1, n_epochs, history[-1][0], history[-1][1],
                         history[-1][2], history[-1][3]))

                if epch_expr:  # eval at each epoch
                    self.eval_progress.append(self.generator(self.eval_label, self.eval_noise).squeeze())

            if not epch_expr:  # eval at each experience
                self.eval_progress.append(self.generator(self.eval_label, self.eval_noise).squeeze())

        self.eval_progress = stack(self.eval_progress).detach().cpu()

        return stack(history).T

    def fit_wgan_gp(self, experiences, batch_size: int, epch_expr: bool,
                    critic_iterations: int = 5, lambda_gp: int = 10):

        device, n_epochs = self.device, self.n_epochs
        history = []
        lam_gp = 10
        n_critic = 5
        errG, accuracy_fake = 0, 0
        usable_num = None  # Number that can be generated, because the model have seen

        self.auxiliary_loss = CrossEntropyLoss().to(self.device)
        for idx, (numbers, x, y) in enumerate(experiences):

            usable_num = tensor(numbers) if usable_num is None else cat((usable_num, tensor(numbers)))

            print("Experience -- ", idx + 1, "numbers", usable_num)

            loader = DataLoader(ExperienceDataset(x, y),
                                shuffle=True, batch_size=batch_size)

            for epoch in arange(0, n_epochs):
                for i, (real_image, real_label) in enumerate(tqdm(loader)):
                    real_image, real_label = real_image.to(device), real_label.to(device)
                    batch_size = real_image.size(0)

                    gen_label = usable_num[randint(0, len(usable_num), size=(batch_size,))].to(device)
                    fake_img = self.generator(gen_label)

                    self.optimizer_d.zero_grad()

                    dis_real, aux_real = self.discriminator(real_image)
                    errD_real = 0.5 * (-torch.mean(dis_real) + self.auxiliary_loss(aux_real, real_label))

                    dis_fake, aux_fake = self.discriminator(fake_img)
                    errD_fake = 0.5 * (torch.mean(dis_fake) + self.auxiliary_loss(aux_fake, gen_label))

                    gp = gradient_penalty2(real_image.detach(), fake_img.detach(), self.discriminator, device)

                    errD = 0.5 * (errD_real + errD_fake) + lam_gp * gp

                    errD.backward()
                    self.optimizer_d.step()

                    accuracy_real = compute_acc(aux_real, real_label)

                    if i % n_critic == 0:
                        self.optimizer_g.zero_grad()

                        # Calculate G loss
                        dis_fake, aux_fake = self.discriminator(fake_img)

                        errG = 0.5 * (-torch.mean(dis_fake) + self.auxiliary_loss(aux_fake, gen_label))

                        # Update G
                        errG.backward(retain_graph=True)
                        self.optimizer_g.step()

                        accuracy_fake = compute_acc(aux_fake, gen_label)

                    history.append(tensor([errD.item(), errG.item(), accuracy_real, accuracy_fake]))

                print("[%d/%d] Loss_D: %.4f Loss_G: %.4f Acc real: %.6f Acc fake %.6f"
                      % (epoch + 1, n_epochs, history[-1][0], history[-1][1],
                         history[-1][2], history[-1][3]))

                if epch_expr:  # eval at each epoch
                    self.eval_progress.append(self.generator(self.eval_label, self.eval_noise).squeeze())

            if not epch_expr:  # eval at each experience
                self.eval_progress.append(self.generator(self.eval_label, self.eval_noise).squeeze())

        self.eval_progress = stack(self.eval_progress).detach().cpu()

        return stack(history).T
