"""
Sources:
- https://arxiv.org/pdf/1406.2661.pdf
- https://github.com/goodfeli/adversarial/blob/master/mnist.yaml
- https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html
- https://github.com/nocotan/pytorch-lightning-gans
- https://github.com/eriklindernoren/PyTorch-GAN/
"""
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, img_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(img_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_size),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.mlp(x)


class GAN(pl.LightningModule):
    def __init__(self, batch_size=32, img_size=28 * 28 * 1, z_dim=64, workers=0, lr=3e-4, b1=0.9, b2=0.9999, **kwargs):
        super().__init__()
        # Save for checkpoint
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.img_size = img_size
        self.z_dim = z_dim
        self.workers = workers
        self.lr = lr
        self.betas = (b1, b2)

        self.fixed_noise = torch.randn((batch_size, z_dim))

        self.generator = Generator(self.z_dim, self.img_size)
        self.discriminator = Discriminator(self.img_size)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch: list[Tensor], batch_idx: int, optimizer_idx: int):
        real_images, _ = batch  # real_images is (32,1,28,28) tensor

        valid = torch.ones(real_images.size(0), 1).type_as(real_images)  # valid is (32,1) tensor with full of 1s
        fake = torch.zeros(real_images.size(0), 1).type_as(real_images)  # valid is (32,1) tensor with full of 0s

        # Sample once to reduce the computation overhead
        fake_images = self.generate_fake_images(real_images)  # fake_images is (32,784)

        # Discriminator
        if optimizer_idx == 0:
            # real_images is converted from (32,1,28,28) to (32,784)
            predict_real_label = self.discriminator(real_images.view(-1, self.img_size))
            loss_discriminator_real = self.adversarial_loss(predict_real_label, valid)
            self.log("d_real_loss", loss_discriminator_real, prog_bar=True, on_epoch=True)

            predict_fake_label = self.discriminator(fake_images.detach())
            loss_discriminator_fake = self.adversarial_loss(predict_fake_label, fake)
            self.log("d_fake_loss", loss_discriminator_real, prog_bar=True, on_epoch=True)

            loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2
            self.log("d_loss", loss_discriminator, prog_bar=True, on_epoch=True)
            tqdm_dict = {'d_loss': loss_discriminator}
            output = OrderedDict({
                'loss': loss_discriminator,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # Generator
        if optimizer_idx == 1:
            predict_fake_label = self.discriminator(fake_images)  # D(G(z))
            # maximize(D(G(z))) instead of minimize(1-D(G(z))) since tensor saturates
            loss_generator = self.adversarial_loss(predict_fake_label, valid)
            self.log("g_loss", loss_generator, prog_bar=True, on_epoch=True)

            tqdm_dict = {'g_loss': loss_generator}
            output = OrderedDict({
                'loss': loss_generator,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            return output

    def generate_fake_images(self, real_images):
        # sample noise
        z = torch.randn(real_images.shape[0], self.z_dim).type_as(real_images)
        fake_images = self.generator(z)
        return fake_images

    def on_epoch_end(self):
        # log sampled images
        fake_samples = self.generator(self.fixed_noise).reshape(-1, 1, 28, 28)
        grid = torchvision.utils.make_grid(fake_samples, normalize=True)
        self.logger.experiment.add_image('MNIST Fake Images', grid, self.current_epoch)

    def configure_optimizers(self):
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)

        return [optimizer_discriminator, optimizer_generator], []

    def train_dataloader(self):
        mnist_dataset = datasets.MNIST(root="dataset/",
                                       transform=transforms.Compose(
                                           [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
                                       ),
                                       download=True)
        data_loader = DataLoader(mnist_dataset, batch_size=self.batch_size, shuffle=True)
        return data_loader


def main(args: Namespace) -> None:
    model = GAN(**vars(args))

    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
    # Automatically logs to a directory (by default ``lightning_logs/``)
    trainer = Trainer(devices=args.devices, accelerator="auto", max_epochs=50)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--devices", type=int, default=0, help="number of devices")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--z_dim", type=int, default=64,
                        help="dimensionality of the latent space")

    hparams = parser.parse_args()

    main(hparams)
