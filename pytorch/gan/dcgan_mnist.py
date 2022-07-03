"""
Sources:
- gan_mlp.py
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
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
    def __init__(self, ndf: int, nc: int):
        super().__init__()
        self.cnn = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cnn(x)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.cnn = nn.Sequential(
            # input is Z, going into a convolution
            # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
            # z = batch_size x 100x1x1
            # (1-1)×1 - 2×0 + 1×(4-1) + 0+1 = 4
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.cnn(x)


class DCGAN(pl.LightningModule):
    def __init__(self, batch_size=32, z_dim: int = 100, nc: int = 1, ngf: int = 64, ndf: int = 64, workers: int = 0,
                 lr=3e-4,
                 b1=0.5, b2=0.9999, **kwargs):
        super().__init__()
        # Save for checkpoint
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.z_dim = z_dim
        self.workers = workers
        self.lr = lr
        self.betas = (b1, b2)

        self.fixed_noise = torch.randn(32, z_dim, 1, 1).to(self.device)

        self.generator = Generator(self.z_dim, self.ngf, self.nc)
        self.generator.apply(weights_init)

        self.discriminator = Discriminator(self.ndf, self.nc)
        self.discriminator.apply(weights_init)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch: list[Tensor], batch_idx: int, optimizer_idx: int):
        real_images, _ = batch  # real_images is (32,1,64,64) tensor
        b_size = real_images.size(0)
        real_label = 1.
        fake_label = 0.

        valid = torch.full((b_size,), real_label, dtype=torch.float).to(self.device)
        fake = torch.full((b_size,), fake_label, dtype=torch.float).to(self.device)

        # Sample once to reduce the computation overhead
        fake_images = self.generate_fake_images(b_size, self.z_dim)  # fake_images is (32,28x28x1)

        # Discriminator
        if optimizer_idx == 0:
            # view(-1) converts from (32,1,2,2) to (32,)
            predict_real_label = self.discriminator(real_images).view(-1)
            loss_discriminator_real = self.adversarial_loss(predict_real_label, valid)
            self.log("d_real_loss", loss_discriminator_real, prog_bar=True, on_epoch=True)

            predict_fake_label = self.discriminator(fake_images.detach()).view(-1)
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
            # view(-1) converts from (32,1,2,2) to (32,)
            predict_fake_label = self.discriminator(fake_images).view(-1)  # D(G(z))
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

    def generate_fake_images(self, b_size: int, z_dim: int):
        # sample noise
        z = torch.randn(b_size, z_dim, 1, 1).to(self.device)  # b_size x z_dimx1x1
        fake_images = self.generator(z)  # b_size x 1xngfxngf
        return fake_images

    def on_epoch_end(self):
        with torch.no_grad():
            # log sampled images
            fake_samples = self.generator(self.fixed_noise.to(self.device))
            grid = torchvision.utils.make_grid(fake_samples[:32], normalize=True)
            self.logger.experiment.add_image('MNIST Fake Images', grid, self.current_epoch)

    def configure_optimizers(self):
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)

        return [optimizer_discriminator, optimizer_generator], []

    def train_dataloader(self):
        mnist_dataset = datasets.MNIST(root="dataset/",
                                       transform=transforms.Compose([
                                           transforms.Resize((self.ngf, self.ngf)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5 for _ in range(self.nc)],
                                                                [0.5 for _ in range(self.nc)]),
                                       ]),
                                       download=True)
        data_loader = DataLoader(mnist_dataset, batch_size=self.batch_size, shuffle=True)
        return data_loader


# randomly initialized from a Normal distribution with mean=0, stdev=0.02
# as mentioned in the DCGAN paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main(args: Namespace) -> None:
    model = DCGAN(**vars(args))

    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
    # Automatically logs to a directory (by default ``lightning_logs/``)
    trainer = Trainer(gpus=args.gpus, max_epochs=args.max_epochs)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of devices")
    parser.add_argument("--max_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--z_dim", type=int, default=100,
                        help="dimensionality of the latent space")

    hparams = parser.parse_args()

    main(hparams)
