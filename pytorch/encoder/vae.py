"""
Sources:
- https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
- Paper: https://arxiv.org/pdf/1312.6114.pdf
"""
import math
import multiprocessing
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning import Trainer, seed_everything
from torch import Tensor
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, image_size: int, h_dim: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    def __init__(self, image_size: int, h_dim: int, z_dim: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.main(z)


class VAE(pl.LightningModule):
    def __init__(self, image_size=784, kl_coeff: float = 0.1, lr: float = 1e-4, z_dim: int = 20, h_dim=400, **kwargs, ):
        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.image_size = image_size
        self.kl_coeff = kl_coeff
        self.lr = lr
        self.encoder = Encoder(image_size, h_dim)
        self.decoder = Decoder(image_size, h_dim, z_dim)
        self.fc_mean = nn.Linear(h_dim, z_dim)
        self.fc_variance = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = self.encoder(x)
        mean = self.fc_mean(x)
        variance = self.fc_variance(x)
        p, q, z = self._reparameterize(mean, variance)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, p, q

    @staticmethod
    def _reparameterize(mean, variance):
        std = torch.exp(variance / 2)
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return p, q, z

    def _step(self, batch, stage: str = 'train'):
        images, _ = batch
        x = images.view(-1, self.image_size)
        x_reconstructed, p, q = self(x)
        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction="mean")
        # https://torchmetrics.readthedocs.io/en/stable/classification/kl_divergence.html
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff
        loss = kl + reconstruction_loss
        tqdm_dict = {
            f"{stage}_reconstruction_loss": reconstruction_loss,
            f"{stage}_kl": kl,
            f"{stage}_loss": loss,
        }
        output = OrderedDict({
            'loss': loss,
            'pred': x_reconstructed,
            f'{stage}_progress_bar': tqdm_dict,
            f'{stage}_log': tqdm_dict
        })
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)

        return output

    def training_step(self, batch: list[Tensor], batch_idx: int):
        return self._step(batch, stage='train')

    def validation_step(self, batch: list[Tensor], batch_idx: int):
        return self._step(batch, stage='val')

    def validation_epoch_end(self, val_step_outputs) -> None:
        with torch.no_grad():
            grid = torchvision.utils.make_grid(val_step_outputs[0]['pred'][:32].reshape(-1, 1, 28, 28), normalize=True)
            self.logger.experiment.add_image('Reconstruction', grid, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main(args: Namespace) -> None:
    seed_everything()
    from pl_bolts.datamodules import MNISTDataModule

    dm = MNISTDataModule.from_argparse_args(args)
    args.image_size = math.prod(dm.size())

    model = VAE(**vars(args))
    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
    # Automatically logs to a directory (by default ``lightning_logs/``)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of devices")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--max_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--kl_coeff", type=float, default=0.1, help="kl_coeff")
    parser.add_argument("--h_dim", type=int, default=400, help="hdim")
    parser.add_argument("--z_dim", type=int, default=20, help="dimensionality of the latent space")

    hparams = parser.parse_args()

    main(hparams)
