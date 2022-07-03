"""
A logger to download scalar metric plots and generated images.
It is useful in case Tensorboard can not be used visually i.e. on kaggle
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


class GANCLILogger:
    def __init__(self, prefix: str = ""):
        self.prefix = prefix + '_' if prefix != "" else ""
        self.fake_images = []
        self.losses = {}

    def log(self, name: str, value, global_step: int = 0, current_epoch: int = 0):
        if name not in self.losses:
            self.losses[name] = []

        if isinstance(value, Tensor):
            value = value.detach().cpu().numpy()

        self.losses[name].append(value)

    def download_plots(self):
        for i in self.losses:
            plt.figure(figsize=(20, 15))
            plt.title({i})
            plt.plot(self.losses[i], label={i})
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"{self.prefix}{i}.svg")

        fig = plt.figure(figsize=(16, 16))
        plt.axis("off")
        images = [[plt.imshow(np.transpose(i.detach().cpu(), (1, 2, 0)), animated=True)] for i in self.fake_images]
        ani = animation.ArtistAnimation(fig, images, interval=1000, repeat_delay=1000, blit=True)
        ani.save(f"{self.prefix}fake_images_progress.mp4")
