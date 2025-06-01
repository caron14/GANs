import torch
import torch.nn as nn

class BaseGenerator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: tuple[int, int, int]):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Generates images from latent vectors.

        Args:
            z: Latent vectors (batch_size, latent_dim).

        Returns:
            Generated images (batch_size, C, H, W).
        '''
        raise NotImplementedError("Generator's forward method must be implemented by a subclass.")

    def sample_noise(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        '''
        Samples random noise vectors.

        Args:
            batch_size: Number of noise vectors to sample.
            device: The device to create the tensor on.

        Returns:
            A tensor of shape (batch_size, self.latent_dim).
        '''
        return torch.randn(batch_size, self.latent_dim, device=device)


class BaseDiscriminator(nn.Module):
    def __init__(self, img_shape: tuple[int, int, int]):
        super().__init__()
        self.img_shape = img_shape

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        '''
        Predicts the probability of images being real.

        Args:
            imgs: Input images (batch_size, C, H, W).

        Returns:
            Discriminator logits (batch_size, 1).
        '''
        raise NotImplementedError("Discriminator's forward method must be implemented by a subclass.")
