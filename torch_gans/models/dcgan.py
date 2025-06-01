import torch
import torch.nn as nn
from torch_gans.models.base import BaseGenerator, BaseDiscriminator



class DCGANGenerator(BaseGenerator):
    """
    Generator

    Args:
        z_dim: int, 
            the dimension of the noise vector
        image_channels: int, default = 1
            the number of channels of the images(1: gray scale, 3: color scale)
        hidden_dim: int, 
            The dimension of the noise vector.
        img_shape: tuple[int, int, int], default = (1, 28, 28)
            The shape of the output images (channels, height, width).
        hidden_dim: int, default = 64
            The unit of intermediate-layer dimensions.
    """
    def __init__(self, z_dim=100, image_channels=1, hidden_dim=64, img_shape=(1, 28, 28)):
        super().__init__(latent_dim=z_dim, img_shape=img_shape)
        self.image_channels = image_channels
        self.latent_dim = z_dim
        self.hidden_dim = hidden_dim
        self.img_shape = img_shape
        # Define the network architecture
        self.gen = nn.Sequential(
            self.block(self.latent_dim, hidden_dim * 4), # Use self.latent_dim
            self.block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.block(hidden_dim * 2, hidden_dim),
            self.block(hidden_dim, image_channels, kernel_size=4, final_layer=True),
        )

    def block(self, input_channels, output_channels,
                kernel_size=3, stride=2, final_layer=False):
        """
        A block consisting of multiple layers,
        e.g. nn layer, batchnorm, acrivation

        Args:
            input_channels: int
                the input-feature channels
            output_channels: int
                the output-feature channels
            kernel_size: int, default = 3
                the stride of the convolustion layers
            stride: int, default = 2
                the stride of the convolustion layers
            final_layer: a boolean, default = False
                True if the final layer and False otherwise.
        """
        if final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, 
                                    kernel_size, stride=stride),
                nn.Tanh(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels,
                                    kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

    def unsqueeze_noise(self, noise: torch.Tensor) -> torch.Tensor: # Added type hints
        """
        Unsqueeze a noise tensor:
            (n_samples, latent_dim) --> (n_samples, latent_dim, 1, 1).

        Args:
            noise: A noise tensor, (n_samples, latent_dim).

        Return:
            An unsqueezed noise tensor, (n_samples, latent_dim, 1, 1).
        """
        return noise.view(len(noise), self.latent_dim, 1, 1) # Use self.latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor: # Changed 'noise' to 'z' for clarity, added type hints
        """
        Forward pass of the generator.

        Args:
            z: torch tensor, (n_samples, latent_dim)
                A noise vector.
        Returns:
            Generated images (batch_size, C, H, W).
        """
        # unsqueeze: (n_samples, latent_dim) --> (n_samples, latent_dim, 1, 1)
        x = self.unsqueeze_noise(z)
        return self.gen(x)


class DCGANDiscriminator(BaseDiscriminator):
    """
    Discriminator

    Args:
        img_shape: tuple[int, int, int], default = (1, 28, 28)
            The shape of the input images (channels, height, width).
        hidden_dim: int, default = 16
            The unit of intermediate-layer dimensions.
    """
    def __init__(self, img_shape: tuple[int, int, int] = (1, 28, 28), hidden_dim: int = 16):
        super().__init__(img_shape=img_shape)
        im_chan = self.img_shape[0] # Use self.img_shape from BaseDiscriminator

        self.disc = nn.Sequential(
            self.block(im_chan, hidden_dim),
            self.block(hidden_dim, hidden_dim * 2),
            self.block(hidden_dim * 2, 1, final_layer=True),
        )

    def block(self, input_channels, output_channels, 
                kernel_size=4, stride=2, final_layer=False):
        """
        One block of some layers

        Args:
            input_channels: int
                the input-feature channels
            output_channels: int
                the output-feature channels
            kernel_size: int
            stride: int
            final_layer: a boolean,
                True if the final layer and False otherwise.
        """
        if final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 
                        kernel_size, stride=stride),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 
                        kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor: # Changed 'image' to 'imgs', added type hints
        """
        Forward pass of the discriminator.

        Args:
            imgs: torch tensor, (batch_size, C, H, W)
                Input images.
        Return:
            Discriminator logits (batch_size, 1).
        """
        disc_pred = self.disc(imgs)
        return disc_pred.view(len(disc_pred), -1)