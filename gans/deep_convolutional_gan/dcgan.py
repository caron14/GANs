from torch import nn



class Generator_DCGAN(nn.Module):
    """
    Generator

    Args:
        z_dim: int, 
            the dimension of the noise vector
        image_channels: int, default = 1
            the number of channels of the images(1: gray scale, 3: color scale)
        hidden_dim: int, 
            the unit of intermediate-layer dimensions
    """
    def __init__(self, z_dim=10, image_channels=1, hidden_dim=64):
        super(Generator_DCGAN, self).__init__()
        self.z_dim = z_dim
        # Define the network architecture
        self.gen = nn.Sequential(
            self.block(z_dim, hidden_dim * 4),
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

    def unsqueeze_noise(self, noise):
        """
        Unsqueeze a noise tensor:
            (n_samples, z_dim) --> (n_samples, z_dim, width, height),
            where width and height = 1 and channels = z_dim.

        Args:
            noise: a noise tensor, (n_samples, z_dim)

        Return:
            an unsqueezed noise tensor, (n_samples, z_dim, width, height)
        """
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        """
        forward pass of the generator

        Args:
            noise: torch tensor, (n_samples, z_dim)
                a noise vector
        """
        # unsqueeze: (n_samples, z_dim) --> (n_samples, z_dim, 1, 1),
        # where (1, 1) indicates (width, height), respectively.
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


class Discriminator_DCGAN(nn.Module):
    """
    Discriminator

    Args:
        image_channels: int, default = 1
            the number of channels of the images(1: gray scale, 3: color scale)
        hidden_dim: the unit of intermediate-layer dimensions
    """
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator_DCGAN, self).__init__()
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

    def forward(self, image):
        """
        Args:
            image: torch tensor, (image_dim)
                a flattened image tensor
        Return:
            torch tensor, (image_dim)
                prediction probability
        """
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)



if __name__ == '__main__':
    pass