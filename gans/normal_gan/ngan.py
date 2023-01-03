import torch
from torch import nn



class Generator_NGAN(nn.Module):
    """
    Generator
    
    Args:
        z_dim: the dimension of the noise vector
        image_dim: int, default = 784(= 28 * 28)
            the dimensiton of flatten images
        hidden_dim: the unit of intermediate-layer dimensions
    """
    def __init__(self, z_dim=10, image_dim=784, hidden_dim=32):
        super(Generator_NGAN, self).__init__()
        self.z_dim = z_dim
        # Define the network architecture
        self.gen = nn.Sequential(
            self.block(z_dim, hidden_dim),
            self.block(hidden_dim, hidden_dim * 2),
            self.block(hidden_dim * 2, hidden_dim * 4),
            self.block(hidden_dim * 4, hidden_dim * 8),
            self.block(hidden_dim * 8, image_dim, final_layer=True),
        )

    def block(self, input_dim, output_dim, final_layer=False):
        """
        One block of some layers

        Args:
            input_dim: int
                the input dimension
            output_channels: int
                the output dimension
            final_layer: a boolean,
                True if the final layer and False otherwise.
        """
        if final_layer:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Sigmoid(),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
            )

    def forward(self, noise):
        """
        Args:
            noise: torch tensor, (n_samples, z_dim)
                a noise vector
        Return:
            generated images, torch.tensor (image_dim,)
        """
        return self.gen(noise)


class Discriminator_NGAN(nn.Module):
    """
    Discriminator

    Args:
        image_dim: int, default = 784(= 28 * 28)
            the dimensiton of flatten images
        hidden_dim: the unit of intermediate-layer dimensions
    """
    def __init__(self, image_dim=784, hidden_dim=32):
        super(Discriminator_NGAN, self).__init__()
        # Define the network architecture
        self.disc = nn.Sequential(
            self.block(image_dim, hidden_dim * 4),
            self.block(hidden_dim * 4, hidden_dim * 2),
            self.block(hidden_dim * 2, hidden_dim),
            self.block(hidden_dim, 1, final_layer=True),
        )

    def block(self, input_dim, output_dim, final_layer=False):
        """
        One block of some layers

        Args:
            input_dim: int
                the input dimension
            output_channels: int
                the output dimension
            final_layer: a boolean,
                True if the final layer and False otherwise.
        """
        if final_layer:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                # nn.BatchNorm1d(output_dim),
                nn.LeakyReLU(negative_slope=0.2),
            )

    def forward(self, image):
        """
        Args:
            image: torch tensor, (image_dim)
                a flattened image tensor
        Return:
            prediction probability, torch tensor
        """
        return self.disc(image)



if __name__ == '__main__':
    """
    Test for the generator
    """
    input_dim = 100
    gen = Generator_NGAN(z_dim=10, image_dim=input_dim, hidden_dim=32)
    
    # Check the layer size and type
    ## hidden-layer block
    output_dim = 20
    hidden_block = gen.block(input_dim, output_dim)
    assert len(hidden_block) == 3
    assert type(hidden_block[0]) == nn.Linear
    assert type(hidden_block[1]) == nn.BatchNorm1d
    assert type(hidden_block[2]) == nn.ReLU
    ## final-layer block
    final_block = gen.block(input_dim, output_dim, final_layer=True)
    assert len(final_block) == 2
    assert type(final_block[0]) == nn.Linear
    assert type(final_block[1]) == nn.Sigmoid

    # Check the output size
    ## create the test noise(input)
    n_samples, noise_dim = 5000, 100
    test_noise = torch.randn(n_samples, noise_dim)
    ## generate the test image(output)
    test_output = hidden_block(test_noise)
    assert tuple(test_output.shape) == (n_samples, output_dim)
    test_output = final_block(test_noise)
    assert tuple(test_output.shape) == (n_samples, output_dim)


    """
    Test for the discriminator
    """
    input_dim, hiddlen_dim = 1000, 100
    disc = Discriminator_NGAN(image_dim=input_dim, hidden_dim=hiddlen_dim)
    assert len(disc.disc) == 4  # There are the 4 blocks
    
    # Check the layer size and type
    ## hidden-layer block
    hidden_block = disc.block(input_dim, hiddlen_dim)
    assert len(hidden_block) == 2
    assert type(hidden_block[0]) == nn.Linear
    assert type(hidden_block[1]) == nn.LeakyReLU
    ## final-layer block
    final_block = disc.block(input_dim, 1, final_layer=True)
    assert len(final_block) == 1
    assert type(final_block[0]) == nn.Linear

    # Check the output size
    ## create the test noise(input)
    n_samples, noise_dim = 5000, input_dim
    test_noise = torch.randn(n_samples, noise_dim)
    ## generate the test image(output)
    test_output = hidden_block(test_noise)
    assert tuple(test_output.shape) == (n_samples, hiddlen_dim)
    test_output = final_block(test_noise)
    assert tuple(test_output.shape) == (n_samples, 1)
    test_output = disc(test_noise)
    assert tuple(test_output.shape) == (n_samples, 1)