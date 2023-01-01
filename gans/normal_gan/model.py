import torch
from torch import nn



class Generator(nn.Module):
    """
    Generator
    
    Args:
        z_dim: the dimension of the noise vector
        image_dim: int, default = 784(= 28 * 28)
            the dimensiton of flatten images
        hidden_dim: the unit of intermediate-layer dimensions
    """
    def __init__(self, z_dim=10, image_dim=784, hidden_dim=32):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Define the network architecture
        self.gen = nn.Sequential(
            self.one_block(z_dim, hidden_dim),
            self.one_block(hidden_dim, hidden_dim * 2),
            self.one_block(hidden_dim * 2, hidden_dim * 4),
            self.one_block(hidden_dim * 4, hidden_dim * 8),
            self.one_block(hidden_dim * 8, image_dim, final_layer=True),
        )

    def one_block(self, input_dim, output_dim, final_layer=False):
        """
        One block of each layer

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
        """
        # unsqueeze: (n_samples, z_dim) --> (n_samples, z_dim, 1, 1),
        # where (1, 1) is width and height, respectively.
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)



if __name__ == '__main__':
    """
    Test for the generator
    """
    gen = Generator()
    
    # Check the layer size and type
    ## hidden-layer block
    input_dim, output_dim = 100, 20
    hidden_block = gen.one_block(input_dim, output_dim)
    assert len(hidden_block) == 3
    assert type(hidden_block[0]) == nn.Linear
    assert type(hidden_block[1]) == nn.BatchNorm1d
    assert type(hidden_block[2]) == nn.ReLU
    ## final-layer block
    final_block = gen.one_block(input_dim, output_dim, final_layer=True)
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