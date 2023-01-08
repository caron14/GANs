import pytest
import torch
import torch.nn as nn

from gans.normal_gan.ngan import Generator_NGAN
from gans.normal_gan.ngan import Discriminator_NGAN



def test_Generator_NGAN():
    """
    Test for the generator of Normal GAN
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


def test_Discriminator_NGAN():
    """
    Test for the discriminator of Normal GAN
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



if __name__ == '__main__':
    test_Generator_NGAN()