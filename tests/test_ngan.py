import pytest
import torch
import torch.nn as nn

from gans.normal_gan.ngan import Generator_NGAN



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



if __name__ == '__main__':
    test_Generator_NGAN()