import pytest
import torch
import torch.nn as nn

from gans.helper_functions import create_noise
from gans.deep_convolutional_gan.dcgan import Generator_DCGAN
from gans.deep_convolutional_gan.dcgan import Discriminator_DCGAN



def test_Generator_DCGAN():
    # Define the generator part
    gen = Generator_DCGAN(z_dim=10, image_channels=1, hidden_dim=64)
    # Define the hidden-layer block
    input_dim, output_dim = 10, 20
    hidden_block = gen.block(input_dim, output_dim, kernel_size=4, stride=1)
    # Define the final-layer block
    final_block = gen.block(input_dim, output_dim,
                            kernel_size=3, stride=2, final_layer=True)
    
    # Define the hidden-layer block with the different stride value
    hidden_block_other_stride = gen.block(output_dim, output_dim, 
                                        kernel_size=4, stride=2)

    """
    Check the layer size and type
    """
    # hidden-layer block
    assert len(hidden_block) == 3
    assert type(hidden_block[0]) == nn.ConvTranspose2d
    assert type(hidden_block[1]) == nn.BatchNorm2d
    assert type(hidden_block[2]) == nn.ReLU
    # final-layer block
    assert len(final_block) == 2
    assert type(final_block[0]) == nn.ConvTranspose2d
    assert type(final_block[1]) == nn.Tanh

    """
    Check the output size of hidden and final blocks
    """
    # create the hidden-block noise
    n_samples, noise_dim = 100, gen.z_dim
    hidden_block_noise = create_noise(n_samples, noise_dim)
    unsqueeze_hidden_block_noise = gen.unsqueeze_noise(hidden_block_noise)
    # test the output size
    output = hidden_block(unsqueeze_hidden_block_noise)
    assert tuple(output.shape) == (n_samples, output_dim, 4, 4)
    output = hidden_block_other_stride(output)
    assert tuple(output.shape) == (n_samples, 20, 10, 10)

    # create the final-block noise
    # n_samples, noise_dim = 100, gen.z_dim
    final_block_noise = create_noise(n_samples, noise_dim)
    unsqueezed_final_block_noise = gen.unsqueeze_noise(final_block_noise)
    output = final_block(unsqueezed_final_block_noise)
    print(final_block_noise.shape)
    print(unsqueezed_final_block_noise.shape)
    print(output.shape)
    assert tuple(final_block_noise.shape) == (n_samples, noise_dim)
    assert tuple(unsqueezed_final_block_noise.shape) == (n_samples, noise_dim, 1, 1)
    assert tuple(output.shape) == (n_samples, output_dim, 3, 3)

    # Generator's output
    gen_noise = create_noise(n_samples, noise_dim)
    unsqueezed_gen_noise = gen.unsqueeze_noise(gen_noise)
    gen_output = gen(unsqueezed_gen_noise)
    assert tuple(gen_output.shape) == (n_samples, 1, 28, 28)


# def test_Discriminator_DCGAN():
#     """
#     Test for the discriminator of DCGAN
#     """
#     input_dim, hiddlen_dim = 1000, 100
#     disc = Discriminator_DCGAN(image_dim=input_dim, hidden_dim=hiddlen_dim)
#     assert len(disc.disc) == 4  # There are the 4 blocks

#     # Check the layer size and type
#     ## hidden-layer block
#     hidden_block = disc.block(input_dim, hiddlen_dim)
#     assert len(hidden_block) == 2
#     assert type(hidden_block[0]) == nn.Linear
#     assert type(hidden_block[1]) == nn.LeakyReLU
#     ## final-layer block
#     final_block = disc.block(input_dim, 1, final_layer=True)
#     assert len(final_block) == 1
#     assert type(final_block[0]) == nn.Linear

#     # Check the output size
#     ## create the test noise(input)
#     n_samples, noise_dim = 5000, input_dim
#     test_noise = torch.randn(n_samples, noise_dim)
#     ## generate the test image(output)
#     test_output = hidden_block(test_noise)
#     assert tuple(test_output.shape) == (n_samples, hiddlen_dim)
#     test_output = final_block(test_noise)
#     assert tuple(test_output.shape) == (n_samples, 1)
#     test_output = disc(test_noise)
#     assert tuple(test_output.shape) == (n_samples, 1)



if __name__ == '__main__':
    test_Generator_DCGAN()
    # test_Discriminator_DCGAN()