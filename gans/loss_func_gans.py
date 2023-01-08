import sys

import torch

sys.path.append('./gans')
from helper_functions import create_noise



def generator_loss(gen, disc, criterion, num_images, z_dim, device):
    """
    Calculate the generator loss
    
    Argss:
        gen: the Generator_NGAN class instance
            generator model
        disc: the Discriminator_NGAN class instance
            discriminator model
        criterion: the function instance
            BCE loss function (fake = 0, real = 1)
        num_images: int
            the number of images
        z_dim: int
            the dimension of the noise vector
        device: str
            the device type

    Returns:
        gen_loss: a torch scalar, a generator loss
    """
    # create noise vectors
    noise_vector = create_noise(num_images, z_dim, device=device)
    
    # generate the fake images
    fake_images = gen(noise_vector)
    
    # prediction of discriminator against the fake image
    pred_fake = disc(fake_images)
    
    # calculate the loss
    gen_loss = criterion(pred_fake, torch.ones_like(pred_fake))

    return gen_loss


def discriminator_loss(gen, disc, criterion, real, num_images, z_dim, device):
    """
    Calculate the discriminator loss
    
    Args:
        gen: the Generator_NGAN class instance
            generator model
        disc: the Discriminator_NGAN class instance
            discriminator model
        criterion: the function instance
            BCE loss function (fake = 0, real = 1)
        real: torch tensor, (batch size, image_dim)
            real images
        num_images: int
            the number of images
        z_dim: int
            the dimension of the noise vector
        device: str
            the device type

    Returns:
        disc_loss: a torch scalar, a discriminator loss
    """
    # create a noise vector
    noise_vector = create_noise(num_images, z_dim, device=device)

    # generate fake images
    fake_images = gen(noise_vector)

    # prediction of discriminator against the fake image
    fake = fake_images.detach()
    pred_fake = disc(fake)

    # BCE loss for the fake images
    loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

    # prediction of discriminator against the real images
    pred_real = disc(real)

    # BCE loss for real images
    loss_real = criterion(pred_real, torch.ones_like(pred_real))

    # average the loss between real and fake
    disc_loss = 0.5 * (loss_fake + loss_real)

    return disc_loss



if __name__ == '__main__':
    pass