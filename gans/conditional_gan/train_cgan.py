import sys

import torch
from torch import nn
from tqdm import tqdm

sys.path.append('./gans')
from helper_functions import create_noise
from helper_functions import show_images
from helper_functions import labels_to_one_hot

sys.path.append('./gans/conditioanl_gan')
from gans.conditional_gan.cgan import Generator_CGAN
from gans.conditional_gan.cgan import Discriminator_CGAN



def combine_vectors(vec1, vec2):
    """
    Concatenate the two vectors.
    
    Args:
        vec1(vec2): torch tensor, (n_samples, n_features)
        (Ex. vec1 is image tensor and vec2 is one-hot-label vector)

    Return:
        concatenated vector
    """
    return torch.cat((vec1.to(torch.float32), vec2.to(torch.float32)), dim=1)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def train_cgan(
        dataloader,
        normalize_mean=None,
        normalize_std=None,
        output_path=None,
        z_dim=64,
        n_classes=10,
        n_epochs=100,
        display_step=500,
        lr=2e-4,
        beta_1=0.5,
        beta_2=0.999,
        device='cpu',
    ):
    """
    Train the model
    
    Args:
        dataloader: torch.utils.data.DataLoader
            DataLoader to recieve a batch dataset in each loop
        normalize_mean: tuple, default = None
            mean values for normalization by transforms.Normalize()
            e.g. (0.5,)
        normalize_std: tuple, default = None
            standard deviation(std) values for normalization by transforms.Normalize()
            e.g. (0.5,)
        output_path: pathlib.Path
            output directory PATH
        z_dim: int
            noise dimension
        n_classes: int
            number of the classes, e.g. 10 for MNIST
        n_epochs: int, default = 100
            the number of epochs
        display_step: int, default = 500
            the number of step to show the intermidiate result
        lr: float, default = 1e-5
            learning rate
        device: str, default = 'cpu'
            the device name, e.g. 'cpu', 'cuda', 'cuda:0'
        beta_1: float, default = 0.5
            momentum parameter for Adam optimizer
        beta_2: float, default = 0.999
            momentum parameter for Adam optimizer

    Return:
        noise vector: torch tensor (n_samples, z_dim)
    """
    # Loss function: Binary Cross Entropy Logit loss
    criterion = nn.BCEWithLogitsLoss()

    """
    Initialization
    """
    # Define a model
    ## Generator
    gen = Generator_CGAN(z_dim + n_classes).to(device)
    ## Discriminator, Note: 3 is the channnel of MNIST image
    disc = Discriminator_CGAN(im_chan=3 + n_classes).to(device)

    # Initialize the weights of the model into the normal distribution.
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    # Define a optimizer
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

    """
    Train the generator and discriminator models
    """
    cur_step = 0
    discriminator_losses = []
    generator_losses = []
    img_idx = 1
    for epoch in range(n_epochs):
        print(f"epoch {epoch}")
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(device)

            """
            One-hot labels
            """
            one_hot_labels = labels_to_one_hot(labels.to(device), n_classes)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = one_hot_labels.repeat(1, 1, 28, 28)

            ## Train discriminator part
            # Initialize the optimizer
            disc_opt.zero_grad()
            # generate noise for fake images
            fake_noise = create_noise(cur_batch_size, z_dim, device=device)
            # Concatenate the noise and label vectors
            noise_and_lbl_vec = combine_vectors(fake_noise, one_hot_labels)
            # generate the fake images
            fake = gen(noise_and_lbl_vec)

            # Prediction from the discriminator
            fake_imgs_with_lbls = combine_vectors(fake.detach(), image_one_hot_labels)
            real_imgs_with_lbls = combine_vectors(real, image_one_hot_labels)
            disc_fake_pred = disc(fake_imgs_with_lbls)
            disc_real_pred = disc(real_imgs_with_lbls)

            # loss
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()
            discriminator_losses += [disc_loss.item()]


            ## Train generator part
            # Initialize the optimizer
            gen_opt.zero_grad()
            # Concatenate the fake images and labels
            fake_imgs_and_lbls = combine_vectors(fake, image_one_hot_labels)
            # prediction and loss
            disc_fake_pred = disc(fake_imgs_and_lbls)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            # Update the generator model params
            gen_loss.backward()
            gen_opt.step()
            generator_losses += [gen_loss.item()]


            # Visualization of results at the specific step
            if cur_step % display_step == 0 and cur_step > 0:
                # Save the images
                show_images(fake, save_path=output_path, filename=f"{img_idx}_fake.png")
                show_images(real, save_path=output_path, filename=f"{img_idx}_real.png")
                img_idx += 1
            cur_step += 1
        print()  # pause

    return gen, disc



if __name__ == '__main__':
    vec1 = torch.tensor([[1, 2], [3, 4]])
    vec2 = torch.tensor([[5, 6]])
    ans = torch.tensor([[1, 2], [3, 4], [5, 6]])
    assert torch.all( combine_vectors(vec1, vec2) == ans )