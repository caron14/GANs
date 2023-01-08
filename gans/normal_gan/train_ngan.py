import sys

import torch
from torch import nn
from tqdm import tqdm

sys.path.append('./gans')
from helper_functions import create_noise
from helper_functions import show_images
from gans.loss_func_gans import generator_loss
from gans.loss_func_gans import discriminator_loss

sys.path.append('./gans/normal_gan')
from gans.normal_gan.ngan import Generator_NGAN
from gans.normal_gan.ngan import Discriminator_NGAN



def train_ngan(
        dataloader,
        output_path=None,
        z_dim=64,
        n_epochs=100,
        display_step=500,
        lr=1e-5,
        device='cpu',
    ):
    """
    Train the model
    
    Args:
        dataloader: torch.utils.data.DataLoader
            DataLoader to recieve a batch dataset in each loop
        output_path: pathlib.Path
            output directory PATH
        n_epochs: int, default = 100
            the number of epochs
        display_step: int, default = 500
            the number of step to show the intermidiate result
        lr: float, default = 1e-5
            learning rate
        device: str, default = 'cpu'
            the device name, e.g. 'cpu', 'cuda', 'cuda:0'
        -------------------------------------------------
        batch_size: int, default = 128
            minibatch size
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
    gen = Generator_NGAN(z_dim).to(device)
    ## Discriminator
    disc = Discriminator_NGAN().to(device)
    
    # Define a optimizer
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    """
    Train the generator and discriminator models
    """
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    gen_loss = False
    img_idx = 1
    for epoch in range(n_epochs):
        print(f"epoch {epoch}")
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)

            # Flatten the batch of real images from the dataset
            real = real.view(cur_batch_size, -1).to(device)

            ## Train discriminator part
            # Initialize the optimizer
            disc_opt.zero_grad()
            # Calculate discriminator loss
            disc_loss = discriminator_loss(gen, disc, criterion, real, 
                                        cur_batch_size, z_dim, device)
            # Update the disctiminator model params
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            ## Train generator part
            # Initialize the optimizer
            gen_opt.zero_grad()
            # Calculate the generator loss
            gen_loss = generator_loss(gen, disc, criterion, 
                                    cur_batch_size, z_dim, device)
            # Update the generator model params
            gen_loss.backward(retain_graph=True)
            gen_opt.step()

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            # Visualization of results at the specific step
            if cur_step % display_step == 0 and cur_step > 0:
                # print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = create_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                show_images(fake, save_path=output_path, filename=f"{img_idx}_fake.png")
                show_images(real, save_path=output_path, filename=f"{img_idx}_real.png")
                img_idx += 1
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
        print()  # pause

    return gen, disc



if __name__ == '__main__':
    pass