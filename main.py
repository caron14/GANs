import os
from pathlib import Path
import sys

import torch

sys.path.append('./utils')
from utils.utils import create_tmp_dir

sys.path.append('./gans/normal_gan')
from train_ngan import train_ngan

torch.manual_seed(0)


def main():
    # current work directory
    cwd_path = os.path.dirname(__file__)
    # Create the output directory if NOT exist,
    # or Remove the previous result and recreate the one.
    create_tmp_dir(cwd_path)
    
    # Specify a device type
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train the models
    z_dim = 64
    n_epochs = 50
    display_step = 500
    batch_size = 128
    lr = 1e-5
    ## gen: generator
    ## disc: discriminator
    gen, disc = train_ngan(
        dataset_path=cwd_path,
        z_dim=z_dim,
        n_epochs=n_epochs,
        display_step=display_step,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )



if __name__ == '__main__':
    main()