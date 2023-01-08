import os
from pathlib import Path
import sys

import torch

sys.path.append('./utils')
from utils.utils import create_tmp_dir
from datasets import load_mnist_dataset

sys.path.append('./gans/normal_gan')
from train_ngan import train_ngan

sys.path.append('./gans/deep_convolutional_gan')
from train_dcgan import train_dcgan

torch.manual_seed(0)



def main(model_type, params):
    """
    Main function for execution
    
    Args:
        model_type: str
            Model-type name, e.g. 'ngan', 'dcgan'
        params: dict
            hyper params
    """
    # current work directory
    # --> the save directory at this script(main.py)
    cwd_path = os.path.dirname(__file__)
    
    # Create the output directory if NOT exist,
    # or Remove the previous result and recreate the one.
    output_path = Path(cwd_path) / 'output'
    create_tmp_dir(output_path)
    
    # Specify a device type
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get the DataLoader:
    # load or download the dataset and return as DataLoader
    dataloader = load_mnist_dataset(model_type,
                                    dataset_path=cwd_path,
                                    download=True,
                                    batch_size=params['batch_size'],
                                    shuffle=True)

    # train the models
    ## gen: generator
    ## disc: discriminator
    if model_type == 'ngan':
        gen, disc = train_ngan(
            dataloader,
            output_path=output_path,
            z_dim=params['z_dim'],
            n_epochs=params['n_epochs'],
            display_step=params['display_step'],
            lr=params['lr'],
            device=device,
        )
    elif model_type == 'dcgan':
        gen, disc = train_dcgan(
            dataloader,
            output_path=output_path,
            z_dim=params['z_dim'],
            n_epochs=params['n_epochs'],
            display_step=params['display_step'],
            lr=params['lr'],
            beta_1=params['beta_1'],
            beta_2=params['beta_2'],
            device=device,
        )
    else:
        print(f"model_type = {model_type} is NOT supported.")
        return None



if __name__ == '__main__':
    """
    GAN type
    --------
    ngan: Normal GAN
    dcgan: Deep Convolutional GAN(DCGAN)
    """
    model_type = 'dcgan'
    
    if model_type == 'ngan':
        params = {
            'z_dim': 64,
            'n_epochs': 50,
            'display_step': 500,
            'batch_size': 128,
            'lr': 1e-5,
        }
    elif model_type == 'dcgan':
        params = {
            'z_dim': 64,
            'n_epochs': 50,
            'display_step': 500,
            'batch_size': 128,
            'lr': 2e-4,
            'beta_1': 0.5,
            'beta_2': 0.999,
        }
    else:
        print(f"model_type = {model_type} is NOT supported.")
        sys.exit()

    main(model_type, params)