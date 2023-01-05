import os
from pathlib import Path
import sys

import torch

sys.path.append('./utils')
from utils.utils import create_tmp_dir
from datasets import load_mnist_dataset

sys.path.append('./gans/normal_gan')
from train_ngan import train_ngan

torch.manual_seed(0)



def main(params):
    """
    Main function for execution
    
    Args:
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
    dataloader = load_mnist_dataset(dataset_path=cwd_path,
                                    download=True,
                                    batch_size=batch_size,
                                    shuffle=True)

    # train the models
    ## gen: generator
    ## disc: discriminator
    gen, disc = train_ngan(
        dataloader,
        output_path=output_path,
        dataset_path=cwd_path,
        z_dim=params['z_dim'],
        n_epochs=params['n_epochs'],
        display_step=params['display_step'],
        batch_size=params['batch_size'],
        lr=params['lr'],
        device=device,
    )



if __name__ == '__main__':
    model_type = 'ngan'
    
    if model_type == 'ngan':
        params = {
            'z_dim': 64,
            'n_epochs': 50,
            'display_step': 500,
            'batch_size': 128,
            'lr': 1e-5,
        }

    main(params)