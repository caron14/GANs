from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid



def create_noise(n_samples, z_dim, device='cpu'):
    """
    Create a noise vector,
    whose elements are generated from the normal distribution.
    
    Args:
        n_samples: int
            the number of samples to generate
        z_dim: int
            the dimension of the noise vector
        device: str, default = 'cpu'
            the device name, e.g. 'cpu', 'cuda', 'cuda:0'
    Return:
        noise vector: torch tensor (n_samples, z_dim)
    """
    return torch.randn(n_samples, z_dim, device=device)


def show_images(
        image_tensor,
        num_images=25,
        size=(1, 28, 28),
        save_path=None,
        filename='image.png',
    ):
    """
    Visualize images from tensor of images
    
    Args:
        image_tensor: torch.tensor
            a tensor of images
        num_images: int, default = 25
            number of images
        size: tuple of int, default = (1, 28, 28)
            size of a image
        save_path: pathlib.Path, default = None
            output directory PATH
        filename: str, default = 'image.png'
            filename to be saved
    """
    plt.figure(figsize=(6, 6))
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5, padding=1)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if save_path:
        plt.savefig(save_path / filename)
    plt.show()
    plt.close()


def labels_to_one_hot(labels, n_classes):
    """
    Convert the labels into one-hot vectors

    Args:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    
    Return:
        torch tensor, (number of samples, number of the classes)
    """
    return F.one_hot(labels, num_classes=n_classes)



if __name__ == '__main__':
    torch.manual_seed(0)

    # Test for the function "create_noise()"
    noise = create_noise(1000, 10, 'cpu')
    assert tuple(noise.shape) == (1000, 10)
    assert str(noise.device).startswith('cpu')
    # check a normal distribution or NOT
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01