from matplotlib import pyplot as plt
import torch
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
    """
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()
    plt.close()



if __name__ == '__main__':
    torch.manual_seed(0)
    
    # Test for the function "create_noise()"
    noise = create_noise(1000, 10, 'cpu')
    assert tuple(noise.shape) == (1000, 10)
    assert str(noise.device).startswith('cpu')
    # check a normal distribution or NOT
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01