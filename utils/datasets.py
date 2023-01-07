from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms



def load_mnist_dataset(
        dataset_path='.', 
        download=True, 
        batch_size=64, 
        shuffle=True,
    ):
    """
    Load the MNIST dataset,
    hand-written images of digits (0-9),
    with preprocessing:
        * convert into torch tensor
        * transform the image values into -1 ~ +1

    Args:
        dataset_path: str, default = '.'
            PATH to load the dataset.
            When NOT exists, the dataset will be downloaded.
        download: bool, default = True
            Download the dataset if NOT exist. 
        batch_size: int, default = 64
            mini-batch size
        shuffle: bool, default = True
            shuffle the dataset or NOT
    Returns:
        dataloader: torch.utils.data.DataLoader
    
    example: we can take a batch of the dataset from DataLoader
    -------
    for images, labels in dataloader:
    """
    # Transform the image values into -1 ~ +1, where this range is determined
    # by usage of the tanh activation function in the output layer.
    # Note that, however, DataLoader(MNIST()) is already normalized.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)),  # (mean, std)
    ])

    # Load the MNIST dataset with preprocessing defined at transform.
    # Note that when the dataset does NOT exist in the local environment,
    # the dataset will be downloaded.
    dataloader = DataLoader(
        MNIST(dataset_path, download=download, transform=transform),
        batch_size=batch_size, shuffle=shuffle,
    )

    return dataloader



if __name__ == '__main__':
    import os
    
    path = os.path.dirname(__file__)
    dataloader = load_mnist_dataset(dataset_path=path,
                                    download=True,
                                    batch_size=128,
                                    shuffle=False)
    for images, lbls in dataloader:
        # images: torch.Size([128, 1, 28, 28])
        assert images.ndim == 4
        assert tuple(images.shape) == (128, 1, 28, 28)
        # lbls: torch.Size([128])
        assert tuple(lbls.shape) == (128,)

        break