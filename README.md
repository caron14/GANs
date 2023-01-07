# Implementation of GANs

This repository is for the practice of implementation of GAN(generative adversarial network).
The models, which can generate hand-written images of digits (0-9), are implemented by [PyTorch](https://pytorch.org/).
To build the model, we use the MNIST dataset, which contains 60,000 images of handwritten digits(0 ~ 9).

# GAN Models

Each of the models listed below is implemented with a method of different features.
Each model improves on the shortcomings of the GAN in turn.
As a result, the model can generate more high-quality images.

## Normal GAN

A normal GAN consists fully connected layers.
In the DCGAN model, this model will be extended by using convolutional layers.
The main features are as follows:

* Only fully connected layers, without convolutional layers.
* In the generator, ReLU activation is used in hidden layers, and sigmoid activation is used in an output layer.
* BatchNorm is used except in the output layer.
* In the discriminator, LeakyReLU activation is used in hidden layers, and no activation is used in an output layer.
* BatchNorm is NOT used.

## DCGAN

[paper](https://arxiv.org/abs/1511.06434)
A deep convolutional GAN(DCGAN) is a model which the normal GAN is extended by convolutional layers.
The main features are as follows:

* Only convolutional layers without fully connected layers.
* In the generator, ReLU activation is used in hidden layers, and Tanh activation is used in an output layer.
* In the discriminator, LeakyReLU activation is used in hidden layers, and no activation is used in an output layer.
* BatchNorm is used, whereas Dropout is NOT used.

## CGAN

A conditional GAN(CGAN) generates images dependent on the conditions, e.g. labels of hand-written images of digits.
The main features are as follows:

* The basic structure of CGAN is the same as DCGAN
* In only the generator, the conditional vector is added.s

## WGAN

Wasserstein GAN(WGAN) is a model which solves generally known problem of GANs, the stability of training a model.
For example, the one of the problem is known as mode collapse.
The cause of this problem stems from the loss function, i.e. Binary Cross Entropy(BCE) loss.
When using the BCE loss, we will encounter the vanishing gradient phenomenon.
The WGAN solves this issue by improving the loss function.
Specifically, this is accomplished by adding the Wasserstein distance to the BCE loss.

## Getting started

You can get started in building the GANs models in this repository by the following procedure. To prepare an environment, we use 
[docker](https://www.docker.com/).

1. Create a docker image from Dockerfile.

    ```bash
    docker build -t gan:practice .
    ```

2. Run a docker container created from the docker image.

    ```bash
    docker run -it --name (container name) -v ~/(local dir)/:/work (IMAGE ID) bash
    ```

    Note that "--name (container name)" is an instruction to add a container name.
    The second option "-v ~/(local dir)/:/work" is an instruction
    to connect the local directory with the container directory.

3. Execute the python main script(main.py).

    ```bash
    python main.py
    ```