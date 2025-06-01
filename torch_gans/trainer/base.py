import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Tuple, Optional, Dict

# Assuming BaseGenerator and BaseDiscriminator will be accessible via torch_gans.models.base
# For now, to make this self-contained for the subtask, we can use stubs or import them if the structure allows.
# In a real scenario, ensure torch_gans is in PYTHONPATH or use relative imports if appropriate.
from torch_gans.models.base import BaseGenerator, BaseDiscriminator


class BaseGANModule(pl.LightningModule):
    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: BaseDiscriminator,
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        b1_g: float = 0.5,
        b2_g: float = 0.999,
        b1_d: float = 0.5,
        b2_d: float = 0.999,
        latent_dim: Optional[int] = None, # Added for sample_noise, can be derived from generator
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['generator', 'discriminator']) # Saves lr, betas, latent_dim

        self.generator = generator
        self.discriminator = discriminator

        # latent_dim can also be accessed via self.generator.latent_dim if BaseGenerator ensures it
        self.latent_dim = latent_dim if latent_dim is not None else getattr(generator, 'latent_dim', 100)


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''Delegates to the generator's forward pass.'''
        return self.generator(z)

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        '''Samples random noise vectors. Delegates to generator if possible, otherwise implements.'''
        if hasattr(self.generator, 'sample_noise') and callable(self.generator.sample_noise):
            return self.generator.sample_noise(batch_size, device=self.device)
        else:
            # Fallback if generator doesn't have its own sample_noise
            return torch.randn(batch_size, self.latent_dim, device=self.device)

    def generator_loss(self, generated_imgs: torch.Tensor, disc_output_on_generated: torch.Tensor) -> torch.Tensor:
        '''
        Computes the generator loss.
        This method should be overridden by subclasses for specific GAN loss formulations (e.g., minimax, Wasserstein).

        Args:
            generated_imgs: Images produced by the generator.
            disc_output_on_generated: Output of the discriminator when run on generated images.

        Returns:
            A scalar tensor representing the generator loss.
        '''
        raise NotImplementedError("generator_loss must be implemented in a subclass.")

    def discriminator_loss(
        self,
        real_imgs: torch.Tensor,
        generated_imgs: torch.Tensor,
        disc_output_on_real: torch.Tensor,
        disc_output_on_generated: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Computes the discriminator loss.
        This method should be overridden by subclasses for specific GAN loss formulations.

        Args:
            real_imgs: Real images from the dataset.
            generated_imgs: Images produced by the generator.
            disc_output_on_real: Output of the discriminator on real images.
            disc_output_on_generated: Output of the discriminator on generated (fake) images.

        Returns:
            A scalar tensor representing the discriminator loss.
        '''
        raise NotImplementedError("discriminator_loss must be implemented in a subclass.")

    def training_step(self, batch: Tuple[torch.Tensor, Any], batch_idx: int, optimizer_idx: int):
        real_imgs, _ = batch  # Assuming labels are not needed for unconditional GANs

        # Sample noise
        z = self.sample_noise(batch_size=real_imgs.size(0))
        generated_imgs = self(z) # Calls self.forward(z) -> self.generator(z)

        # Train generator
        if optimizer_idx == 0:
            disc_output_on_generated = self.discriminator(generated_imgs)
            g_loss = self.generator_loss(generated_imgs, disc_output_on_generated)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # Train discriminator
        if optimizer_idx == 1:
            disc_output_on_real = self.discriminator(real_imgs)
            # Detach generated_imgs to avoid backpropagating through G when training D
            disc_output_on_generated = self.discriminator(generated_imgs.detach())

            d_loss = self.discriminator_loss(
                real_imgs, generated_imgs.detach(), disc_output_on_real, disc_output_on_generated
            )
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self) -> Tuple[list, list]:
        lr_g = self.hparams.lr_g
        lr_d = self.hparams.lr_d
        b1_g = self.hparams.b1_g
        b2_g = self.hparams.b2_g
        b1_d = self.hparams.b1_d
        b2_d = self.hparams.b2_d

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(b1_g, b2_g))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(b1_d, b2_d))

        return [opt_g, opt_d], [] # No learning rate schedulers for now

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[Dict[str, Any]]:
        # Optional: Implement validation logic, e.g., log generated images
        # For now, let's keep it simple
        pass

    def test_step(self, batch: Any, batch_idx: int) -> Optional[Dict[str, Any]]:
        # Optional: Implement test logic
        pass
