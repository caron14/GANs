import torch
import torch.nn.functional as F
from typing import Any, Tuple, Dict, Optional

# Attempt absolute import first, assuming torch_gans is in path
# and models.__init__.py is correctly set up.
from torch_gans.models.dcgan import DCGANGenerator, DCGANDiscriminator
from torch_gans.trainer.base import BaseGANModule

class DCGANModule(BaseGANModule):
    def __init__(
        self,
        # Using specific types for clarity and future Hydra instantiation
        generator: DCGANGenerator,
        discriminator: DCGANDiscriminator,
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        b1_g: float = 0.5,
        b2_g: float = 0.999,
        b1_d: float = 0.5,
        b2_d: float = 0.999,
    ):
        # latent_dim will be inferred from generator.latent_dim by BaseGANModule's __init__
        # if generator has latent_dim attribute.
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            lr_g=lr_g,
            lr_d=lr_d,
            b1_g=b1_g,
            b2_g=b2_g,
            b1_d=b1_d,
            b2_d=b2_d
            # latent_dim is derived by BaseGANModule from generator
        )
        # self.save_hyperparameters(ignore=['generator', 'discriminator']) # Already called in BaseGANModule

    def generator_loss(self, generated_imgs: torch.Tensor, disc_output_on_generated: torch.Tensor) -> torch.Tensor:
        '''
        Computes the generator loss for DCGAN.
        The generator tries to make the discriminator classify generated images as real.
        '''
        target_is_real = torch.ones_like(disc_output_on_generated, device=self.device)
        g_loss = F.binary_cross_entropy_with_logits(disc_output_on_generated, target_is_real)
        return g_loss

    def discriminator_loss(
        self,
        real_imgs: torch.Tensor,
        generated_imgs: torch.Tensor, # Unused in this standard DCGAN loss, but part of base signature
        disc_output_on_real: torch.Tensor,
        disc_output_on_generated: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Computes the discriminator loss for DCGAN.
        The discriminator tries to correctly classify real images as real and generated images as fake.
        '''
        target_is_real = torch.ones_like(disc_output_on_real, device=self.device)
        target_is_fake = torch.zeros_like(disc_output_on_generated, device=self.device)

        loss_real = F.binary_cross_entropy_with_logits(disc_output_on_real, target_is_real)
        loss_fake = F.binary_cross_entropy_with_logits(disc_output_on_generated, target_is_fake)

        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    @torch.no_grad()
    def log_images(self, batch_size: int = 8, log_name: str = "generated_images_dcgan"):
        '''Logs a batch of generated images to the logger.'''
        # Ensure this runs on the correct device and only on global rank 0
        if hasattr(self.trainer, 'is_global_zero') and not self.trainer.is_global_zero:
            return
        # Fallback if is_global_zero is not available (e.g. not in a PL Trainer context)
        # This might happen if called directly, though less common for this hook.
        if not hasattr(self.trainer, 'is_global_zero') and self.global_rank != 0 :
             return


        z = self.sample_noise(batch_size) # sample_noise is on BaseGANModule, uses self.device
        generated_imgs = self(z) # self.forward(z) -> self.generator(z)

        if self.logger and self.logger.experiment:
            # Normalize images to [0,1] for visualization (common for Tanh output in [-1,1])
            generated_imgs_vis = (generated_imgs + 1) / 2.0

            if hasattr(self.logger.experiment, 'add_images'): # TensorBoard
                self.logger.experiment.add_images(log_name, generated_imgs_vis, self.current_epoch)
            elif hasattr(self.logger.experiment, 'log_image'): # MLFlow, TestTube, etc.
                # MLFlow's log_image expects a file path or BytesIO, not a tensor directly.
                # For simplicity, skipping detailed MLFlow handling here.
                # This part would need torchvision.utils.save_image or similar to save to a buffer.
                # Also, log_image typically logs one image, not a batch.
                # Placeholder for more complex logger integration:
                print(f"Rank {self.global_rank}: Logger {type(self.logger.experiment)} may require specific image logging format.")
            # A more generic way for image logging if add_images is not present but add_image (singular) is
            elif hasattr(self.logger.experiment, 'add_image') and not hasattr(self.logger.experiment, 'add_images'):
                 # Log first image of the batch as an example
                 self.logger.experiment.add_image(f"{log_name}_sample", generated_imgs_vis[0], self.current_epoch)

        else:
            if self.global_rank == 0:
                 print(f"Rank {self.global_rank}: Logger not available or compatible for logging images.")


    def on_train_epoch_end(self) -> None:
        # Call BaseGANModule's on_train_epoch_end if it exists and does something
        if hasattr(super(), 'on_train_epoch_end'):
             super().on_train_epoch_end()
        self.log_images(batch_size=16)
