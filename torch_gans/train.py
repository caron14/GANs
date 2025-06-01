from pytorch_lightning.cli import LightningCLI
import torch # For torch.set_float32_matmul_precision

# Import modules that LightningCLI needs to know about.
# These imports ensure that the classes are registered with Python's type system
# so Hydra can find them by their string paths specified in YAML configs.
from torch_gans.models.dcgan import DCGANGenerator, DCGANDiscriminator # For model parts
from torch_gans.trainer.dcgan_module import DCGANModule # Main GAN system
from torch_gans.data.mnist_datamodule import MNISTDataModule # Data handling
# Base classes are not directly instantiated by CLI from top-level config, but good to have them recognized.
from torch_gans.models.base import BaseGenerator, BaseDiscriminator
from torch_gans.trainer.base import BaseGANModule


# Custom CLI subclass to potentially add more arguments or modify behavior later.
# For now, it's a direct use of LightningCLI.
class GANsCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Example: Add custom arguments not managed by Hydra/Lightning
        # parser.add_argument("--my_custom_arg", type=str, default="default_value")
        # parser.link_arguments("data.seed", "seed_everything") # Already handled by config
        pass

    def before_instantiate_classes(self) -> None:
        # Called before any objects are instantiated.
        # Useful for things like setting global flags.
        # Example: Set precision for matmul operations (relevant for some GPUs/PyTorch versions)
        torch.set_float32_matmul_precision('medium') # or 'high'

        # The config is available as self.config
        # If seed_everything is in the config, LightningCLI handles it.
        # We ensured 'seed_everything' is in config.yaml linked to data.seed.
        pass

    # You can override other methods like after_fit, etc.


def main():
    # LightningCLI takes the LightningModule class, LightningDataModule class,
    # and trainer defaults.
    # It will use Hydra to parse YAML configurations and command-line overrides.
    # The actual classes instantiated for model and data will be determined by
    # the '_target_' in the resolved Hydra config (e.g., from config.yaml).
    #
    # We provide DCGANModule and MNISTDataModule as defaults if nothing is specified,
    # but the config.yaml already specifies them.
    # The CLI will look for 'gan_module' and 'data' groups in the config.
    # The 'trainer' group is automatically picked up for Trainer settings.

    # Note: LightningCLI expects `model_class` and `datamodule_class` arguments.
    # It will use the `gan_module` key from `config.yaml` for the model_class instance,
    # and the `data` key for the datamodule_class instance.
    cli = GANsCLI(
        model_class=DCGANModule,  # Default if not in config, but config.yaml's gan_module._target_ takes precedence.
        datamodule_class=MNISTDataModule, # Default if not in config, config.yaml's data._target_ takes precedence.
        save_config_overwrite=True, # Save the resolved Hydra config to the log dir.
        # run=True, # Automatically calls trainer.fit(model, datamodule) if True. Default is True.
        # seed_everything_default=None, # Let config handle it via 'seed_everything' key
        subclass_mode_model=True, # Allows model_class to be a base class, with _target_ resolving to subclass
        subclass_mode_datamodule=True, # Same for datamodule_class
        # Assuming train.py is in torch_gans/ and configs/ is a subdirectory within torch_gans.
        # This tells LightningCLI to look for 'config.yaml' in the 'configs' directory
        # relative to the directory of train.py (which is torch_gans/).
        config_path_override="configs/config.yaml",
        # trainer_defaults={'logger': True} # Example: ensure logger is enabled
    )
    # cli.trainer.fit(cli.model, cli.datamodule) # This is called by cli.run() if run=True (default)
    # No need to call fit manually unless run=False

if __name__ == "__main__":
    main()
