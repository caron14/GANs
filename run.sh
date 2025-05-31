#!/bin/zsh
# Execution Bash script
# Activate Python virtual environment if needed (uncomment below)
source .venv/bin/activate

# Set the model type, you can change this to "ngan", "dcgan", "cgan", etc.
model_type="cgan"

# Run the main script
python main.py --model_type model_type

# If you want to run another entry point, use below
# python torch_gans/train.py
