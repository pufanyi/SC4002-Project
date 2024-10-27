# SC4002 Project

## Installation

```bash
# Create a new conda environment
conda create --name sc4002 python=3.11
conda activate sc4002

# Install the package
python -m pip install -e .
```

## Training

Before training, make sure to activate the `wandb`.

```bash
python -m pip install wandb

export WANDB_API_KEY=<YOUR_WANDB_KEY>
export WANDB_PROJECT=<YOUR_PROJECT>
export WANDB_ENTITY=<YOUR_USER_NAME>
export WANDB_MODE=online
```

```bash
sh scripts/train.sh
```

## Test

```bash
python test/run_suite.py
```
