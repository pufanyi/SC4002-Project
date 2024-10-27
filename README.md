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

Before training, make sure to activate the [Weights & Biases](https://wandb.ai/).

```bash
wandb login
```

After that, you can run the training script.

```bash
sh scripts/train.sh
```

## Test

```bash
python test/run_suite.py
```
