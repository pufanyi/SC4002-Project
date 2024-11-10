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

To run the parameter search, you can run the search script by

```bash
bash scripts/train_rnn_sweep_not_freeze.sh
```

You can find how to run other model using different scripts in the folder

## Unit Test

We add some of the basic unit testing in the run suite. If you want to run the testing on the test set or the val set, it is being logged at every train and their is no need to run extra scripts.

```bash
python test/run_suite.py
```
