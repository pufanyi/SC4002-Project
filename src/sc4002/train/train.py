from dataclasses import dataclass, field

from transformers import HfArgumentParser, Trainer, TrainingArguments


@dataclass
class ModelArguments:
    model_type: str = field(default=None)


@dataclass
class DataArguments:
    dataset_name: str = field(default="rotten_tomatoes")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))


if __name__ == "__main__":
    main()
