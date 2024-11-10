import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import HfArgumentParser, TrainingArguments

from sc4002.eval import compute_acc
from sc4002.models import RNN
from sc4002.train.config import CustomTrainingArguments, DataArguments, ModelArguments
from sc4002.train.trainer import CustomTrainer
from sc4002.train.utils import DataCollator, get_model, preprocess_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer_path = hf_hub_download(repo_id=model_args.download_repo, filename=model_args.tokenizer_path)
    checkpoint_path = hf_hub_download(repo_id=model_args.download_repo, filename=model_args.word_embed_path)

    model = get_model(model_args, tokenizer_path, checkpoint_path)
    tokenizer = model.word_embedding.tokenizer

    dataset = load_dataset(data_args.dataset_name)

    if model_args.freeze_word_embed:
        for p in model.word_embedding.parameters():
            p.requires_grad = False

    train_dataset, eval_dataset, test_dataset = preprocess_dataset(
        dataset,
        tokenizer,
        data_args.train_split,
        data_args.val_split,
        data_args.test_split,
    )
    collator = DataCollator(tokenizer=tokenizer)
    trainer = CustomTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_acc,
    )

    trainer.train()
    test_output = trainer.predict(test_dataset)


if __name__ == "__main__":
    main()
