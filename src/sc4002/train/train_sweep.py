import json

import wandb
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import HfArgumentParser, TrainingArguments

import wandb
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
    wandb.init(project=training_args.wandb_project)
    with open(training_args.sweep_config, "r") as f:
        sweep_config = json.load(f)
    sweep_id = wandb.sweep(sweep_config)

    def train_sweep(config=None):
        with wandb.init(config=config):
            config = wandb.config
            tokenizer_path = hf_hub_download(repo_id=model_args.download_repo, filename=model_args.tokenizer_path)
            checkpoint_path = hf_hub_download(repo_id=model_args.download_repo, filename=model_args.word_embed_path)
            num_layers = config.num_layers

            model = get_model(model_args, tokenizer_path, checkpoint_path, num_layers=num_layers)
            tokenizer = model.word_embedding.tokenizer

            if model_args.freeze_word_embed:
                for p in model.word_embedding.parameters():
                    p.requires_grad = False

            dataset = load_dataset(data_args.dataset_name)
            train_dataset, eval_dataset, test_dataset = preprocess_dataset(
                dataset,
                tokenizer,
                data_args.train_split,
                data_args.val_split,
                data_args.test_split,
            )

            inner_training_args = TrainingArguments(
                output_dir="checkpoints-sweeps",
                report_to="wandb",  # Turn on Weights & Biases logging
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=16,
                save_strategy="epoch",
                evaluation_strategy="epoch",
                logging_strategy="epoch",
                load_best_model_at_end=True,
                remove_unused_columns=False,
                fp16=True,
                metric_for_best_model="accuracy",
                label_names=["labels"],
            )

            collator = DataCollator(tokenizer=tokenizer)
            trainer = CustomTrainer(
                model,
                args=inner_training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
                compute_metrics=compute_acc,
            )

            trainer.train()
            trainer.predict(test_dataset)

    wandb.agent(sweep_id, train_sweep, count=training_args.sweep_count)


if __name__ == "__main__":
    main()
