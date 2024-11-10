# export WANDB_API_KEY=<YOUR_WANDB_KEY>
# export WANDB_PROJECT=<YOUR_PROJECT>
# export WANDB_ENTITY=<YOUR_USER_NAME>
# export WANDB_MODE=online

RUN_NAME=train_rnn

python3 -m sc4002.train.train_sweep \
    --output_dir ./checkpoints \
    --report_to wandb \
    --model_type rnn \
    --eval_strategy steps \
    --eval_steps 100 \
    --logging_steps 10 \
    --label_names "labels" \
    --num_train_epochs 10 \
    --freeze_word_embed \
    --sweep_config ./scripts/config/sweep_config.json \
    --sweep_count 20 \
    --run_name $RUN_NAME \
    --freeze_word_embed False
    # --lr_scheduler_type "cosine"
    # --weight_decay 0. \
    # --warmup_ratio 0.03