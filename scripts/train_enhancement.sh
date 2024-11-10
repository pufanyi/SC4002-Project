# export WANDB_API_KEY=<YOUR_WANDB_KEY>
# export WANDB_PROJECT=<YOUR_PROJECT>
# export WANDB_ENTITY=<YOUR_USER_NAME>
# export WANDB_MODE=online

RUN_NAME=train_rnn

python3 -m sc4002.train.train \
    --output_dir ./checkpoints \
    --report_to wandb \
    --model_type bilstm \
    --eval_strategy steps \
    --eval_steps 100 \
    --logging_steps 10 \
    --label_names "labels" \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --freeze_word_embed \
    --run_name $RUN_NAME
    # --lr_scheduler_type "cosine" \
    # --weight_decay 0. \
    # --warmup_ratio 0.03