# export WANDB_API_KEY=<YOUR_WANDB_KEY>
# export WANDB_PROJECT=<YOUR_PROJECT>
# export WANDB_ENTITY=<YOUR_USER_NAME>
# export WANDB_MODE=online

RUN_NAME=train_bilstm

python3 -m sc4002.train.train_sweep \
    --output_dir ./checkpoints \
    --report_to wandb \
    --model_type bilstm \
    --eval_strategy steps \
    --eval_steps 100 \
    --logging_steps 10 \
    --label_names "labels" \
    --num_train_epochs 10 \
    --run_name $RUN_NAME \
    --sweep_count 20 \
    --lr_scheduler_type "cosine" \
    --sweep_config ./scripts/config/sweep_config_bilstm.json \
    --wandb_project "sc4002_bilstm"
    # --weight_decay 0. \
    # --warmup_ratio 0.03