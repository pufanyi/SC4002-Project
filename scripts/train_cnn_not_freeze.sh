# export WANDB_API_KEY=<YOUR_WANDB_KEY>
# export WANDB_PROJECT=<YOUR_PROJECT>
# export WANDB_ENTITY=<YOUR_USER_NAME>
# export WANDB_MODE=online

RUN_NAME=train_rnn

python3 -m sc4002.train.train \
    --output_dir ./checkpoints \
    --report_to wandb \
    --model_type rnn \
    --eval_strategy steps \
    --eval_steps 100 \
    --logging_steps 10 \
    --label_names "labels" \
    --num_train_epochs 10 \
    --run_name $RUN_NAME \
    --lr_scheduler_type "cosine" \
    --wandb_project "sc4002_rnn_freeze_danchou" \
    --warmup_ratio 0.03 \
    --learning_rate 1.63e-5