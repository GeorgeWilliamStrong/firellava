accelerate launch \
    --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 6 \
    vsft_llava.py \
        --dataset_name="/home/ubitus07/setup2/dataset/hf" \
        --model_name_or_path="/home/ubitus07/FireLLaVA-13b" \
        --report_to="wandb" \
        --learning_rate=1e-7 \
        --per_device_train_batch_size=8 \
        --gradient_accumulation_steps=1 \
        --output_dir="/home/ubitus07/experiments/test3/vsft-firellava-13b" \
        --logging_steps=5 \
        --num_train_epochs=1 \
        --gradient_checkpointing \
        --remove_unused_columns=False \
        --torch_dtype=float16 \
        --fp16=True
