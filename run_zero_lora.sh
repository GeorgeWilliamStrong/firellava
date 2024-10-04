accelerate launch \
    --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 8 \
    vsft_llava.py \
        --dataset_name="/path/to/hf-dataset" \
        --model_name_or_path="/path/to/FireLLaVA-13b" \
        --report_to="wandb" \
        --learning_rate=1e-4 \
        --per_device_train_batch_size=8 \
        --gradient_accumulation_steps=1 \
        --output_dir="/path/to/outputs" \
        --logging_steps=5 \
        --num_train_epochs=5 \
        --gradient_checkpointing \
        --remove_unused_columns=False \
        --torch_dtype=float16 \
        --fp16=True \
        --use_peft \
        --lora_target_modules=all-linear