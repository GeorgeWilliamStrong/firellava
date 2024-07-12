accelerate launch \
    --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 8 \
    vsft_llava.py \
        --dataset_name="/home/ubitus04/dataset/hf" \
        --model_name_or_path="fireworks-ai/FireLLaVA-13b" \
        --report_to="wandb" \
        --learning_rate=1.4e-5 \
        --per_device_train_batch_size=8 \
        --gradient_accumulation_steps=1 \
        --output_dir="/home/ubitus04/run1/vsft-firellava-13b" \
        --logging_steps=5 \
        --num_train_epochs=1 \
        --gradient_checkpointing \
        --remove_unused_columns=False \
        --torch_dtype=float16 \
        --fp16=True
