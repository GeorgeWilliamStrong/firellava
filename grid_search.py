import subprocess
import itertools

data_path = "/path/to/hf-dataset"
model_path = "/path/to/FireLLaVA-13b"
output_path = "/path/to/outputs"

# Define the parameter grid
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
gradient_accumulation_steps = [1, 2]
num_train_epochs = [5, 10, 20, 30]

# Path to the accelerate launch script
launch_command = [
    "accelerate", "launch",
    "--config_file=accelerate_configs/deepspeed_zero3.yaml",
    "--num_processes", "8",
    "vsft_llava.py"
]

# Base arguments
base_args = [
    f"--dataset_name={data_path}",
    f"--model_name_or_path={model_path}",
    "--report_to=wandb",
    "--per_device_train_batch_size=8",
    "--logging_steps=5",
    "--gradient_checkpointing",
    "--remove_unused_columns=False",
    "--torch_dtype=float16",
    "--fp16=True",
    "--use_peft",
    "--lora_target_modules=all-linear"
]

# Generate combinations of parameters
param_grid = list(itertools.product(learning_rates, gradient_accumulation_steps, num_train_epochs))

# Loop through each combination of parameters and run the script
for lr, gas, epochs in param_grid:
    # Construct the arguments for this specific run
    specific_args = [
        f"--learning_rate={lr}",
        f"--gradient_accumulation_steps={gas}",
        f"--num_train_epochs={epochs}"
    ]

    # Create a unique output directory for this combination
    output_dir = f"{output_path}/lr{lr}-gas{gas}-epochs{epochs}"
    specific_args.append(f"--output_dir={output_dir}")

    # Combine all arguments
    command = launch_command + base_args + specific_args

    # Run the command
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)
