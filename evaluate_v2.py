import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed

model_id = "fireworks-ai/FireLLaVA-13b"
model_path = "/home/ubitus09/runs/full_finetune"
test_path = "/home/ubitus09/data/test.xlsx"
images_path = "/home/ubitus09/data/dataset/images"

# Load model and processor
model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(0)
processor = AutoProcessor.from_pretrained(model_id)

model.eval()

# Load test data
test_df = pd.read_excel(test_path)
print('Num test samples:', test_df.shape[0])


# Define the function for processing a single row
def process_row(row):
    sample_id = row['id']
    human_prompt = row['human']

    prompt = f"USER: <image>\n{human_prompt}\n\nASSISTANT:"
    image_path = images_path + f"/{sample_id}.jpg"

    try:
        raw_image = Image.open(image_path)
        inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
        output = model.generate(**inputs, max_new_tokens=600, do_sample=False)
        result = processor.decode(output[0], skip_special_tokens=True)
        return {'id': sample_id, 'human': human_prompt, 'result': result}
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        return None


# Use ThreadPoolExecutor to parallelize over rows, preserving order
results = [None] * len(test_df)  # Preallocate a list to hold results in the correct order

with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust the number of workers as needed
    future_to_index = {executor.submit(process_row, row): i for i, row in test_df.iterrows()}

    for future in as_completed(future_to_index):
        index = future_to_index[future]
        result = future.result()
        if result:
            results[index] = result

# Filter out None values (in case of errors)
results = [res for res in results if res]

# Save results to Excel
result_df = pd.DataFrame(results)
result_df.to_excel("test_evaluation.xlsx", index=False)

print("Inference completed and results saved to test_evaluation.xlsx")
