import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

model_id = "fireworks-ai/FireLLaVA-13b"
model_path = "/home/ubitus09/runs/full_finetune"
test_path = "/home/ubitus09/data/test.xlsx"
images_path = "/home/ubitus09/data/dataset/images"

model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(0)
processor = AutoProcessor.from_pretrained(model_id)
model.eval()

test_df = pd.read_excel(test_path)
print('Num test samples:', test_df.shape[0])

results = []


def load_image(image_path):
    return Image.open(image_path)


def process_batch(batch_rows):
    batch_results = []
    image_paths = [images_path + f"/{row['id']}.jpg" for _, row in batch_rows]

    with ThreadPoolExecutor() as executor:
        raw_images = list(executor.map(load_image, image_paths))

    prompts = [f"USER: <image>\n{row['human']}\n\nASSISTANT:" for _, row in batch_rows]
    inputs = processor(prompts, raw_images, return_tensors='pt', padding=True).to(0, torch.float16)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=600, do_sample=False)

    decoded_results = processor.batch_decode(output, skip_special_tokens=True)

    for i, (_, row) in enumerate(batch_rows):
        batch_results.append({'id': row['id'], 'human': row['human'], 'result': decoded_results[i]})

    return batch_results


batch_size = 4
for i in range(0, len(test_df), batch_size):
    batch_rows = test_df.iloc[i:i + batch_size]
    results.extend(process_batch(batch_rows))

result_df = pd.DataFrame(results)
result_df.to_excel("test_evaluation.xlsx", index=False)

print("Inference completed and results saved to test_evaluation.xlsx")
