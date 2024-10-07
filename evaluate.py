import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "fireworks-ai/FireLLaVA-13b"
model_path = "/path/to/model-checkpoint"
test_path = "/path/to/test-dataset.xlsx"
images_path = "/path/to/images-folder"

model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(0)
processor = AutoProcessor.from_pretrained(model_id)

model.eval()

test_df = pd.read_excel(test_path)
print('Num test samples:', test_df.shape[0])

results = []
counter = 1
for _, row in test_df.iterrows():

    print(f"Sample {counter}")
    counter += 1

    sample_id = row['id']
    human_prompt = row['human']

    prompt = f"USER: <image>\n{human_prompt}\n\nASSISTANT:"

    image_path = images_path+f"/{sample_id}.jpg"
    raw_image = Image.open(image_path)

    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=600, do_sample=False)
    result = processor.decode(output[0], skip_special_tokens=True)

    results.append({'id': sample_id, 'human': human_prompt, 'result': result})

result_df = pd.DataFrame(results)
result_df.to_excel("test_evaluation.xlsx", index=False)

print("Inference completed and results saved to test_evaluation.xlsx")
