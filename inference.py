import os
from huggingface_hub import login
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


login(os.environ.get("HF_API_TOKEN"))
model_id = "fireworks-ai/FireLLaVA-13b"

prompt = "USER: <image>\nWhat is this?\n\nASSISTANT:"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve\
/main/transformers/tasks/car.jpg"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

raw_image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))
