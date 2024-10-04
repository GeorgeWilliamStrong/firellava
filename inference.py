from PIL import Image
import requests
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from huggingface_hub import login

login("INSERT_HF_API_TOKEN")
model_id = "fireworks-ai/FireLLaVA-13b"

prompt = "USER: <image>\nPlease describe the contents of this image?\
\n\nASSISTANT:"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve\
/main/transformers/tasks/car.jpg"

model = LlavaForConditionalGeneration.from_pretrained(
    "model_id or path/to/model",
    torch_dtype=torch.float16,
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

raw_image = Image.open(requests.get(url, stream=True).raw)
# raw_image = Image.open("/path/to/image.jpg")

inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=400, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))
