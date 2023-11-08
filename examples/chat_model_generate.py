import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "Nanbeige/Nanbeige-16B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.eval()

output, messages = model.chat(tokenizer, "如何有效地提高网站流量？")
print(output)

output, messages = model.chat(tokenizer, "你可以给我一些具体的SEO优化技巧吗？", messages=messages)
print(output)
