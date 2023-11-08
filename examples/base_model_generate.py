import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

model_path = "Nanbeige/Nanbeige-16B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

inputs = tokenizer('中国的首都是北京\n德国的首都是柏林\n孟加拉的首都是', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# 中国的首都是北京\n德国的首都是柏林\n孟加拉的首都是达卡\n巴西的首都是巴西利亚\n印度的首都是新德里\n法国的首都是巴黎\n美国的首都是华盛顿\n日本的首都是东京\n俄罗斯的首都是莫斯科\n澳大利亚的首都是堪培拉\n加拿大的首都是渥太华