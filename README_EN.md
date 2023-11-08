<!-- markdownlint-disable first-line-h1 -->

<!-- markdownlint-disable html -->

<div align="center">
<h1>
  Nanbeige-16B
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/Nanbeige/" target="_blank">Hugging Face</a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/Nanbeige/Nanbeige/blob/main/README.md">中文</a> |
        <b>English</b>
    <p>
</h4>


# <span id="Introduction">Model Introduction</span>

Nanbeige-16B is a 16 billion parameter language model developed by Nanbeige LLM Lab. It uses 2.5T Tokens for pre-training. The training data includes a large amount of high-quality internet corpus, various books, code, etc. It has achieved good results on various authoritative evaluation data sets. This release includes the Base, Chat, Base-32k and Chat-32k.

The Base-32k version is based on the Nanbeige-16B-Base model, which uses YaRN interpolation method to modify the position encoding, and performs full parameter incremental pre-training with 20 billion tokens of Chinese, English, and code corpora, under 32k max length.

The Chat version and Chat-32k version are based on the Nanbeige-16B-Base model and Nanbeige-16B-Base-32k model respectively. They have undergone extensive human-aligned training.

# <span id="Evaluation">Performance Evaluation</span>

We selected datasets such as C-Eval, CMMLU, MMLU, GSM8K, HumanEval, BBH, MBPP, to evaluate the capabilities of the Base model. Among open-source models of similar scale, it achieved relatively good performance.

###
| Model              | C-Eval | CMMLU | MMLU  | GSM8K | HumanEval | BBH   | MBPP  |
|--------------------|--------|-------|-------|-------|-----------|-------|-------|
| LLaMA2-13B         | 35.80  | 38.40 | 54.80 | 29.60 | 20.10     | 45.62 | 26.80 |
| Baichuan2-13B-Base | 58.10  | 61.30 | 59.17 | 52.77 | 17.07     | 48.98 | 30.80 |
| Qwen-14B           | 72.10  | 70.2  | 66.30 | 61.30 | 32.30     | 53.40 | 39.80 |
| InternLM-20B       | 58.80  | 59    | 62.05 | 52.62 | 25.61     | 52.51 | 35.60 |
| XVERSE-13B         | 53.70  | 59.1  | 55.21 | 18.20 | 15.85     | 38.06 | -     |
| Skywork-13B        | 60.60  | 61.8  | 62.10 | 55.80 | -         | -     | -     |
| Nanbeige-16B-Base  | 63.80  | 66.58 | 64.80 | 57.62 | 24.56     | 50.68 | 36.40 |


### C-Eval Scores
|                   | Avg   | Avg（Hard） | STEM | Social Sciences | Humanities | Other  |
|-------------------|------|----------|------|------|------|------|
| Chinese-LLaMA-13B | 33.3 | 27.3     | 31.6 | 37.2 | 33.6 | 32.8 |
| Baichuan-13B      | 53.6 | 36.7     | 47.0 | 66.8 | 57.3 | 49.8 |
| Qwen-14B          | 72.1 | 53.7     | 65.7 | 85.4 | 75.3 | 68.4 |
| XVERSE-13B        | 54.7 | 33.5     | 45.6 | 66.2 | 58.3 | 56.9 |
| Skywork-13B       | 60.6 | 39.4     | 51.2 | 74.6 | 67.8 | 57.5 |
| Nanbeige-16B-Base | 63.8 | 43.5     | 57.8 | 77.2 | 66.9 | 59.4 |

### Long Context Understanding Base Model
We evaluated the Nanbeige-16B-Base-32k model using LSHT, LCC, and Multifiled_QA_ZH from LongBench datasets. Compared to the Base model of the same parameter size with long-context comprehension capabilities, it achieved impressive results.

|                            |  LSHT  |  LCC  | Multifiled_QA_ZH |
|----------------------------|--------|-------|------------------|
| Chinese-Llama2-13B-16k     |  31.0  |  9.6  |       36.4       |
| Qwen-14B-Dynamnic_ntk-Logn |  16.0  |  66.7 |       30.0       | 
| Nanbeige-16B-Base-32k      |  40.3  |  70.7 |       47.2       |

### Long Context Understanding Chat Model
We evaluated the Nanbeige-16B-Chat-32k model using LongBench datasets. Compared to the Chat model of the same parameter size with long-context comprehension capabilities, it achieved impressive results.

|                          | Average | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code |
|--------------------------|---------|---------------|--------------|---------------|----------|-----------|------|
| BlueLM-7B-Chat-32K       |  41.2   |     35.6      |     36.2     |     18.8      |   56.9   |   47.6    | 52.8 |
| Chatglm2-6B-32k          |  41.5   |     37.6      |     34.6     |     24.7      |   51.3   |   47.6    | 54.2 |
| Chatglm3-6B-32k          |  50.2   |     45.8      |     46.1     |     26.6      |   61.2   |   65.0    | 56.2 |
| Chinese-Alpaca-2-13B-16K |  29.7   |     47.9      |     26.7     |     13.0      |   22.3   |   21.5    | 46.6 |
| Ziya-Reader-13B-v1.0     |    \    |       \       |     42.8     |     15.3      |     \    |   66.0    |   \  |
| Nanbeige-16B-Chat-32k    |  52.3   |     48.9      |     41.1     |     26.3      |   63.3   |   66.8    | 67.5 |


### LLMEval-3
We conducted a comprehensive evaluation of Nanbeige-16B-Chat model based on **LLMEval-3** ( [Github](https://github.com/llmeval/llmeval-3) / [Homepage](http://llmeval.com/index) ), and the results are as follows:

| 模型名称    | 相对分数-GPT4 | 相对分数-GPT3.5 | 绝对分数  | 工学   | 经济学  | 教育学  | 法学   | 文学   | 管理学  | 理学   | 历史学  | 医学   | 军事学  |
|----------------------------|-----------|-------------|-------|------|------|------|------|------|------|------|------|------|------|
| Baidu3.5                   | 104.21    | 121.39      | 77.53 | 8.13 | 8.00 | 8.63 | 7.97 | 6.23 | 7.63 | 7.33 | 8.77 | 7.47 | 7.37 |
| ChatGLM-pro                | 103.45    | 120.51      | 76.97 | 6.97 | 8.47 | 7.97 | 8.93 | 7.23 | 7.70 | 6.33 | 8.37 | 7.13 | 7.87 |
| GPT-4                      | 100.00    | 116.49      | 74.40 | 7.23 | 7.80 | 7.73 | 8.40 | 6.73 | 7.67 | 7.73 | 7.07 | 6.20 | 7.83 |
| **Nanbeige-16B-Chat**               | 94.26     | 109.80      | 70.13 | 6.00 | 7.87 | 8.20 | 8.33 | 6.07 | 6.83 | 6.00 | 7.80 | 5.80 | 7.23 |
| minimax-abab5              | 93.28     | 108.66      | 69.40 | 5.83 | 7.50 | 7.77 | 8.37 | 6.40 | 6.33 | 5.07 | 8.33 | 5.93 | 7.87 |
| Baichuan2-13B-Chat         | 92.91     | 108.23      | 69.13 | 6.00 | 7.53 | 8.63 | 8.13 | 6.23 | 6.33 | 5.63 | 8.20 | 5.43 | 7.00 |
| Qwen-14B-Chat              | 86.33     | 100.56      | 64.23 | 5.77 | 7.07 | 7.07 | 7.37 | 5.70 | 6.20 | 5.93 | 6.97 | 5.40 | 6.77 |
| GPT-3.5-turbo              | 85.84     | 100.00      | 63.87 | 6.27 | 6.87 | 7.23 | 7.40 | 5.40 | 6.30 | 6.37 | 6.00 | 5.17 | 6.87 |
| ChatGLM2-6B                | 75.71     | 88.19       | 56.33 | 4.03 | 6.33 | 7.00 | 7.57 | 4.77 | 5.93 | 4.23 | 5.87 | 5.07 | 5.53 |
| ziya_v1.1-13b              | 70.20     | 81.78       | 52.23 | 4.67 | 5.77 | 6.07 | 6.53 | 4.53 | 5.33 | 3.70 | 5.00 | 4.63 | 6.00 |
| BELLE-Llama2-13B-chat-0.4M | 68.82     | 80.16       | 51.20 | 4.47 | 5.93 | 6.20 | 6.77 | 4.33 | 4.97 | 4.10 | 5.07 | 3.77 | 5.60 |
| Linly-Chinese-LLaMA2-13B   | 67.82     | 79.00       | 50.46 | 3.87 | 5.80 | 5.83 | 6.57 | 3.93 | 5.37 | 4.07 | 5.43 | 3.93 | 5.67 |
| InternLM-Chat-7B           | 61.06     | 71.13       | 45.43 | 3.83 | 5.13 | 5.27 | 6.57 | 3.90 | 4.83 | 3.10 | 4.87 | 3.67 | 4.27 |
| Llama-2-7b-chat-hf         | 51.11     | 59.54       | 38.03 | 3.33 | 4.77 | 3.77 | 5.03 | 3.07 | 3.77 | 3.93 | 4.00 | 2.40 | 3.97 |



# <span id="Inference">Inference</span>

## Requirements

- python 3.8 and above
  
- transformers 4.33.3
  
- pytorch 2.0 and above

You can install the dependent libraries with the following pip command

```
pip install transformers==4.33.3 transformers_stream_generator deepspeed einops==0.3.2 datasets==2.10.0
```

## Inference Code

The following code can be used to invoke the model for text generation:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("Nanbeige/Nanbeige-16B-Base", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Nanbeige/Nanbeige-16B-Base", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

inputs = tokenizer('中国的首都是北京\n德国的首都是柏林\n孟加拉国的首都是', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# 中国的首都是北京\n德国的首都是柏林\n孟加拉国的首都是达卡\n巴西的首都是巴西利亚\n印度的首都是新德里\n法国的首都是巴黎\n美国的首都是华盛顿\n日本的首都是东京\n俄罗斯的首都是莫斯科\n澳大利亚的首都是堪培拉\n加拿大的首都是渥太华
```


# <span id="Demo">Chat Demo</span>

### CLI Demo
We provide a CLI demo example in cli_demo.py, which supports streaming output for the generation. Users can interact with Nanbeige-16B-Chat by inputting prompts, and the model returns model outputs in the streaming mode. Run the command below:

```
python cli_demo.py --cpu # inference cpu only
```

# <span id="Finetuning">Finetuning</span>
We provide a sample training script that utilizes DeepSpeed for training. Users need to ensure that the training environment has DeepSpeed installed and modify the config/hostfile based on the cluster setup. If using multiple machines and multiple GPUs, the IP configuration in SSH needs to be modified for each node. For more details, please refer to the official DeepSpeed documentation. To start training, execute the scripts/train.sh script.

```
scripts/train.sh
```

The training data file is in JSONL format, with each line representing a JSON-formatted sample. Each sample is a dict that includes an ID and a list of dialogues. Our training script by default reads the train.jsonl and eval.jsonl files located in the data/ directory. Here is an example of a simple sample:

``` json
{
  "id": 312,
  "messages": [
    {"role": "system", "content": "你是一个聪明且博学、乐于助人、恭敬而诚实的助手。"},
    {"role": "human", "content": "太阳表面的温度有多高?"},
    {"role": "assistant", "content": "太阳表面的温度非常高，约为约 5,500°C（9,932°F）左右。这个温度是通过太阳的光谱特征和其他观测数据测得的。太阳的外部大气层称为光球，是我们能够直接观察到的部分，它的温度非常高。然而，太阳的内部温度要更高，中心核心的温度估计约为 15,000,000°C（27,000,000°F）。这些高温是由太阳内部的核聚变反应引起的，核聚变将氢转化为氦并释放出巨大的能量。"},
    {"role": "human", "content": "那金星呢"},
    {"role": "assistant", "content": "金星是太阳系中最接近太阳的行星之一，因此它的表面温度非常高，是整个太阳系中最高的之一。金星的表面温度约为约 462°C（864°F）。这个高温是由于金星的大气层中含有大量的温室气体，如二氧化碳，导致温室效应的加强。金星的大气层中的温室效应使得热量无法逃逸，导致表面温度异常升高。金星的高温和极高的大气压使得金星表面条件极为恶劣，表面覆盖着熔岩平原和火山，没有液态水存在。这使得金星成为一个类似地球的炙热地狱，无法支持生命存在。"},
  ]
}
```

# <span id="Limitations">Limitations</span>
While we place great emphasis on the safety of the model during the training process, striving to ensure that its outputs align with ethical and legal requirements, it may not completely avoid generating unexpected outputs due to the model's size and probabilistic nature. These outputs may include harmful content such as bias or discrimination. Please don't propagate such content. We do not assume any responsibility for the consequences resulting from the dissemination of inappropriate information.

# <span id="License">License</span>
When using the Nanbeige models, you must comply with the Apache 2.0 License and the Nanbeige Models Community License Agreement. If you intend to use the Nanbeige Models or its derivatives for commercial purposes, please submit application materials to meet the requirements of the Nanbeige Models Community License Agreement by contacting nanbeige@126.com. After review, We will grant you a non-exclusive, worldwide, non-transferable, non-sublicensable and revocable commercial copyright license.

## Citation
If you find our work useful, please cite our project:
```
@misc{NanBeiGe,
  title = {NanBeiGe LLM},
  author = {NanBeiGe LLM Lab},
  howpublished = {\url{https://github.com/Nanbeige/Nanbeige)},
  year = {2023},
  month = {Nov},
}
```
