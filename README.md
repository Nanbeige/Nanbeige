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
        <b>中文</b> |
        <a href="https://github.com/Nanbeige/Nanbeige/blob/main/README_EN.md">English</a>
    <p>
</h4>


# <span id="Introduction">模型介绍</span>

Nanbeige-16B（南北阁-16B）是南北阁大模型实验室研发的160亿参数规模的大语言模型，采用了2.5T Tokens进行预训练，数据包含大量互联网高质量语料、各类书籍、代码等领域脱敏文本，在各个权威测评数据集上都取得了不错的效果。本次发布包含有 Base、Chat 以及扩展上下文长度的 Base-32k、Chat-32k 版本。

Base-32k 版本基于Nanbeige-16B-Base模型，采用YaRN插值方法对位置编码进行修改，并以32k为最大长度进行了20B Tokens的中文、英文、代码语料的全参数增量预训练。

Chat 版本和 Chat-32k 版本分别基于Nanbeige-16B-Base模型和Nanbeige-16B-Base-32k模型，经过了大量人类对齐训练，能够更好、更安全地回复用户的问题。

# <span id="Evaluation">性能测评</span>

我们选取了C-Eval、CMMLU、MMLU、GSM8K、HumanEval、BBH、MBPP等数据集，对 Base 模型的中英文知识、数学、逻辑推理、代码等能力进行全面测评，在同级别开源模型中，取得了相对不错的表现。

###
| 模型                | C-Eval | CMMLU | MMLU  | GSM8K | HumanEval | BBH   | MBPP  |
|--------------------|--------|-------|-------|-------|-----------|-------|-------|
| LLaMA2-13B         | 35.80  | 38.40 | 54.80 | 29.60 | 20.10     | 45.62 | 26.80 |
| Baichuan2-13B-Base | 58.10  | 61.30 | 59.17 | 52.77 | 17.07     | 48.98 | 30.80 |
| Qwen-14B           | 72.10  | 70.2  | 66.30 | 61.30 | 32.30     | 53.40 | 39.80 |
| InternLM-20B       | 58.80  | 59    | 62.05 | 52.62 | 25.61     | 52.51 | 35.60 |
| XVERSE-13B         | 53.70  | 59.1  | 55.21 | 18.20 | 15.85     | 38.06 | -     |
| Skywork-13B        | 60.60  | 61.8  | 62.10 | 55.80 | -         | -     | -     |
| Nanbeige-16B-Base  | 63.80  | 66.58 | 64.80 | 57.62 | 24.56     | 50.68 | 36.40 |


### C-Eval各项分数
|                   | 平均  | 平均（Hard） | STEM | 社会科学 | 人文科学 | 其他   |
|-------------------|------|----------|------|------|------|------|
| Chinese-LLaMA-13B | 33.3 | 27.3     | 31.6 | 37.2 | 33.6 | 32.8 |
| Baichuan-13B      | 53.6 | 36.7     | 47.0 | 66.8 | 57.3 | 49.8 |
| Qwen-14B          | 72.1 | 53.7     | 65.7 | 85.4 | 75.3 | 68.4 |
| XVERSE-13B        | 54.7 | 33.5     | 45.6 | 66.2 | 58.3 | 56.9 |
| Skywork-13B       | 60.6 | 39.4     | 51.2 | 74.6 | 67.8 | 57.5 |
| Nanbeige-16B-Base | 63.8 | 43.5     | 57.8 | 77.2 | 66.9 | 59.4 |


### 长文本理解 Base
我们使用LongBench的LSHT、LCC、Multifiled_QA_ZH数据集，对 Nanbeige-16B-Base-32k 模型进行了测评，相较具有长文本理解能力的同参数规模Base模型取得了不错的效果。

|                            |  LSHT (分类)  |  LCC (代码) | Multifiled_QA_ZH (问答) |
|----------------------------|--------|-------|------------------|
| Chinese-Llama2-13B-16k     |  31.0  |  9.6  |       36.4       |
| Qwen-14B-Dynamnic_ntk-Logn |  16.0  |  66.7 |       30.0       | 
| Nanbeige-16B-Base-32k      |  40.3  |  70.7 |       47.2       |

### 长文本理解 Chat
我们使用LongBench的全部数据集对 Nanbeige-16B-Chat-32k 模型进行了测评，相较具有长文本理解能力的其他开源Chat模型取得了不错的效果。

|                          | Average | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code |
|--------------------------|---------|---------------|--------------|---------------|----------|-----------|------|
| BlueLM-7B-Chat-32K       |  41.2   |     35.6      |     36.2     |     18.8      |   56.9   |   47.6    | 52.8 |
| Chatglm2-6B-32k          |  41.5   |     37.6      |     34.6     |     24.7      |   51.3   |   47.6    | 54.2 |
| Chatglm3-6B-32k          |  50.2   |     45.8      |     46.1     |     26.6      |   61.2   |   65.0    | 56.2 |
| Chinese-Alpaca-2-13B-16K |  29.7   |     47.9      |     26.7     |     13.0      |   22.3   |   21.5    | 46.6 |
| Ziya-Reader-13B-v1.0     |    \    |       \       |     42.8     |     15.3      |     \    |   66.0    |   \  |
| Nanbeige-16B-Chat-32k    |  52.3   |     48.9      |     41.1     |     26.3      |   63.3   |   66.8    | 67.5 |


### LLMEval-3
**LLMEval-3** ( [Github](https://github.com/llmeval/llmeval-3) / [主页](http://llmeval.com/index) ) 聚焦于专业知识能力评测，涵盖哲学、经济学、法学、教育学、文学、历史学、理学、工学、农学、医学、军事学、管理学、艺术学等教育部划定的13个学科门类、50余个二级学科，共计约20W道标准生成式问答题目。防止作弊是LLMEval-3考虑的重要因素。现有公开评测基准存在测试题库泄露的问题，因此可能出现“刷榜”、“刷分”等不公平现象，在LLMEval-3中，每个参与评测的系统需要完成从总题库中随机抽样的1000题，针对同一机构的模型，确保每次评测题目不重复。

我们基于 LLMEval-3 对 Nanbeige-16B-Chat 模型进行了全面测评，测评结果如下：

| 模型名称                       | 相对分数-GPT4 | 相对分数-GPT3.5 | 绝对分数  | 工学   | 经济学  | 教育学  | 法学   | 文学   | 管理学  | 理学   | 历史学  | 医学   | 军事学  |
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



# <span id="Inference">推理</span>

## 相关依赖

- python 3.8及以上版本
  
- transformers 4.33.3
  
- pytorch 2.0及以上版本
  
可以通过以下pip命令安装相关依赖库

```
pip install transformers==4.33.3 transformers_stream_generator deepspeed einops==0.3.2 datasets==2.10.0
```

## 推理代码

通过以下代码可以调用Base模型进行续写生成：

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

# <span id="Demo">聊天Demo</span>

### CLI Demo
我们在cli_demo.py中提供了一个CLI演示示例，它支持生成的流输出。用户可以通过输入提示与Nanbeige-16B-Chat进行交互，模型以流方式返回模型输出。运行如下命令:

```
python cli_demo.py --cpu # CPU上加载模型推理
```

# <span id="Finetuning">微调训练</span>
我们提供了训练脚本的样例，该脚本采用DeepSpeed训练，用户需要确保训练环境已经安装好DeepSpeed，并根据集群情况，修改 config/hostfile ，如果是多机多卡，需要修改 ssh 中各个节点的 IP 配置。具体可以参见 DeepSpeed 官方说明，执行 scripts/train.sh 脚本即可开始训练:

```
scripts/train.sh
```

训练数据文件格式为jsonl格式，即每一行一条JSON格式的样本，每个样本都是一个字典，包含一个id和一个对话列表，我们的训练脚本默认读取data/目录下的train.jsonl和eval.jsonl文件。下面是一条简单的样本示例:

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

# <span id="Limitations">局限性</span>

虽然我们在训练过程中非常注重模型的安全性，力求确保其输出符合伦理和法律要求的文本，但由于模型大小和概率生成范式的限制，无法完全避免产生各种不符合预期的输出情况。这些输出可能包含偏见、歧视等有害内容，请勿传播这些内容。我们不承担因传播不良信息而导致的任何后果。

# <span id="License">协议</span>

使用 Nanbeige 模型时，您必须遵守 Apache 2.0 许可证和[《南北阁大语言模型许可协议》](https://github.com/Nanbeige/Nanbeige/blob/main/%E5%8D%97%E5%8C%97%E9%98%81%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf)。如果您打算将 Nanbeige 模型或其衍生产品用于商业目的，请通过以下邮箱 nanbeige@126.com 提交申请材料，以满足《南北阁大语言模型许可协议》的要求。经过审核后，我们将授予您非排他性、全球范围内、不可转让、不可再许可、可撤销的商业版权许可。

## 引用
如果您觉得我们的工作对您有帮助，欢迎引用我们的项目：
```
@misc{NanBeiGe,
  title = {NanBeiGe LLM},
  author = {NanBeiGe LLM Lab},
  howpublished = {\url{https://github.com/Nanbeige/Nanbeige)},
  year = {2023},
  month = {Nov},
}
```
