# Copyright (c) 2023 Nanbeige LLM Lab All Rights Reserved.

import argparse
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import SequentialSampler
from torch.utils.data import Dataset, Subset, DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import copy
import os

IGNORE_INDEX = -100
PAD_TOKEN_ID = 0
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SPLIT_TOKEN = "<n>"
LABEL2IDX = {"A": 0,
             "B": 1,
             "C": 2,
             "D": 3}

IDX2LABEL = {0: "A",
             1: "B",
             2: "C",
             3: "D"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument('--model_path',
                        type=str,
                        default=None,
                        help='model path')
    parser.add_argument('--dev_path',
                        type=str,
                        default=None,
                        help='ceval dev set path')
    parser.add_argument('--test_path',
                        type=str,
                        default=None,
                        help='ceval test set path')
    parser.add_argument('--result_path',
                        type=str,
                        default=None,
                        help='ceval output result path')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='batchsize')
    parser.add_argument('--shot_num',
                        type=int,
                        default=0,
                        help='shot num')

    args = parser.parse_args()
    return args

class EvalDataset_(Dataset):
    def __init__(self, tokenizer,
                 max_len,
                 shot_num,
                 dev_path,
                 test_path):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.shot_num = shot_num
        self.dev_path = dev_path
        self.test_path = test_path
        self.dataset = self.process_dataset()

    def __len__(self):
        return len(self.dataset)

    def process_dataset(self):
        TASK2DESC = {
            "high_school_physics": "高中物理",
            "fire_engineer": "注册消防工程师",
            "computer_network": "计算机网络",
            "advanced_mathematics": "高等数学",
            "logic": "逻辑学",
            "middle_school_physics": "初中物理",
            "clinical_medicine": "临床医学",
            "probability_and_statistics": "概率统计",
            "ideological_and_moral_cultivation": "思想道德修养与法律基础",
            "operating_system": "操作系统",
            "middle_school_mathematics": "初中数学",
            "chinese_language_and_literature": "中国语言文学",
            "electrical_engineer": "注册电气工程师",
            "business_administration": "工商管理",
            "high_school_geography": "高中地理",
            "modern_chinese_history": "近代史纲要",
            "legal_professional": "法律职业资格",
            "middle_school_geography": "初中地理",
            "middle_school_chemistry": "初中化学",
            "high_school_biology": "高中生物",
            "high_school_chemistry": "高中化学",
            "physician": "医师资格",
            "high_school_chinese": "高中语文",
            "tax_accountant": "税务师",
            "high_school_history": "高中历史",
            "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论概论",
            "high_school_mathematics": "高中数学",
            "professional_tour_guide": "导游资格",
            "veterinary_medicine": "兽医学",
            "environmental_impact_assessment_engineer": "环境影响评价工程师",
            "basic_medicine": "基础医学",
            "education_science": "教育学",
            "urban_and_rural_planner": "注册城乡规划师",
            "middle_school_biology": "初中生物",
            "plant_protection": "植物保护",
            "middle_school_history": "初中历史",
            "high_school_politics": "高中政治",
            "metrology_engineer": "注册计量师",
            "art_studies": "艺术学",
            "college_economics": "大学经济学",
            "college_chemistry": "大学化学",
            "law": "法学",
            "sports_science": "体育学",
            "civil_servant": "公务员",
            "college_programming": "大学编程",
            "middle_school_politics": "初中政治",
            "teacher_qualification": "教师资格",
            "computer_architecture": "计算机组成",
            "college_physics": "大学物理",
            "discrete_mathematics": "离散数学",
            "marxism": "马克思主义基本原理",
            "accountant": "注册会计师",
        }
        subject_names = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming',
                         'college_physics', 'college_chemistry', 'advanced_mathematics',
                         'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer',
                         'metrology_engineer', 'high_school_mathematics', 'high_school_physics',
                         'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics',
                         'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry',
                         'veterinary_medicine', 'college_economics', 'business_administration', 'marxism',
                         'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics',
                         'high_school_geography', 'middle_school_politics', 'middle_school_geography',
                         'modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law',
                         'chinese_language_and_literature', 'art_studies', 'professional_tour_guide',
                         'legal_professional', 'high_school_chinese', 'high_school_history',
                         'middle_school_history', 'civil_servant', 'sports_science', 'plant_protection',
                         'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant',
                         'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
        processed_dataset = []
        for s in subject_names:
            dev_dataset = load_dataset(self.dev_path, data_files=f"{s}_dev.csv")
            test_dataset = load_dataset(self.test_path, data_files=f"{s}_test.csv")
            devs = dev_dataset['train']
            tests = test_dataset['train']
            dev_context = []
            dev_answer = []
            subject_topic = f"以下是中国关于{TASK2DESC[s]}考试的单项选择题，请选出其中的正确答案。\n\n"
            for dev in devs:
                a = dev['A'].strip()
                b = dev['B'].strip()
                c = dev['C'].strip()
                d = dev['D'].strip()
                question = dev['question'].strip()
                basic_context = question + '\t' + 'A: ' + a + ' B: ' + b + ' C: ' + c + ' D: ' + d
                dev_context.append(basic_context)
                dev_answer.append(dev['answer'])

            for idx, test in enumerate(tests):
                # print(str(idx)+"  "+test['question'])
                choice_label = 'A'
                choice_label = LABEL2IDX[choice_label]
                a = test['A'].strip()
                b = test['B'].strip()
                c = test['C'].strip()
                d = test['D'].strip()
                question = test['question'].strip()
                if self.shot_num != 0:
                    context = question + '\n' + 'A. ' + a + '\nB. ' + b + '\nC. ' + c + '\nD. ' + d
                    context_list = []
                    for idd in range(self.shot_num):
                        c_context_concat = dev_context[idd] + '\n选项ABCD中正确的答案：' + dev_answer[idd]
                        context_list.append(c_context_concat)
                    context = subject_topic + '\n\n'.join(context_list) + '\n\n' + context + '\n选项ABCD中正确的答案：'
                    choices = ["A", "B", "C", "D"]
                    text_sample = (context, choices, choice_label, s, idx)
                    processed_dataset.append(text_sample)
                else:
                    context = subject_topic + question + '\n' + 'A. ' + a + '\nB. ' + b + '\nC. ' + c + '\nD. ' + d + '\n选项ABCD中正确的答案：'
                    choices = ["A", "B", "C", "D"]
                    text_sample = (context, choices, choice_label, s, idx)
                    processed_dataset.append(text_sample)

        return processed_dataset

    def __getitem__(self, idx):
        select_idx = idx
        text_sample = self.dataset[select_idx]
        context, choices, right_index, subject, idd = text_sample
        if idx % 1000 == 0: print(context)
        context_length = self.tokenizer(context, return_tensors="pt")["input_ids"].squeeze(0).size()[-1]
        inputs = []
        labels = []
        attention_masks = []
        for candidate in choices:
            concat_text = context + candidate + EOS_TOKEN
            chosen_tokens = self.tokenizer(concat_text,
                                           max_length=self.max_len,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt")
            input_ids = chosen_tokens["input_ids"].squeeze(0)
            attention_mask = chosen_tokens["attention_mask"].squeeze(0)
            token_labels = copy.deepcopy(input_ids)
            token_labels[:context_length] = IGNORE_INDEX
            token_labels = torch.where(token_labels == PAD_TOKEN_ID, IGNORE_INDEX, token_labels)
            token_labels = torch.where(token_labels == 2, IGNORE_INDEX, token_labels)
            inputs.append(input_ids[:-1])
            labels.append(token_labels[1:])
            attention_masks.append(attention_mask[:-1])
        return {
            "input_ids": torch.stack(inputs),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
            "choice_label": right_index,
            "subject": subject,
            "id": idd
        }


def make_dataloader(dataset, batch_size):
    eval_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            collate_fn=choice_collator_fn,
                            sampler=eval_sampler,
                            batch_size=batch_size,
                            num_workers=5,
                            pin_memory=True,
                            persistent_workers=True,
                            drop_last=False
                            )
    return dataloader


def choice_collator_fn(data):
    input_ids = torch.concat([d["input_ids"] for d in data])
    labels = torch.concat([d["labels"] for d in data])
    attention_mask = torch.concat([d["attention_mask"] for d in data])
    right_answer = torch.Tensor([d["choice_label"] for d in data]).long()
    return (input_ids, attention_mask), (labels, right_answer, [d['subject'] for d in data], [d['id'] for d in data])



def loss_fn_flatten(outputs, labels):
    logits = outputs
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1), reduce=False
    )


def test(model, eval_dataloader):
    answer_list = []
    subject_list = []
    id_list = []
    for step, batch in enumerate(eval_dataloader):
        inputs = batch[0]
        for i, item in enumerate(inputs):
            inputs[i] = item.cuda()
        labels = batch[1]
        token_labels = labels[0].cuda()
        outputs = model(input_ids=inputs[0], attention_mask=inputs[1])
        ouput_logit = outputs.logits
        loss = loss_fn_flatten(ouput_logit, token_labels)
        loss_3d = loss.view(-1, 4, 2047)
        loss_2d = torch.sum(loss_3d, -1)
        choice = torch.argmin(loss_2d, axis=1)
        answer_list.append(choice)
        subject_list.extend(labels[2])
        id_list.extend(labels[3])
    ans = torch.cat(answer_list).tolist()
    return (ans, subject_list, id_list)


def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    model = model.cuda()
    tokenizer.pad_token = tokenizer.unk_token
    batch_size = args.batch_size

    dataset = EvalDataset_(tokenizer=tokenizer,
                           max_len=2048,
                           shot_num=args.shot_num,
                           dev_path=args.dev_path,
                           test_path=args.test_path)
    eval_dataloader = make_dataloader(dataset, batch_size)

    with torch.no_grad():
        model.eval()

        ans = test(model, eval_dataloader)

    np.save(f"{args.result_path}/output", ans)
    files = os.listdir(args.result_path)
    files = [f for f in files if "output" in f]
    res = {}
    for f in files:
        c = np.load(os.path.join(args.result_path, f))
        for idx, choice in enumerate(c[0]):
            if c[1][idx] not in res.keys():
                res[c[1][idx]] = {}
                res[c[1][idx]][c[2][idx]] = chr(int(c[0][idx]) + 65)
            else:
                res[c[1][idx]][c[2][idx]] = chr(int(c[0][idx]) + 65)
    with open(os.path.join(args.result_path, f"result.json"), mode='w', encoding="utf-8") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
