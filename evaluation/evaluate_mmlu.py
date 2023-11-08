# Copyright (c) 2023 Nanbeige LLM Lab All Rights Reserved.

import argparse
import transformers
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader, IterableDataset
import deepspeed
import copy
import pandas as pd
import os
from transformers import AutoModelForCausalLM

IGNORE_INDEX = -100
PAD_TOKEN_ID = 0
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

LABEL2IDX = {"A": 0,
             "B": 1,
             "C": 2,
             "D": 3}

IDX2LABEL = {0: "A",
             1: "B",
             2: "C",
             3: "D"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        default=None,
                        help='model path')
    parser.add_argument('--dev_path',
                        type=str,
                        default=None,
                        help='mmlu dev set path')
    parser.add_argument('--test_path',
                        type=str,
                        default=None,
                        help='mmlu test set path')
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='batchsize')
    parser.add_argument('--shot_num',
                        type=int,
                        default=5,
                        help='batchsize')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def make_mmlu_dataset(tokenizer, shot_num, dev_path, test_path):
    
    file_names = [f for f in os.listdir(test_path) if "csv" in f]
    text_samples = []
    for file_name in file_names:
        name_split = file_name.split('_')
        name_prefix = '_'.join(name_split[:-1])
        name_subfix = "_dev.csv"
        dev_file_name = name_prefix + name_subfix
        df = pd.DataFrame(pd.read_csv(test_path + file_name, header=None))
        dev_df = pd.DataFrame(pd.read_csv(dev_path + dev_file_name, header=None))
        dev_context = ""
        for i in range(shot_num):
            df_sample = dev_df.loc[i]
            question = df_sample[0]
            answer_letter = df_sample[5]
            choices = [df_sample[j] for j in range(1, 5)]
            context = question + '\n' \
                      + 'A. ' + str(choices[0]) \
                      + '\nB. ' + str(choices[1]) \
                      + '\nC. ' + str(choices[2]) \
                      + '\nD. ' + str(choices[3]) + '\nAnswer: '
            context = context + answer_letter + '\n\n'
            dev_context += context
        for i in range(len(df)):
            df_sample = df.loc[i]
            test_question = df_sample[0]
            test_answer_letter = df_sample[5]
            test_choices = [df_sample[j] for j in range(1, 5)]
            test_context = test_question + '\n' \
                           + 'A. ' + str(test_choices[0]) \
                           + '\nB. ' + str(test_choices[1]) \
                           + '\nC. ' + str(test_choices[2]) \
                           + '\nD. ' + str(test_choices[3]) + '\nAnswer: '
            test_context = dev_context + test_context
            test_sample = (test_context, test_answer_letter)
            text_samples.append(test_sample)

    dataset = text_samples
    dataset_processed = []
    for idx in range(len(dataset)):
        select_idx = idx
        text_sample = dataset[select_idx]
        context, right_letter = text_sample
        right_index = LABEL2IDX[right_letter]
        context_length = tokenizer(context, return_tensors="pt")["input_ids"].squeeze(0).size()[-1] - 1
        inputs = []
        labels = []
        attention_masks = []
        for choice in ("A", "B", "C", "D"):
            concat_text = context + str(choice)
            chosen_tokens = tokenizer(concat_text,
                                           max_length=4096,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt",
                                           )
            input_ids = chosen_tokens["input_ids"].squeeze(0)
            attention_mask = chosen_tokens["attention_mask"].squeeze(0)
            token_labels = copy.deepcopy(input_ids)
            token_labels[:context_length] = IGNORE_INDEX
            token_labels = torch.where(token_labels == PAD_TOKEN_ID, IGNORE_INDEX, token_labels)
            token_labels = torch.where(token_labels == 2, IGNORE_INDEX, token_labels)
            inputs.append(input_ids[:-1])
            labels.append(token_labels[1:])
            attention_masks.append(attention_mask[:-1])
        sample = (torch.stack(inputs), torch.stack(attention_masks), torch.stack(labels), right_index)
        dataset_processed.append(sample)
    
    return dataset_processed


def loss_fn_flatten(outputs, labels):
    logits = outputs
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1), reduce=False
    )


def evaluation(model, eval_dataset, device):
    count = 0
    correct = 0
    for step, sample in enumerate(eval_dataset):
        input_ids = sample[0]
        input_ids = input_ids.to(device)
        attention_mask = sample[1]
        attention_mask = attention_mask.to(device)
        labels = sample[2]
        labels = labels.to(device)
        right_index = sample[3]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        ouput_logit = outputs.logits
        loss = loss_fn_flatten(ouput_logit, labels)
        loss_3d = loss.view(1, 4, -1)
        loss_2d = torch.sum(loss_3d, -1)
        choice = torch.argmin(loss_2d, axis=1)
        batch_correct = torch.sum(choice == right_index).int()
        count += 1
        correct += batch_correct
        if step % 200 == 0:
            print("acc", correct / float(count + 1), flush=True)
    acc = correct / (count + 1)
    print("avg acc", acc, flush=True)
    return acc

def main():
    args = parse_args()
    assert (not args.local_rank == -1)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed(dist_backend=args.backend)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_path,
        model_max_length=4096,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    eval_dataset = make_mmlu_dataset(tokenizer, args.shot_num, args.dev_path, args.test_path)
    model_config = transformers.AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_config.torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=model_config, torch_dtype=torch.bfloat16)
    model.eval()
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    model = deepspeed.init_inference(model,
                                     dtype="bfloat16",
                                     max_out_tokens=4096,
                                     replace_with_kernel_inject=False,
                                     mp_size=world_size)
    with torch.no_grad():
        acc = evaluation(model, eval_dataset, device)
        print("final_acc\t", acc, flush=True)
    torch.distributed.barrier()


if __name__ == "__main__":
    main()