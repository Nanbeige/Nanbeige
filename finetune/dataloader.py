import os

import datasets
import torch
import transformers


def create_dataset(training_args, sft_config, tokenizer):
    if sft_config.train_file_path:
        train_file_path = sft_config.train_file_path
    elif sft_config.dataset_name:
        train_file_path = sft_config.dataset_name + "/train.jsonl"

    if sft_config.validate_file_path:
        validate_file_path = sft_config.validate_file_path
    elif sft_config.dataset_name:
        validate_file_path = sft_config.dataset_name + "/eval.jsonl"

    raw_datasets = datasets.load_dataset("json", data_files={'train': train_file_path,
                                                             'validation': validate_file_path})


    def process_supervised(record):
        def format_message(message):
            return "\n[{0}]: {1}\n".format(message['role'].capitalize(), message['content'])
        messages = [format_message(item) for item in record['messages']]
        roles = [item['role'] for item in record['messages']]
        tokenized = tokenizer(messages)
        token_ids = []
        attention_mask = []
        is_first = True
        for role, tok_ids, masks in zip(roles, tokenized['input_ids'], tokenized['attention_mask']):
            # remove bos if isn't fisrt message
            if not is_first and tok_ids[0] == tokenizer.bos_token_id:
                tok_ids.pop(0)
                masks.pop(0)

            for tok_id, mask in zip(tok_ids, masks):
                token_ids.append(tok_id)
                attention_mask.append(mask)
            
            if token_ids[-1] != tokenizer.eos_token_id and role == 'assistant':
                # append eos if assistant 
                token_ids.append(tokenizer.eos_token_id)
                attention_mask.append(1)
            elif token_ids[-1] == tokenizer.eos_token_id and role != 'assistant':
                # remove eos if not assistant 
                token_ids.pop()
                attention_mask.pop()
            
            if is_first:
                is_first = False

        processed_record = {
            "input_ids": token_ids[:sft_config.max_length],
            "attention_mask": attention_mask[:sft_config.max_length],
            "labels": token_ids.copy()[:sft_config.max_length]
        }
        # ignore input label, label is ignored if value is -100
        processed_record["labels"][:min(len(tokenized["input_ids"][0]), sft_config.max_length)] = [-100] * min(len(tokenized["input_ids"][0]), sft_config.max_length)
        return {k: torch.tensor(v, dtype=torch.int) for k, v in processed_record.items()}

    with training_args.main_process_first(desc="Process supervised dataset"):
        return raw_datasets.map(
            process_supervised,
            batched=False,
            num_proc=sft_config.preprocess_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Process supervised dataset"
        )

