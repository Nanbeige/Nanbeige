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
            assert message['role'] in ['human', 'assistant', 'system']
            if message['role'] == 'human':
                return f"""### Human: \n{message['content']}\n\n"""
            elif message['role'] == 'assistant':
                return f"""### Assistant: {message['content']}</s>"""
            elif message['role'] == 'system':
                return f"""### System:{message['content']}\n</s>"""

        messages = [format_message(item) for item in record['messages']]
        roles = [item['role'] for item in record['messages']]
        tokenized = tokenizer(messages)
        input_ids = []
        labels = []
        attention_mask = []
        for role, tok_ids, masks in zip(roles, tokenized['input_ids'], tokenized['attention_mask']):
            # remove bos if isn't fisrt message
            if tok_ids[0] == tokenizer.bos_token_id:
                tok_ids.pop(0)
                masks.pop(0)

            for tok_id, mask in zip(tok_ids, masks):
                input_ids.append(tok_id)
                if role == 'assistant':
                    labels.append(tok_id)
                else:
                    labels.append(-100)
                attention_mask.append(mask)
            
            if input_ids[-1] != tokenizer.eos_token_id and role == 'assistant':
                # append eos if assistant 
                input_ids.append(tokenizer.eos_token_id)
                labels.append(tokenizer.eos_token_id)
                attention_mask.append(1)
            elif input_ids[-1] == tokenizer.eos_token_id and role != 'assistant':
                # remove eos if not assistant 
                input_ids.pop()
                labels.pop()
                attention_mask.pop()
            
        if len(input_ids) < sft_config.max_length:
            input_ids += [tokenizer.pad_token_id] * (sft_config.max_length - len(input_ids))

        if len(labels) < sft_config.max_length:
            labels += [-100] * (sft_config.max_length - len(labels))

        if len(attention_mask) < sft_config.max_length:
            attention_mask += [0] * (sft_config.max_length - len(attention_mask))

        processed_record = {
            "input_ids": input_ids[:sft_config.max_length],
            "labels": labels[:sft_config.max_length],
            "attention_mask": attention_mask[:sft_config.max_length],
        }
        return {k: torch.LongTensor(v) for k, v in processed_record.items()}
        
    with training_args.main_process_first(desc="Process supervised dataset"):
        return raw_datasets.map(
            process_supervised,
            batched=False,
            num_proc=sft_config.preprocess_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Process supervised dataset"
        )

