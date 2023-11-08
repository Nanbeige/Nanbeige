# Copyright (c) 2023 Nanbeige LLM Lab All Rights Reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple command-line interactive chat demo."""

import argparse
import os
import platform
import shutil
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed

DEFAULT_MODEL_PATH = 'Nanbeige/Nanbeige-16B-Chat'

WELCOME_INFOBOX = '''\
/*******************************************************************************/
欢迎使用 Nanbeige-Chat 模型，输入内容进行对话。
(Welcome to use Nanbeige-Chat model, type text to start chat.)

注：本演示受Nanbeige的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。
(Note: This demo is governed by the original license of Nanbeige. We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc.)

命令(Commands):
  :exit | :quit | :q  Exit the demo                       退出Demo
  :clear              Clear screen                        清屏
/*******************************************************************************/

'''

def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu:
        device_map = "cpu"
    else:
        device_map = "auto"

    if not args.cpu and args.fp16:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            resume_download=True,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device_map,
            trust_remote_code=True,
            resume_download=True,
        ).eval()

    config = GenerationConfig.from_pretrained(
        args.model_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def _get_input() -> str:
    while True:
        try:
            message = input('User> ').strip()
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input')
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            return message
        print('[ERROR] Query is empty')


def main():
    parser = argparse.ArgumentParser(
        description='Nanbeige-Chat command-line interactive chat demo.')
    parser.add_argument("-m", "--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu", action="store_false", help="Run demo with CPU only")
    parser.add_argument("--fp16", action="store_true", help="Run demo with fp16")
    args = parser.parse_args()

    history, response = [], ''

    print("正在加载模型....")
    model, tokenizer, config = _load_model_tokenizer(args)
    orig_gen_config = deepcopy(model.generation_config)

    _clear_screen()
    print(WELCOME_INFOBOX)

    while True:
        query = _get_input()

        # Process commands.
        if query.startswith(':'):
            command_words = query[1:].strip().split()
            if not command_words:
                command = ''
            else:
                command = command_words[0]

            if command in ['exit', 'quit', 'q']:
                break
            elif command == 'clear':
                _clear_screen()
                history.clear()
                print(WELCOME_INFOBOX)
                _gc()
                continue

        try:
            for response in model.stream_chat(tokenizer, query, messages=history, temperature=config.temperature, top_p=config.top_p):
                _clear_screen()
                print(f"\nUser: {query}")
                print(f"\nAssistant: {response[0]}")
        except KeyboardInterrupt:
            print('[WARNING] Generation interrupted')
            continue


if __name__ == "__main__":
    main()
