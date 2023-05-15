

# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import time
import json
import copy

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.nn.functional import softmax
from tqdm.auto import tqdm

import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
#python run_unbalance_conditional_text_classifier_Estep.py --model_name_or_path ./ESConv_unbalance_conditional_tc_model --tokenizer_name_or_path ./roberta_base
#--validation_file ESConv_tc_test_result/ESConv_train_causal.json --output_dir ESConv_conditional_tc_EM_result --save_file ESConv_train_causal_E1.json

#python run_unbalance_conditional_text_classifier_Estep.py --model_name_or_path ./msc_unbalance_conditional_tc_model/ --tokenizer_name_or_path ./roberta_base
#--validation_file msc_tc_test_result/msc_train_causal.json --output_dir msc_conditional_tc_EM_result --save_file msc_train_causal_E1.json
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--save_file", type=str, default=None, help="file to save test result"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Path to pretrained tokenizer.",
        required=True,
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

    # Sanity checks
    if args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.to(device)

    def compute_score(sentence_1, sentence_2):
        model.eval()
        with torch.no_grad():
            # make batch
            texts = (sentence_1, sentence_2)
            inputs = tokenizer(*texts, padding=True, max_length=args.max_length, truncation=True, return_tensors="pt")
            inputs.to(device)
            outputs = model(**inputs)

            predictions = softmax(outputs.logits, dim=-1)
        return predictions.data[:,1]

    if "ESConv" in args.validation_file:
        print("test in ESConversation dataset")
        # load test dataset
        test_dataset = json.load(open(args.validation_file, "r"))

        for dia_id, dialog in tqdm(enumerate(test_dataset), total=len(test_dataset)):
            for idx in range(len(dialog[:-2])):
                context = dialog[idx]["content"].strip() + f" </s> <s> " + dialog[-2]["content"].strip()
                response = dialog[-1]["content"].strip()
                causal_scores = compute_score([context], [response])
                dialog[idx]["conditional_causal_score_with_last_response"] = round(causal_scores[0].cpu().detach().item(), 4)

            if dia_id%1000 == 0:
                json.dump(test_dataset, open(f"{args.output_dir}/{args.save_file}", "w+"), indent=4)
        json.dump(test_dataset, open(f"{args.output_dir}/{args.save_file}", "w+"), indent=4)

    elif "msc" in args.validation_file:
        print("test in msc dataset")
        # load test dataset
        test_dataset = json.load(open(args.validation_file, "r"))

        for dia_id, dialog in tqdm(enumerate(test_dataset), total=len(test_dataset)):
            for idx in range(len(dialog[:-2])):
                context = dialog[idx]["text"].strip() + f" </s> <s> " + dialog[-2]["text"].strip()
                response = dialog[-1]["text"].strip()
                causal_scores = compute_score([context], [response])
                dialog[idx]["conditional_causal_score_with_last_response"] = round(causal_scores[0].cpu().detach().item(), 4)

            if dia_id % 1000 == 0:
                json.dump(test_dataset, open(f"{args.output_dir}/{args.save_file}", "w+"), indent=4)
        json.dump(test_dataset, open(f"{args.output_dir}/{args.save_file}", "w+"), indent=4)

if __name__ == "__main__":
    main()