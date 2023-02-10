#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import json
import os
import random

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.nn.functional import softmax
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#python test_causal_response_generator.py --validation_file downstream_data/ESConv/test_dataset.json --model_name_or_path blenderbot_ESConv_EM1_causal_generator_model/
# --tokenizer_name blenderbot_400M_distill/ --twoCondition_tc_model_name_or_path ESConv_unbalance_conditional_tc_EM1_best_model/
# --tc_tokenizer_name roberta_base/ --output_dir ESConv_EM1_causal_generator_test_result


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--validation_file", type=str, default=None, help="downstream_data/ESConv/test_dataset.json"
                                                          "A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
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
        "--num_beams",
        type=int,
        default=10,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--oneCondition_tc_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--twoCondition_tc_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tc_tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    args = parser.parse_args()

    # Sanity checks
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def compute_dependence_score(dependence_tc_model, dependence_tc_tokenizer, dataset, output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    tc_config = AutoConfig.from_pretrained(dependence_tc_model)
    tc_model = AutoModelForSequenceClassification.from_pretrained(
            dependence_tc_model,
            config=tc_config,
        )
    tc_tokenizer = AutoTokenizer.from_pretrained(dependence_tc_tokenizer, truncation_side="left", use_fast=False)
    tc_model.to(device)
    tc_model.eval()

    test_dataset = json.load(open(dataset, "r"))

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            for j in range(len(test_dataset[i]["generated_response"])):
                additional_utterance = test_dataset[i]["generated_response"][j]["additional_utterance"]
                response = test_dataset[i]["generated_response"][j]["generated_response"]
                texts = ([additional_utterance], [response])
                inputs = tc_tokenizer(*texts, padding=True, max_length=128, truncation=True,
                                      return_tensors="pt")
                inputs.to(device)
                outputs = tc_model(**inputs)
                dependence = softmax(outputs.logits, dim=-1)[:,1]
                test_dataset[i]["generated_response"][j]["dependence"] = round(dependence[0].cpu().detach().item(), 4)
    json.dump(test_dataset, open(f"{output_dir}/test_result_in_testset_dependenceScore.json", "w+"), indent=4)

#compute_dependence_score(dependence_tc_model="ESConv_tc_negSameDial_model", dependence_tc_tokenizer="roberta_base/",
#                         dataset="ESConv_allCondition_generator_paper_example_test_result/test_result_in_testset.json",
#                         output_dir="ESConv_allCondition_generator_paper_example_test_result")
#compute_dependence_score(dependence_tc_model="msc_tc_negFromSession_model", dependence_tc_tokenizer="roberta_base/",
#                         dataset="msc_EM1_causal_generator_test_result/test_result_in_testset.json",
#                         output_dir="msc_EM1_causal_generator_test_result")


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load generator
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if "blenderbot" in args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    elif "DialoGPT" in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    # load one condition two-sample classifier model
    #if args.oneCondition_tc_model_name_or_path:
    #    tc_config = AutoConfig.from_pretrained(args.oneCondition_tc_model_name_or_path)
    #else:
    #    tc_config = CONFIG_MAPPING[args.model_type]()
    #    logger.warning("You are instantiating a new config instance from scratch.")

    if args.tc_tokenizer_name:
        tc_tokenizer = AutoTokenizer.from_pretrained(args.tc_tokenizer_name, use_fast=False)
    #elif args.oneCondition_tc_model_name_or_path:
    #    tc_tokenizer = AutoTokenizer.from_pretrained(args.oneCondition_tc_model_name_or_path, use_fast=False)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    #if args.oneCondition_tc_model_name_or_path:
    #    oneCondition_tc_model = AutoModelForSequenceClassification.from_pretrained(
    #        args.oneCondition_tc_model_name_or_path,
    #        from_tf=bool(".ckpt" in args.oneCondition_tc_model_name_or_path),
    #        config=tc_config,
    #    )
    #else:
    #    raise ValueError(
    #        "You need provide a oneCondition tc model."
    #    )

    # load two condition two-sample classifier model
    if args.twoCondition_tc_model_name_or_path:
        tc_config = AutoConfig.from_pretrained(args.twoCondition_tc_model_name_or_path)
    else:
        tc_config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.twoCondition_tc_model_name_or_path:
        twoCondition_tc_model = AutoModelForSequenceClassification.from_pretrained(
            args.twoCondition_tc_model_name_or_path,
            from_tf=bool(".ckpt" in args.twoCondition_tc_model_name_or_path),
            config=tc_config,
        )
    else:
        raise ValueError(
            "You need provide a oneCondition tc model."
        )


    def compute_score(tc_model, sentence_1, sentence_2):
        with torch.no_grad():
            # make batch
            texts = (sentence_1, sentence_2)
            inputs = tc_tokenizer(*texts, padding=True, max_length=args.max_length, truncation=True, return_tensors="pt")
            inputs.to(device)
            outputs = tc_model(**inputs)

            predictions = softmax(outputs.logits, dim=-1)
        return predictions.data[:,1]

    # Prepare everything with our `accelerator`.
    model, twoCondition_tc_model = accelerator.prepare(model, twoCondition_tc_model)

    logger.info("***** Running evaluation *****")
    model.eval()
    #oneCondition_tc_model.eval()
    twoCondition_tc_model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "min_length": 20,
        "num_beams": args.num_beams,
        "no_repeat_ngram_size": 3,
        "encoder_no_repeat_ngram_size": 3,
    }
    max_source_length = min(args.max_source_length, model.config.max_position_embeddings)
    with torch.no_grad():
        test_result = []
        if "ESConv" in args.validation_file:
            test_dataset = json.load(open(args.validation_file, "r"))
            for eg_id, example in tqdm(enumerate(test_dataset), total=len(test_dataset)):
                dialog = example["dialog"]
                for idx in range(len(dialog)):
                    if idx > 1 and dialog[idx-1]["speaker"] == "seeker" and dialog[idx]["speaker"] == "supporter":
                        one_result = {}
                        one_result["dialog_history"] = dialog[:idx+1]
                        one_result["generated_response"] = []

                        preceding_utterance = dialog[idx-1]["content"].strip()
                        twoCondition_contexts = []
                        for j in range(idx-1):
                            additional_utterance = dialog[j]["content"].strip()
                            context = additional_utterance +  " </s> <s> "  +preceding_utterance
                            twoCondition_contexts.append(context)
                        input = tokenizer(twoCondition_contexts, max_length=max_source_length, padding=True,
                                                     truncation=True, return_tensors='pt')
                        input.to(device)
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            input["input_ids"],
                            attention_mask=input["attention_mask"],
                            **gen_kwargs,
                        )
                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                        if isinstance(generated_tokens, tuple):
                            generated_tokens = generated_tokens[0]
                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        print(f"context: {twoCondition_contexts} \n response: {decoded_preds}")

                        for twoCondition_context, response in zip(twoCondition_contexts, decoded_preds):
                            #dependence = compute_score(oneCondition_tc_model, [additional_utterance], [response])
                            conditional_depend = compute_score(twoCondition_tc_model, [twoCondition_context], [response])
                            # compute ppl
                            input = tokenizer(twoCondition_context, max_length=max_source_length, padding=False,
                                                         truncation=True, return_tensors='pt')
                            target = tokenizer(response, max_length=128, padding=False, truncation=True,
                                                          return_tensors='pt')
                            input["labels"] = target["input_ids"]
                            input.to(device)
                            output = model(**input)
                            loss = output.loss
                            ppl = torch.exp(loss)
                            one_generation = {}
                            additional_utterance, preceding_utterance = twoCondition_context.split(" </s> <s> ")
                            one_generation["additional_utterance"] = additional_utterance
                            one_generation["preceding_utterance"] = preceding_utterance
                            one_generation["generated_response"] = response
                            #one_generation["dependence"] = round(dependence[0].cpu().detach().item(), 4)
                            one_generation["conditional_dependence"] = round(conditional_depend[0].cpu().detach().item(), 4)
                            one_generation["ppl"] = round(ppl.cpu().detach().item(), 4)
                            one_result["generated_response"].append(one_generation)

                        test_result.append(one_result)

                json.dump(test_result, open(f"{args.output_dir}/test_result_in_testset.json", "w+"), indent=4)

            json.dump(test_result, open(f"{args.output_dir}/test_result_in_testset.json", "w+"), indent=4)

        elif "msc" in args.validation_file:
            test_dataset = json.load(open(args.validation_file, "r"))
            # add role
            speaker = ["speaker1", "speaker2"]
            for example in test_dataset:
                for session in example["previous_dialogs"]:
                    for id in range(len(session["dialog"])):
                        session["dialog"][id]["role"] = speaker[id % 2]
                for id in range(len(example["dialog"])):
                    example["dialog"][id]["role"] = speaker[id % 2]

            all_dialogues = []
            for example in test_dataset:
                one_dialog = []
                for session in example["previous_dialogs"]:
                    one_dialog += session["dialog"]
                one_dialog += example["dialog"]
                all_dialogues.append(one_dialog)

            # compute causal score
            for dialog in tqdm(all_dialogues):
                for id in tqdm(range(len(dialog))):
                    if id > 1 and dialog[id]["role"] == "speaker2":
                        one_result = {}
                        one_result["dialog_history"] = dialog[:id + 1]
                        one_result["generated_response"] = []

                        preceding_utterance = dialog[id-1]["text"].strip()

                        for j in range(id-1):
                            twoCondition_contexts = []
                            additional_utterance = dialog[j]["text"].strip()
                            context = additional_utterance + " </s> <s> " + preceding_utterance
                            twoCondition_contexts.append(context)
                            input = tokenizer(twoCondition_contexts, max_length=max_source_length, padding=True,
                                              truncation=True, return_tensors='pt')
                            input.to(device)
                            generated_tokens = model.generate(
                                input["input_ids"],
                                attention_mask=input["attention_mask"],
                                **gen_kwargs,
                            )
                            generated_tokens = accelerator.pad_across_processes(
                                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                            )
                            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                            if isinstance(generated_tokens, tuple):
                                generated_tokens = generated_tokens[0]
                            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                            #print(len(twoCondition_contexts), len(decoded_preds))
                            print(f"context: {context} \n response: {decoded_preds}")
                            for twoCondition_context, response in zip(twoCondition_contexts, decoded_preds):
                                #dependence = compute_score(oneCondition_tc_model, [additional_utterance], [response])
                                conditional_depend = compute_score(twoCondition_tc_model, [twoCondition_context], [response])
                                # compute ppl
                                input = tokenizer(twoCondition_context, max_length=max_source_length, padding=False,
                                                  truncation=True, return_tensors='pt')
                                target = tokenizer(response, max_length=128, padding=False, truncation=True,
                                                   return_tensors='pt')
                                input["labels"] = target["input_ids"]
                                input.to(device)
                                output = model(**input)
                                loss = output.loss
                                ppl = torch.exp(loss)
                                additional_utterance, preceding_utterance = twoCondition_context.split(" </s> <s> ")
                                one_generation = {}
                                one_generation["additional_utterance"] = additional_utterance
                                one_generation["preceding_utterance"] = preceding_utterance
                                one_generation["generated_response"] = response
                                #one_generation["dependence"] = round(dependence[0].cpu().detach().item(), 4)
                                one_generation["conditional_dependence"] = round(conditional_depend[0].cpu().detach().item(), 4)
                                one_generation["ppl"] = round(ppl.cpu().detach().item(), 4)

                                one_result["generated_response"].append(one_generation)

                        test_result.append(one_result)

                json.dump(test_result, open(f"{args.output_dir}/test_result_in_testset.json", "w+"), indent=4)

            json.dump(test_result, open(f"{args.output_dir}/test_result_in_testset.json", "w+"), indent=4)

if __name__ == "__main__":
    main()