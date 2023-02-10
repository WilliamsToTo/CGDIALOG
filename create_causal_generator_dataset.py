import json
import pandas as pd

# create dataset for causal generator
# ESConv dataset
#dataset_path = "ESConv_tc_test_result/ESConv_unbalance_conditional_train_causal.json"
#save_path = "ESConv_causal_generator_dataset/ESConv_unbalance_conditional_causal_generator_train.csv"
def ESConv_create_causal_generator_dataset(dataset_path, save_path):
    print("ESConv_create_causal_generator_dataset")
    dataset = json.load(open(dataset_path, "r"))
    context_response_pairs = []
    for dialog in dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] and dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > 0.5:
                    highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                    highest_idx = i
            if highest_idx != -1:
                context = dialog[highest_idx]["content"].strip() + " </s> <s> " + dialog[-2]["content"].strip()
                response = dialog[-1]["content"].strip()
                context_response_pairs.append((context, response))
    print(len(context_response_pairs))
    rg_dataset = pd.DataFrame(context_response_pairs, columns=["context", "response"])
    rg_dataset.to_csv(save_path, index=False, sep="\t")



#dataset_path = "./msc_tc_test_result/msc_unbalance_conditional_train_causal.json"
#save_path = "./msc_causal_generator_dataset/msc_unbalance_conditional_causal_generator_train.csv"
def msc_create_causal_generator_dataset(dataset_path, save_path):
    print("msc_create_causal_generator_dataset")
    dataset = json.load(open(dataset_path, "r"))
    context_response_pairs = []
    for dialog in dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] and dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > 0.5:
                    highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                    highest_idx = i
            if highest_idx != -1:
                context = dialog[highest_idx]["text"].strip() + " </s> <s> " + dialog[-2]["text"].strip()
                response = dialog[-1]["text"].strip()
                context_response_pairs.append((context, response))
    print(len(context_response_pairs))
    rg_dataset = pd.DataFrame(context_response_pairs, columns=["context", "response"])
    rg_dataset.to_csv(save_path, index=False, sep="\t")

#ESConv_create_causal_generator_dataset(dataset_path = "ESConv_tc_test_result/ESConv_EM1_conditional_train_causal.json", save_path = "ESConv_causal_generator_dataset/ESConv_unbalance_EM1_conditional_causal_generator_train.csv")
#msc_create_causal_generator_dataset(dataset_path = "./msc_tc_test_result/msc_EM1_conditional_train_causal.json", save_path = "./msc_causal_generator_dataset/msc_unbalance_EM1_conditional_causal_generator_train.csv")

def ESConv_DialoGPT_create_causal_generator_dataset(dataset_path, save_path):
    print("ESConv_DialoGPT_create_causal_generator_dataset")
    dataset = json.load(open(dataset_path, "r"))
    context_response_pairs = []
    for dialog in dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] and dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > 0.5:
                    highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                    highest_idx = i
            if highest_idx != -1:
                context = dialog[highest_idx]["content"].strip() + "<|endoftext|>" + dialog[-2]["content"].strip()
                response = dialog[-1]["content"].strip()
                context_response_pairs.append(context + "<|endoftext|>" + response)
    print(len(context_response_pairs))
    rg_dataset = pd.DataFrame(context_response_pairs, columns=["text"])
    rg_dataset.to_csv(save_path, index=False, sep="\t")

def msc_DialoGPT_create_causal_generator_dataset(dataset_path, save_path):
    print("msc_DialoGPT_create_causal_generator_dataset")
    dataset = json.load(open(dataset_path, "r"))
    context_response_pairs = []
    for dialog in dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] and dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > 0.5:
                    highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                    highest_idx = i
            if highest_idx != -1:
                context = dialog[highest_idx]["text"].strip() + "<|endoftext|>" + dialog[-2]["text"].strip()
                response = dialog[-1]["text"].strip()
                context_response_pairs.append(context + "<|endoftext|>" + response)
    print(len(context_response_pairs))
    rg_dataset = pd.DataFrame(context_response_pairs, columns=["text"])
    rg_dataset.to_csv(save_path, index=False, sep="\t")

ESConv_DialoGPT_create_causal_generator_dataset(dataset_path = "ESConv_tc_test_result/ESConv_EM1_conditional_train_causal.json", save_path = "ESConv_causal_generator_dataset/ESConv_DialoGPT_causal_generator_train.csv")
msc_DialoGPT_create_causal_generator_dataset(dataset_path = "./msc_tc_test_result/msc_EM1_conditional_train_causal.json", save_path = "./msc_causal_generator_dataset/msc_DialoGPT_causal_generator_train.csv")


def ESConv_create_dependent_generator_dataset(dataset_path, save_path):
    print(f"ESConv_create_dependent_generator_dataset: {dataset_path} {save_path}")
    # pick u_{j} with highest dependent score
    dataset = json.load(open(dataset_path, "r"))
    context_response_pairs = []
    for dialog in dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "causal_score_with_last_response" in dialog[i] and dialog[i]["causal_score_with_last_response"] > highest_score \
                        and dialog[i]["causal_score_with_last_response"] > 0.5:
                    highest_score = dialog[i]["causal_score_with_last_response"]
                    highest_idx = i
            if highest_idx != -1:
                context = dialog[highest_idx]["content"].strip() + "  </s> <s>  " + dialog[-2]["content"].strip()
                response = dialog[-1]["content"].strip()
                context_response_pairs.append((context, response))
    print(len(context_response_pairs))
    rg_dataset = pd.DataFrame(context_response_pairs, columns=["context", "response"])
    rg_dataset.to_csv(save_path, index=False, sep="\t")

def msc_create_dependent_generator_dataset(dataset_path, save_path):
    print(f"msc_create_dependent_generator_dataset: {dataset_path} {save_path}")
    dataset = json.load(open(dataset_path, "r"))
    context_response_pairs = []
    for dialog in dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "causal_score_with_last_response" in dialog[i] and dialog[i]["causal_score_with_last_response"] > highest_score \
                        and dialog[i]["causal_score_with_last_response"] > 0.5:
                    highest_score = dialog[i]["causal_score_with_last_response"]
                    highest_idx = i
            if highest_idx != -1:
                context = dialog[highest_idx]["text"].strip() + " </s> <s> " + dialog[-2]["text"].strip()
                response = dialog[-1]["text"].strip()
                context_response_pairs.append((context, response))
    print(len(context_response_pairs))
    rg_dataset = pd.DataFrame(context_response_pairs, columns=["context", "response"])
    rg_dataset.to_csv(save_path, index=False, sep="\t")


def ESConv_create_PC_causal_generator_dataset(dataset_path, save_path):
    print("ESConv_create_PC_causal_generator_dataset")
    dataset = json.load(open(dataset_path, "r"))
    context_response_pairs = []
    for dialog in dataset:
        if len(dialog["predicted_causal_utterances"]) > 0:
            causes = []
            for utter in dialog["history"]:
                if utter["content"] in dialog["predicted_causal_utterances"]:
                    causes.append(utter["content"].strip())
            context = " </s> <s> ".join(causes)
            response = dialog["response"]["content"].strip()
            context_response_pairs.append((context, response))
    print(len(context_response_pairs))
    rg_dataset = pd.DataFrame(context_response_pairs, columns=["context", "response"])
    rg_dataset.to_csv(save_path, index=False, sep="\t")

def msc_create_PC_causal_generator_dataset(dataset_path, save_path):
    print("msc_create_PC_causal_generator_dataset")
    dataset = json.load(open(dataset_path, "r"))
    context_response_pairs = []
    for dialog in dataset:
        if len(dialog["predicted_causal_utterances"]) > 0:
            causes = []
            for utter in dialog["history"]:
                if utter["text"] in dialog["predicted_causal_utterances"]:
                    causes.append(utter["text"].strip())
            context = " </s> <s> ".join(causes)
            response = dialog["response"]["text"].strip()
            context_response_pairs.append((context, response))
    print(len(context_response_pairs))
    rg_dataset = pd.DataFrame(context_response_pairs, columns=["context", "response"])
    rg_dataset.to_csv(save_path, index=False, sep="\t")

#ESConv_create_PC_causal_generator_dataset(dataset_path = "ESConv_causal_generator_dataset/ESConv_PC_generator_train_dataset.json", save_path = "ESConv_causal_generator_dataset/ESConv_PC_causal_generator_train.csv")
#msc_create_PC_causal_generator_dataset(dataset_path = "msc_causal_generator_dataset/msc_PC_generator_train_dataset.json", save_path = "./msc_causal_generator_dataset/msc_PC_causal_generator_train.csv")


def ESConv_create_equation2_causal_generator_dataset(dataset_path, save_path):
    print("ESConv_create_equation2_causal_generator_dataset")
    dataset = json.load(open(dataset_path, "r"))
    context_response_pairs = []
    for dialog in dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "ESConv_additional_utterance_score" in dialog[i] and dialog[i]["ESConv_additional_utterance_score"] > highest_score:
                    highest_score = dialog[i]["ESConv_additional_utterance_score"]
                    highest_idx = i
            if highest_idx != -1:
                context = dialog[highest_idx]["content"].strip() + " </s> <s> " + dialog[-2]["content"].strip()
                response = dialog[-1]["content"].strip()
                context_response_pairs.append((context, response))
    print(len(context_response_pairs))
    rg_dataset = pd.DataFrame(context_response_pairs, columns=["context", "response"])
    rg_dataset.to_csv(save_path, index=False, sep="\t")

def msc_create_equation2_causal_generator_dataset(dataset_path, save_path):
    print("msc_create_equation2_causal_generator_dataset")
    dataset = json.load(open(dataset_path, "r"))
    context_response_pairs = []
    for dialog in dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "msc_additional_utterance_score" in dialog[i] and dialog[i]["msc_additional_utterance_score"] > highest_score:
                    highest_score = dialog[i]["msc_additional_utterance_score"]
                    highest_idx = i
            if highest_idx != -1:
                context = dialog[highest_idx]["text"].strip() + "  </s> <s>  " + dialog[-2]["text"].strip()
                response = dialog[-1]["text"].strip()
                context_response_pairs.append((context, response))
    print(len(context_response_pairs))
    rg_dataset = pd.DataFrame(context_response_pairs, columns=["context", "response"])
    rg_dataset.to_csv(save_path, index=False, sep="\t")

#ESConv_create_equation2_causal_generator_dataset(dataset_path = "ESConv_additional_utterance_predict/ESConv_train_additional_score.json", save_path = "ESConv_causal_generator_dataset/ESConv_equation2_causal_generator_train.csv")
#msc_create_equation2_causal_generator_dataset(dataset_path = "msc_additional_utterance_predict/msc_train_additional_score.json", save_path = "./msc_causal_generator_dataset/msc_equation2_causal_generator_train.csv")
