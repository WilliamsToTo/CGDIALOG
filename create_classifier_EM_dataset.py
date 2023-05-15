import os
import json
import random
import pandas as pd
from collections import OrderedDict

def find_threshold_by_ratio(dataset, ratio=0.3):
    # find the threshold by ratio
    highest_score_list = []
    for dialog in dataset:
        if len(dialog) > 2:
            highest_score = -1000
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] \
                        and dialog[i]["conditional_causal_score_with_last_response"] > highest_score:
                    highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                    highest_score_list.append(highest_score)
    highest_score_list.sort()
    threshold_index = int(len(highest_score_list)*ratio)
    threshold = highest_score_list[threshold_index]

    return threshold

def compare_ESConv_E_datasets(previous_dataset, current_dataset):
    print(previous_dataset, current_dataset)
    if "E1" in current_dataset:
        current_dataset = json.load(open(current_dataset, "r"))
        difference_count = 0
        for dialog in current_dataset:
            if len(dialog) > 2:
                previous_u_j = dialog[-3]["content"]
                current_u_j = ""
                highest_causal_score = -100
                for utter in dialog:
                    if "conditional_causal_score_with_last_response" in utter and \
                            utter["conditional_causal_score_with_last_response"]>highest_causal_score:
                        highest_causal_score = utter["conditional_causal_score_with_last_response"]
                        current_u_j = utter["content"]
                if previous_u_j != current_u_j:
                    difference_count += 1
        print(f"difference {difference_count} over total {len(current_dataset)}")
    else:
        previous_dataset = json.load(open(previous_dataset, "r"))
        current_dataset = json.load(open(current_dataset, "r"))
        difference_count = 0
        print(len(previous_dataset), len(current_dataset))
        previous_threshold = find_threshold_by_ratio(previous_dataset)
        current_threshold = find_threshold_by_ratio(current_dataset)
        for pre_dial, cur_dial in zip(previous_dataset, current_dataset):
            if len(pre_dial)!=len(cur_dial) and pre_dial[-1]["content"] == cur_dial[-1]["content"]:
                raise ValueError(
                    "Warning, dialogue doesn't match"
                )
            if len(pre_dial)>2:
                previous_u_j = ""
                highest_causal_score = -100
                for utter in pre_dial:
                    if "conditional_causal_score_with_last_response" in utter and \
                            utter["conditional_causal_score_with_last_response"]>highest_causal_score and \
                            utter["conditional_causal_score_with_last_response"]>previous_threshold:
                        highest_causal_score = utter["conditional_causal_score_with_last_response"]
                        previous_u_j = utter["content"]

                current_u_j = ""
                for utter in cur_dial:
                    if "conditional_causal_score_with_last_response" in utter and \
                            utter["conditional_causal_score_with_last_response"]>highest_causal_score and \
                            utter["conditional_causal_score_with_last_response"]>current_threshold:
                        highest_causal_score = utter["conditional_causal_score_with_last_response"]
                        current_u_j = utter["content"]
                if previous_u_j != current_u_j:
                    difference_count += 1
        print(f"difference {difference_count} over total {len(current_dataset)}")
#compare_ESConv_E_datasets(previous_dataset="", current_dataset="ESConv_conditional_tc_EM_result/ESConv_train_causal_E1.json")

def create_ESConv_M_training_datasets(load_path, positive_save_path, negative_save_path, valid_save_path):
    # create positive/negative file
    # create validation file
    print(load_path)
    dataset = json.load(open(load_path, "r"))
    valid_length = int(len(dataset)*0.2)
    train_dataset = dataset[:len(dataset)-valid_length]
    valid_dataset = dataset[len(dataset)-valid_length:]
    positive_pairs = []
    negative_pairs = []
    # get threshold of one cause dialogues
    threshold = find_threshold_by_ratio(train_dataset)
    print("Threshold of one cause dialogue: ", threshold)
    for dialog in train_dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] \
                        and dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > threshold:
                        highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                        highest_idx = i

            if highest_idx != -1:
                response = dialog[-1]["content"].strip()
                positive_context = dialog[highest_idx]["content"].strip() + f" </s> <s> " + dialog[-2]["content"].strip()
                positive_pairs.append((positive_context, response, 1))
                for j in range(len(dialog)):
                    if j != highest_idx and "conditional_causal_score_with_last_response" in dialog[j]:
                            #and dialog[j]["conditional_causal_score_with_last_response"] < 0.5:
                        negative_context = dialog[j]["content"].strip() + f" </s> <s> " + dialog[-2]["content"].strip()
                        negative_pairs.append((negative_context, response, 0))

    print(len(positive_pairs), len(negative_pairs))
    positive_pairs = [ele for ele in positive_pairs for i in range(len(negative_pairs) // len(positive_pairs))]
    print(len(positive_pairs), len(negative_pairs))
    extend_pairs = random.sample(positive_pairs, len(negative_pairs) - len(positive_pairs))
    positive_pairs += extend_pairs
    print(len(positive_pairs), len(negative_pairs))

    conditional_tc_dataset = pd.DataFrame(positive_pairs, columns=["utterance", "response", "label"])
    conditional_tc_dataset.to_csv(positive_save_path, index=False, sep="\t")

    conditional_tc_dataset = pd.DataFrame(negative_pairs, columns=["utterance", "response", "label"])
    conditional_tc_dataset.to_csv(negative_save_path, index=False, sep="\t")

    valid_pairs = []
    for dialog in valid_dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] and \
                        dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > threshold:
                        highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                        highest_idx = i

            if highest_idx != -1:
                response = dialog[-1]["content"].strip()
                positive_context = dialog[highest_idx]["content"].strip() + f" </s> <s> " + dialog[-2]["content"].strip()
                valid_pairs.append((positive_context, response, 1))
                for j in range(len(dialog)):
                    if j != highest_idx and "conditional_causal_score_with_last_response" in dialog[j]:
                        negative_context = dialog[j]["content"].strip() + f" </s> <s> " + dialog[-2]["content"].strip()
                        valid_pairs.append((negative_context, response, 0))
                        break
    valid_conditional_tc_dataset = pd.DataFrame(valid_pairs, columns=["utterance", "response", "label"])
    valid_conditional_tc_dataset.to_csv(valid_save_path, index=False, sep="\t")
#create_ESConv_M_training_datasets(load_path="ESConv_conditional_tc_EM_result/ESConv_train_causal_E1.json",
#                           positive_save_path="ESConv_conditional_tc_EM_result/ESConv_train_causal_M1_positive.csv",
#                           negative_save_path="ESConv_conditional_tc_EM_result/ESConv_train_causal_M1_negative.csv",
#                           valid_save_path="ESConv_conditional_tc_EM_result/ESConv_train_causal_M1_valid.csv")

def create_ESConv_M_incremental_training_datasets(load_path, previous_positive_path, previous_negative_path,
                                                  positive_save_path, negative_save_path, valid_save_path):
    # create confident positive/negative data samples
    # create validation file
    print(load_path)
    dataset = json.load(open(load_path, "r"))
    previous_positive_dataset = pd.read_csv(previous_positive_path, sep="\t")
    previous_positive_dataset = list(previous_positive_dataset.itertuples(index=False, name=None))
    previous_negative_dataset = pd.read_csv(previous_negative_path, sep="\t")
    previous_negative_dataset = list(previous_negative_dataset.itertuples(index=False, name=None))
    valid_length = int(len(dataset)*0.2)
    train_dataset = dataset[:len(dataset)-valid_length]
    valid_dataset = dataset[len(dataset)-valid_length:]
    positive_pairs = []
    negative_pairs = []
    # get threshold of one cause dialogues
    threshold = 0.90 #max(find_threshold_by_ratio(train_dataset, ratio=1-0.05), 0.95)
    #print(find_threshold_by_ratio(train_dataset, ratio=1-0.05))
    print(f"Threshold of one cause dialogue: {threshold}")
    for dialog in train_dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] \
                        and dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > threshold:
                        highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                        highest_idx = i

            if highest_idx != -1:
                response = dialog[-1]["content"].strip()
                positive_context = dialog[highest_idx]["content"].strip() + " </s> <s> " + dialog[-2]["content"].strip()
                positive_pairs.append((positive_context, response, 1))
                for j in random.sample(range(0, len(dialog)), 3):
                    if j != highest_idx and "conditional_causal_score_with_last_response" in dialog[j]:
                        negative_context = dialog[j]["content"].strip() + f" </s> <s> " + dialog[-2]["content"].strip()
                        negative_pairs.append((negative_context, response, 0))
    print("True samples: ", len(previous_positive_dataset), len(previous_negative_dataset))
    print("New samples: ", len(positive_pairs), len(negative_pairs))
    positive_pairs += previous_positive_dataset
    positive_pairs = list(OrderedDict.fromkeys(positive_pairs))
    negative_pairs += previous_negative_dataset
    negative_pairs = list(OrderedDict.fromkeys(negative_pairs))

    print("Incremental samples: ", len(positive_pairs), len(negative_pairs))
    positive_pairs = [ele for ele in positive_pairs for i in range(len(negative_pairs) // len(positive_pairs))]
    print(len(positive_pairs), len(negative_pairs))
    extend_pairs = random.sample(positive_pairs, len(negative_pairs) - len(positive_pairs))
    positive_pairs += extend_pairs
    print(len(positive_pairs), len(negative_pairs))

    current_positive_dataset = pd.DataFrame(positive_pairs, columns=["utterance", "response", "label"])
    #current_positive_dataset = pd.concat([previous_positive_dataset, current_positive_dataset], ignore_index=True)
    current_positive_dataset.to_csv(positive_save_path, index=False, sep="\t")

    current_negative_dataset = pd.DataFrame(negative_pairs, columns=["utterance", "response", "label"])
    #current_negative_dataset = pd.concat([previous_negative_dataset, current_negative_dataset], ignore_index=True)
    current_negative_dataset.to_csv(negative_save_path, index=False, sep="\t")

    valid_pairs = []
    for dialog in valid_dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] and \
                        dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > threshold:
                        highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                        highest_idx = i

            if highest_idx != -1:
                response = dialog[-1]["content"].strip()
                positive_context = dialog[highest_idx]["content"].strip() + f" </s> <s> " + dialog[-2]["content"].strip()
                valid_pairs.append((positive_context, response, 1))
                for j in random.sample(range(0, len(dialog)), 3):
                    if j != highest_idx and "conditional_causal_score_with_last_response" in dialog[j]:
                        negative_context = dialog[j]["content"].strip() + f" </s> <s> " + dialog[-2]["content"].strip()
                        valid_pairs.append((negative_context, response, 0))
                        break
    valid_conditional_tc_dataset = pd.DataFrame(valid_pairs, columns=["utterance", "response", "label"])
    valid_conditional_tc_dataset.to_csv(valid_save_path, index=False, sep="\t")

def compare_msc_E_datasets(previous_dataset, current_dataset):
    print(previous_dataset, current_dataset)
    if "E1" in current_dataset:
        current_dataset = json.load(open(current_dataset, "r"))
        difference_count = 0
        for dialog in current_dataset:
            if len(dialog) > 2:
                previous_u_j = dialog[-3]["text"]
                current_u_j = ""
                highest_causal_score = -100
                for utter in dialog:
                    if "conditional_causal_score_with_last_response" in utter and \
                            utter["conditional_causal_score_with_last_response"] > highest_causal_score:
                        highest_causal_score = utter["conditional_causal_score_with_last_response"]
                        current_u_j = utter["text"]
                if previous_u_j != current_u_j:
                    difference_count += 1
        print(f"difference {difference_count} over total {len(current_dataset)}")
    else:
        previous_dataset = json.load(open(previous_dataset, "r"))
        current_dataset = json.load(open(current_dataset, "r"))
        difference_count = 0
        print(len(previous_dataset), len(current_dataset))
        previous_threshold = find_threshold_by_ratio(previous_dataset)
        current_threshold = find_threshold_by_ratio(current_dataset)
        for pre_dial, cur_dial in zip(previous_dataset, current_dataset):
            if len(pre_dial) != len(cur_dial) and pre_dial[-1]["text"] == cur_dial[-1]["text"]:
                raise ValueError(
                    "Warning, dialogue doesn't match"
                )
            if len(pre_dial) > 2:
                previous_u_j = ""
                highest_causal_score = -100
                for utter in pre_dial:
                    if "conditional_causal_score_with_last_response" in utter and \
                            utter["conditional_causal_score_with_last_response"] > highest_causal_score and \
                            utter["conditional_causal_score_with_last_response"] > previous_threshold:
                        highest_causal_score = utter["conditional_causal_score_with_last_response"]
                        previous_u_j = utter["text"]

                current_u_j = ""
                for utter in cur_dial:
                    if "conditional_causal_score_with_last_response" in utter and \
                            utter["conditional_causal_score_with_last_response"] > highest_causal_score and \
                            utter["conditional_causal_score_with_last_response"] > current_threshold:
                        highest_causal_score = utter["conditional_causal_score_with_last_response"]
                        current_u_j = utter["text"]
                if previous_u_j != current_u_j:
                    difference_count += 1
        print(f"difference {difference_count} over total {len(current_dataset)}")
#compare_msc_E_datasets(previous_dataset="", current_dataset="msc_conditional_tc_EM_result/msc_train_causal_E1.json")


def create_msc_M_training_datasets(load_path, positive_save_path, negative_save_path, valid_save_path):
    # create positive/negative file
    # create validation file
    print(load_path)
    dataset = json.load(open(load_path, "r"))
    valid_length = int(len(dataset) * 0.2)
    train_dataset = dataset[:len(dataset) - valid_length]
    valid_dataset = dataset[len(dataset) - valid_length:]
    positive_pairs = []
    negative_pairs = []
    # get threshold of one cause dialogues
    threshold = find_threshold_by_ratio(train_dataset)
    print("Threshold of one cause dialogue: ", threshold)
    for dialog in train_dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] \
                        and dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > threshold:
                    highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                    highest_idx = i

            if highest_idx != -1:
                response = dialog[-1]["text"].strip()
                positive_context = dialog[highest_idx]["text"].strip() + f" </s> <s> " + dialog[-2]["text"].strip()
                positive_pairs.append((positive_context, response, 1))
                for j in range(len(dialog)):
                    if j != highest_idx and "conditional_causal_score_with_last_response" in dialog[j]:
                        # and dialog[j]["conditional_causal_score_with_last_response"] < 0.5:
                        negative_context = dialog[j]["text"].strip() + f" </s> <s> " + dialog[-2]["text"].strip()
                        negative_pairs.append((negative_context, response, 0))

    print(len(positive_pairs), len(negative_pairs))
    positive_pairs = [ele for ele in positive_pairs for i in range(len(negative_pairs) // len(positive_pairs))]
    print(len(positive_pairs), len(negative_pairs))
    extend_pairs = random.sample(positive_pairs, len(negative_pairs) - len(positive_pairs))
    positive_pairs += extend_pairs
    print(len(positive_pairs), len(negative_pairs))

    conditional_tc_dataset = pd.DataFrame(positive_pairs, columns=["utterance", "response", "label"])
    conditional_tc_dataset.to_csv(positive_save_path, index=False, sep="\t")

    conditional_tc_dataset = pd.DataFrame(negative_pairs, columns=["utterance", "response", "label"])
    conditional_tc_dataset.to_csv(negative_save_path, index=False, sep="\t")

    valid_pairs = []
    for dialog in valid_dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] and \
                        dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > threshold:
                    highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                    highest_idx = i

            if highest_idx != -1:
                response = dialog[-1]["text"].strip()
                positive_context = dialog[highest_idx]["text"].strip() + f" </s> <s> " + dialog[-2]["text"].strip()
                valid_pairs.append((positive_context, response, 1))
                for j in range(len(dialog)):
                    if j != highest_idx and "conditional_causal_score_with_last_response" in dialog[j]:
                        negative_context = dialog[j]["text"].strip() + f" </s> <s> " + dialog[-2]["text"].strip()
                        valid_pairs.append((negative_context, response, 0))
                        break
    valid_conditional_tc_dataset = pd.DataFrame(valid_pairs, columns=["utterance", "response", "label"])
    valid_conditional_tc_dataset.to_csv(valid_save_path, index=False, sep="\t")
#create_msc_M_training_datasets(load_path="msc_conditional_tc_EM_result/msc_train_causal_E1.json",
#                           positive_save_path="msc_conditional_tc_EM_result/msc_train_causal_M1_positive.csv",
#                           negative_save_path="msc_conditional_tc_EM_result/msc_train_causal_M1_negative.csv",
#                           valid_save_path="msc_conditional_tc_EM_result/msc_train_causal_M1_valid.csv")

def create_msc_M_incremental_training_datasets(load_path, previous_positive_path, previous_negative_path,
                                                  positive_save_path, negative_save_path, valid_save_path):
    # create positive/negative file
    # create validation file
    print(load_path)
    dataset = json.load(open(os.path.expanduser(load_path), "r"))
    previous_positive_dataset = pd.read_csv(previous_positive_path, sep="\t")
    previous_positive_dataset = list(previous_positive_dataset.itertuples(index=False, name=None))
    previous_negative_dataset = pd.read_csv(previous_negative_path, sep="\t")
    previous_negative_dataset = list(previous_negative_dataset.itertuples(index=False, name=None))
    valid_length = int(len(dataset) * 0.2)
    train_dataset = dataset[:len(dataset) - valid_length]
    valid_dataset = dataset[len(dataset) - valid_length:]
    positive_pairs = []
    negative_pairs = []
    # get threshold of one cause dialogues
    threshold = 0.9 #max(find_threshold_by_ratio(train_dataset, ratio=1-0.05), 0.95)
    #print(find_threshold_by_ratio(train_dataset, ratio=1-0.05))
    print(f"Threshold of one cause dialogue: {threshold}")
    for dialog in train_dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] \
                        and dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > threshold:
                    highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                    highest_idx = i

            if highest_idx != -1:
                response = dialog[-1]["text"].strip()
                positive_context = dialog[highest_idx]["text"].strip() + f" </s> <s> " + dialog[-2]["text"].strip()
                positive_pairs.append((positive_context, response, 1))
                for j in random.sample(range(0, len(dialog)), 3):
                    if j != highest_idx and "conditional_causal_score_with_last_response" in dialog[j]:
                        # and dialog[j]["conditional_causal_score_with_last_response"] < 0.5:
                        negative_context = dialog[j]["text"].strip() + f" </s> <s> " + dialog[-2]["text"].strip()
                        negative_pairs.append((negative_context, response, 0))
    print("True samples: ", len(previous_positive_dataset), len(previous_negative_dataset))
    print("New samples: ", len(positive_pairs), len(negative_pairs))
    positive_pairs += previous_positive_dataset
    positive_pairs = list(OrderedDict.fromkeys(positive_pairs))
    negative_pairs += previous_negative_dataset
    negative_pairs = list(OrderedDict.fromkeys(negative_pairs))

    print("Incremental samples: ", len(positive_pairs), len(negative_pairs))
    positive_pairs = [ele for ele in positive_pairs for i in range(len(negative_pairs) // len(positive_pairs))]
    print(len(positive_pairs), len(negative_pairs))
    extend_pairs = random.sample(positive_pairs, len(negative_pairs) - len(positive_pairs))
    positive_pairs += extend_pairs
    print(len(positive_pairs), len(negative_pairs))

    current_positive_dataset = pd.DataFrame(positive_pairs, columns=["utterance", "response", "label"])
    current_positive_dataset.to_csv(positive_save_path, index=False, sep="\t")

    current_negative_dataset = pd.DataFrame(negative_pairs, columns=["utterance", "response", "label"])
    current_negative_dataset.to_csv(negative_save_path, index=False, sep="\t")

    valid_pairs = []
    for dialog in valid_dataset:
        if len(dialog) > 2:
            highest_score = -1000
            highest_idx = -1
            for i in range(len(dialog)):
                if "conditional_causal_score_with_last_response" in dialog[i] and \
                        dialog[i]["conditional_causal_score_with_last_response"] > highest_score \
                        and dialog[i]["conditional_causal_score_with_last_response"] > threshold:
                    highest_score = dialog[i]["conditional_causal_score_with_last_response"]
                    highest_idx = i

            if highest_idx != -1:
                response = dialog[-1]["text"].strip()
                positive_context = dialog[highest_idx]["text"].strip() + f" </s> <s> " + dialog[-2]["text"].strip()
                valid_pairs.append((positive_context, response, 1))
                for j in random.sample(range(0, len(dialog)), 3):
                    if j != highest_idx and "conditional_causal_score_with_last_response" in dialog[j]:
                        negative_context = dialog[j]["text"].strip() + f" </s> <s> " + dialog[-2]["text"].strip()
                        valid_pairs.append((negative_context, response, 0))
                        break
    valid_conditional_tc_dataset = pd.DataFrame(valid_pairs, columns=["utterance", "response", "label"])
    valid_conditional_tc_dataset.to_csv(valid_save_path, index=False, sep="\t")