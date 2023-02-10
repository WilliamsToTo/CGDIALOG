import copy
import json
import random

#ourModel_file = "msc_allTwoCondition_unbalance_generator_test_result/test_result_in_testset.json"

def select_human_response(ourModel_file, save_file):
    print("select_human_response")
    ourModel_result = json.load(open(ourModel_file, "r"))
    print(len(ourModel_result))
    ourModel_select_result = []
    for dialog in ourModel_result:
        one_dialogue = {}
        one_dialogue["dialog_history"] = dialog["dialog_history"]
        selected_response = dialog["dialog_history"][-1]["text"]

        one_dialogue["generated_response"] = [{"generated_response": selected_response}]
        ourModel_select_result.append(one_dialogue)
    json.dump(ourModel_select_result, open(save_file, "w+"), indent=4)
save_file = "msc_EM1_causal_generator_test_result/test_result_in_testset_human.json"
select_human_response(ourModel_file="msc_EM1_causal_generator_test_result/test_result_in_testset.json", save_file=save_file)


def select_highScore_response(ourModel_file, save_file):
    print("select_highScore_response")
    ourModel_result = json.load(open(ourModel_file, "r"))
    print(len(ourModel_result))
    ourModel_select_result = []
    for dialog in ourModel_result:
        one_dialogue = {}
        one_dialogue["dialog_history"] = dialog["dialog_history"]
        selected_response = ""
        highest_conditional_score = -100
        for response in dialog["generated_response"]:
            if response["conditional_dependence"] > highest_conditional_score:
                highest_conditional_score = response["conditional_dependence"]
                selected_response = response["generated_response"]
        one_dialogue["generated_response"] = [{"generated_response": selected_response}]
        ourModel_select_result.append(one_dialogue)
    json.dump(ourModel_select_result, open(save_file, "w+"), indent=4)

#save_file = "msc_EM1_best_beam_causal_generator_test_result/test_result_in_testset_highestScore.json"
#select_highScore_response(ourModel_file="msc_EM1_best_beam_causal_generator_test_result/test_result_in_testset.json", save_file=save_file)

def select_ppl_response(ourModel_file, save_file):
    print("select_ppl_response")
    ourModel_result = json.load(open(ourModel_file, "r"))
    print(len(ourModel_result))
    ourModel_select_result = []
    for dialog in ourModel_result:
        one_dialogue = {}
        one_dialogue["dialog_history"] = dialog["dialog_history"]
        selected_response = ""
        lowest_ppl = 100
        for response in dialog["generated_response"]:
            if response["ppl"] < lowest_ppl:
                lowest_ppl = response["ppl"]
                selected_response = response["generated_response"]
        one_dialogue["generated_response"] = [{"generated_response": selected_response}]
        ourModel_select_result.append(one_dialogue)
    json.dump(ourModel_select_result, open(save_file, "w+"), indent=4)

#save_file = "msc_EM1_causal_generator_test_result/test_result_in_testset_lowPPL.json"
#select_ppl_response(ourModel_file = "msc_EM1_causal_generator_test_result/test_result_in_testset.json", save_file=save_file)

def select_fromLast_response(ourModel_file, save_file):
    print("select_fromLast_response")
    # select r|u_{t-2}, t_{t-1}, if r is good enough
    ourModel_result = json.load(open(ourModel_file, "r"))
    print(len(ourModel_result))
    ourModel_select_result = []
    for dialog in ourModel_result:
        one_dialogue = {}
        one_dialogue["dialog_history"] = dialog["dialog_history"]
        selected_response = ""
        for response in reversed(dialog["generated_response"]):
            if response["conditional_dependence"] > 0.9:
                selected_response = response["generated_response"]
                break

        if len(selected_response) < 1:
            highest_conditional_score = -100
            for response in dialog["generated_response"]:
                if response["conditional_dependence"] > highest_conditional_score:
                    highest_conditional_score = response["conditional_dependence"]
                    selected_response = response["generated_response"]
        one_dialogue["generated_response"] = [{"generated_response": selected_response}]
        ourModel_select_result.append(one_dialogue)
    json.dump(ourModel_select_result, open(save_file, "w+"), indent=4)

#save_file = "ESConv_EM1_causal_generator_test_result/test_result_in_testset_fromLast.json"
#select_fromLast_response(ourModel_file="ESConv_EM1_causal_generator_test_result/test_result_in_testset.json", save_file=save_file)

def select_ut_2_ut_1_response(ourModel_file, save_file):
    print("select_ut_2_ut_1_response")
    ourModel_result = json.load(open(ourModel_file, "r"))
    print(len(ourModel_result))
    ourModel_select_result = []
    for dialog in ourModel_result:
        one_dialogue = {}
        one_dialogue["dialog_history"] = dialog["dialog_history"]
        selected_response = dialog["generated_response"][-1]["generated_response"]
        one_dialogue["generated_response"] = [{"generated_response": selected_response}]
        ourModel_select_result.append(one_dialogue)
    json.dump(ourModel_select_result, open(save_file, "w+"), indent=4)

#save_file = "msc_allCondition_causal_inference_test_result/test_result_in_testset_ut_2_ut_1.json"
#select_ut_2_ut_1_response(ourModel_file = "msc_allCondition_causal_inference_test_result/test_result_in_testset.json", save_file=save_file)


def select_random_response(ourModel_file, save_file):
    print("select_random_response")
    ourModel_result = json.load(open(ourModel_file, "r"))
    print(len(ourModel_result))
    ourModel_select_result = []
    for dialog in ourModel_result:
        one_dialogue = {}
        one_dialogue["dialog_history"] = dialog["dialog_history"]
        selected_response = random.sample(dialog["generated_response"], k=1)[0]["generated_response"]
        one_dialogue["generated_response"] = [{"generated_response": selected_response}]
        ourModel_select_result.append(one_dialogue)
    json.dump(ourModel_select_result, open(save_file, "w+"), indent=4)

#save_file="msc_EM1_causal_generator_test_result/test_result_in_testset_random.json"
#select_random_response(ourModel_file="msc_EM1_causal_generator_test_result/test_result_in_testset.json", save_file=save_file)

def select_dependent_highScore_response(ourModel_file, save_file):
    print("select_dependent_highScore_response")
    ourModel_result = json.load(open(ourModel_file, "r"))
    ourModel_select_result = []
    for dialog in ourModel_result:
        one_dialogue = {}
        one_dialogue["dialog_history"] = dialog["dialog_history"]
        selected_response = ""
        highest_score = -100
        for response in dialog["generated_response"]:
            if response["dependence"] > highest_score:
                highest_score = response["dependence"]
                selected_response = response["generated_response"]
        print(selected_response)
        one_dialogue["generated_response"] = [{"generated_response": selected_response}]
        ourModel_select_result.append(one_dialogue)
    json.dump(ourModel_select_result, open(save_file, "w+"), indent=4)

#select_dependent_highScore_response(ourModel_file="msc_EM1_causal_generator_test_result/test_result_in_testset_dependenceScore.json",
#                                    save_file="msc_EM1_causal_generator_test_result/test_result_in_testset_dependenceHighestScore.json")


def select_PC_highScore_response(ourModel_file, save_file):
    print("select_PC_highScore_response")
    ourModel_result = json.load(open(ourModel_file, "r"))
    ourModel_select_result = []
    for dialog in ourModel_result:
        one_dialogue = {}
        one_dialogue["dialog_history"] = dialog["dialog_history"]
        selected_response = ""
        highest_conditional_score = -100
        for response in dialog["twoCondition_generated_response"]:
            if response["dependence"] > 0.9 and response["conditional_dependence"] > 0.9 and response["conditional_dependence"] > highest_conditional_score:
                highest_conditional_score = response["conditional_dependence"]
                selected_response = response["generated_response"]
        # if selected_response is still empty
        if len(selected_response) < 1:
            highest_score = -100
            for response in dialog["oneCondition_generated_response"]:
                if response["dependence"] > highest_score:
                    highest_score = response["dependence"]
                    selected_response = response["generated_response"]
            print(selected_response)
        one_dialogue["generated_response"] = [{"generated_response": selected_response}]
        ourModel_select_result.append(one_dialogue)
    json.dump(ourModel_select_result, open(save_file, "w+"), indent=4)

#save_file = "msc_PC_causal_generator_test_result/test_result_in_testset_highestScore.json"
#select_PC_highScore_response(ourModel_file = "msc_PC_causal_generator_test_result/test_result_in_testset.json", save_file=save_file)

def select_PC_fromLast_response(ourModel_file, save_file):
    print("select_PC_fromLast_response")
    # select r|u_{t-2}, t_{t-1}, if r is good enough
    ourModel_result = json.load(open(ourModel_file, "r"))
    ourModel_select_result = []
    for dialog in ourModel_result:
        one_dialogue = {}
        one_dialogue["dialog_history"] = dialog["dialog_history"]
        selected_response = ""
        for response in reversed(dialog["twoCondition_generated_response"]):
            if response["dependence"] > 0.9 and response["conditional_dependence"] > 0.9:
                selected_response = response["generated_response"]
                break
        if len(selected_response) < 1:
            highest_score = -100
            for response in dialog["oneCondition_generated_response"]:
                if response["dependence"] > highest_score:
                    highest_score = response["dependence"]
                    selected_response = response["generated_response"]
            print(selected_response)
        one_dialogue["generated_response"] = [{"generated_response": selected_response}]
        ourModel_select_result.append(one_dialogue)
    json.dump(ourModel_select_result, open(save_file, "w+"), indent=4)

#save_file = "msc_PC_causal_generator_test_result/test_result_in_testset_fromLast.json"
#select_PC_fromLast_response(ourModel_file="msc_PC_causal_generator_test_result/test_result_in_testset.json", save_file=save_file)

def select_equation2_highScore_response(ourModel_file, save_file):
    print("select_equation2_highScore_response")
    ourModel_result = json.load(open(ourModel_file, "r"))
    ourModel_select_result = []
    for dialog in ourModel_result:
        one_dialogue = {}
        one_dialogue["dialog_history"] = dialog["dialog_history"]
        selected_response = ""
        highest_score = -100
        for response in dialog["generated_response"]:
            if response["causal_score"] > highest_score:
                highest_score = response["causal_score"]
                selected_response = response["generated_response"]
        if highest_score > 5:
            one_dialogue["generated_response"] = [{"generated_response": selected_response}]
        else:
            print(selected_response)
        ourModel_select_result.append(one_dialogue)
    json.dump(ourModel_select_result, open(save_file, "w+"), indent=4)
#select_equation2_highScore_response(ourModel_file="ESConv_equ2_causal_generator_test_result/test_result_in_testset.json",
#                                    save_file="ESConv_equ2_causal_generator_test_result/test_result_in_testset_highestScore.json")
#select_equation2_highScore_response(ourModel_file="msc_equ2_causal_generator_test_result/test_result_in_testset.json",
#                                    save_file="msc_equ2_causal_generator_test_result/test_result_in_testset_highestScore.json")