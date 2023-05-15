import os
from create_classifier_EM_dataset import compare_ESConv_E_datasets, create_ESConv_M_training_datasets, \
    create_ESConv_M_incremental_training_datasets, create_msc_M_incremental_training_datasets, compare_msc_E_datasets, create_msc_M_training_datasets

def ESConv_EM_train():
    print("ESConv_EM_train")
    start_step = 0
    EM_steps = 10
    for i in range(start_step, start_step+EM_steps):
        print(f"Starting iteration {i}")
        # E step
        E_step_signal = os.system(f"python run_unbalance_conditional_text_classifier_Estep.py "
                                  f"--model_name_or_path ESConv_unbalance_conditional_tc_EM{i}_model/ "
                                  f"--tokenizer_name_or_path ./roberta_base "
                                  f"--validation_file ESConv_tc_test_result/ESConv_train_causal.json "
                                  f"--output_dir ESConv_conditional_tc_EM_result --save_file ESConv_train_causal_E{i+1}.json")
        print(f"Step {i} E signal: ", E_step_signal)

        #compare_ESConv_E_datasets(previous_dataset=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_E{i}.json",
        #                          current_dataset=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_E{i+1}.json")
        #create_ESConv_M_training_datasets(load_path=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_E{i+1}.json",
        #                               positive_save_path=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_M{i+1}_positive.csv",
        #                               negative_save_path=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_M{i+1}_negative.csv",
        #                               valid_save_path=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_M{i+1}_valid.csv")
        create_ESConv_M_incremental_training_datasets(load_path=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_E{i+1}.json",
                                                      previous_positive_path=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_M{i}_positive.csv",
                                                      previous_negative_path=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_M{i}_negative.csv",
                                                      positive_save_path=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_M{i+1}_positive.csv",
                                                      negative_save_path=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_M{i+1}_negative.csv",
                                                      valid_save_path=f"ESConv_conditional_tc_EM_result/ESConv_train_causal_M{i+1}_valid.csv")

        # M step
        M_step_signal = os.system(f"python run_unbalance_text_classification.py "
                                  f"--positive_label_file ESConv_conditional_tc_EM_result/ESConv_train_causal_M{i+1}_positive.csv "
                                  f"--negative_label_file ESConv_conditional_tc_EM_result/ESConv_train_causal_M{i+1}_negative.csv "
                                  f"--validation_file ESConv_conditional_tc_EM_result/ESConv_test_conditional_tc_dataset_fromAnnotation.csv "
                                  f"--model_name_or_path ESConv_unbalance_conditional_tc_EM{i}_model/ "
                                  f"--tokenizer_name roberta_base/ --output_dir ESConv_unbalance_conditional_tc_EM{i+1}_model/ "
                                  f"--use_slow_tokenizer --num_train_epochs 1")
        print(f"Step {i} M signal: ", M_step_signal)

#ESConv_EM_train()

def msc_EM_train():
    print("msc_EM_train")
    start_step = 4
    EM_steps = 10
    for i in range(start_step, start_step+EM_steps):
        print(f"Starting iteration {i}")
        # E step
        E_step_signal = os.system(f"python run_unbalance_conditional_text_classifier_Estep.py "
                                  f"--model_name_or_path ~/da33_scratch/tao/test-project/Causal_response/msc_unbalance_conditional_tc_EM{i}_model/ "
                                  f"--tokenizer_name_or_path ./roberta_base "
                                  f"--validation_file msc_tc_test_result/msc_train_causal.json "
                                  f"--output_dir ~/da33_scratch/tao/test-project/Causal_response/msc_conditional_tc_EM_result --save_file msc_train_causal_E{i+1}.json")
        print("E step signal: ", E_step_signal)

        #compare_msc_E_datasets(previous_dataset=f"msc_conditional_tc_EM_result/msc_train_causal_E{i}.json",
        #                          current_dataset=f"msc_conditional_tc_EM_result/msc_train_causal_E{i+1}.json")
        create_msc_M_incremental_training_datasets(load_path=f"~/da33_scratch/tao/test-project/Causal_response/msc_conditional_tc_EM_result/msc_train_causal_E{i+1}.json",
                                                      previous_positive_path=f"~/da33_scratch/tao/test-project/Causal_response/msc_conditional_tc_EM_result/msc_train_causal_M{i}_positive.csv",
                                                      previous_negative_path=f"~/da33_scratch/tao/test-project/Causal_response/msc_conditional_tc_EM_result/msc_train_causal_M{i}_negative.csv",
                                                      positive_save_path=f"~/da33_scratch/tao/test-project/Causal_response/msc_conditional_tc_EM_result/msc_train_causal_M{i+1}_positive.csv",
                                                      negative_save_path=f"~/da33_scratch/tao/test-project/Causal_response/msc_conditional_tc_EM_result/msc_train_causal_M{i+1}_negative.csv",
                                                      valid_save_path=f"~/da33_scratch/tao/test-project/Causal_response/msc_conditional_tc_EM_result/msc_train_causal_M{i+1}_valid.csv")
        # M step
        M_step_signal = os.system(f"python run_unbalance_text_classification.py "
                                  f"--positive_label_file msc_conditional_tc_EM_result/msc_train_causal_M{i+1}_positive.csv "
                                  f"--negative_label_file msc_conditional_tc_EM_result/msc_train_causal_M{i+1}_negative.csv "
                                  f"--validation_file msc_conditional_tc_EM_result/msc_test_conditional_tc_dataset_fromAnnotation.csv "
                                  f"--model_name_or_path ~/da33_scratch/tao/test-project/Causal_response/msc_unbalance_conditional_tc_EM{i}_model/ "
                                  f"--tokenizer_name roberta_base/ --output_dir ~/da33_scratch/tao/test-project/Causal_response/msc_unbalance_conditional_tc_EM{i+1}_model/ "
                                  f"--use_slow_tokenizer --num_train_epochs 1")
        print("M step signal: ", M_step_signal)
msc_EM_train()