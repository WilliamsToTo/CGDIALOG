# <img src="img/logo.jpg" width="8%" alt="" align=center /> CGDIALOG (Under construction)

This repository contains the dataset and the pytorch implementations of the models from the paper [Less is More: Mitigate Spurious Correlations for Open-Domain Dialogue Response Generation Models by Causal Discovery]().

## Dataset
We leverage two public dialogue corpora (ESConv and MSC) to construct a corpus annotated with direct causes of responses called CGDIALOG.
The original annotated dataset can be found in `datasets/CGDIALOG`.
### Data Format
The dataset format is like the following.
```bash
{
"HITId": "3RIHDBQ1OL0HF1CNXY41R5Y9Q7OHMC",
"WorkerId": "ARH3NPT7GUFQ6",
"history": [ # dialogue history
    "seeker: Hi!",
    "supporter: Hello, how are you doing today?",
    "seeker: Not so good. I have conspiracy theorist as a friend who is now mad at me because I told her to pull up her mask while talking to me.",
    "seeker: We have been friends for 13 years",
    "seeker: I am hurt and confused that she still thinks this is a game.",
    "seeker: She thought Corona was fake until someone we know caught it.",
    "seeker: It is like she is mad she was wrong and is taking it out and lashing out at those who have been trying to persuade her the whole time...",
    "seeker: what do you think?",
    ""
],
"response": "It sounds like you care a lot about your friend and others. How old is your friend?",
"entities": [ # direct causes of responses that are annotated by workers.
    "Not so good. I have conspiracy theorist as a friend who is now mad at me because I told her to pull up her mask while talking to me.",
    "I am hurt and confused that she still thinks this is a game.",
    "It is like she is mad she was wrong and is taking it out and lashing out at those who have been trying to persuade her the whole time..."
]
}
```

## Setup:
The code is based on PyTorch and HuggingFace `transformers`.
```bash 
cd CGDIALOG
conda create --prefix env/ python=3.6
conda activate env/
pip install -r requirements.txt 
```

## How to use our models
You can download our models, then put to `models` folder:
- [ESConv_classifier](https://drive.google.com/drive/folders/109rlsiHP0o2-w2Dy0DVqwWPa6joYTNBo?usp=sharing)
- [MSC_classifier](https://drive.google.com/drive/folders/1_t9mPzQQFHhcbp1azQ11e6cse_Tz9vR0?usp=sharing)
- [ESConv_generator](https://drive.google.com/drive/folders/1RWvbklirSxaHjofJyIxYMu42Ge444WDv?usp=sharing)
- [MSC_generator](https://drive.google.com/drive/folders/1tBAvLN9W_dxQqNcAAxdblTUtVbFjaJUN?usp=sharing)
- [blenderbot](https://drive.google.com/drive/folders/1vkslYrL0epbLeoP131Wh3a2BJaBuvHd3?usp=sharing)
- [roberta](https://drive.google.com/drive/folders/1rF9fx3cFZ3VG5huIvj3hXLTKGt2dRFYj?usp=sharing)


### Generate ESConv responses
```bash
python test_causal_response_generator.py --validation_file datasets/ESConv/test_dataset.json --model_name_or_path models/ESConv_causal_generator_model/ 
--tokenizer_name models/blenderbot_400M_distill/ --twoCondition_tc_model_name_or_path models/ESConv_classifier/ --tc_tokenizer_name roberta_base/ --output_dir outputs/ESConv_causal_generator_test_result
```


### Generate MSC responses
```bash
python test_causal_response_generator.py --validation_file datasets/msc/msc_dialogue/session_4/test_dataset.json --model_name_or_path models/msc_causal_generator_model/ 
--tokenizer_name models/blenderbot_400M_distill/ --twoCondition_tc_model_name_or_path models/msc_classifier/ --tc_tokenizer_name roberta_base/ --output_dir outputs/msc_causal_generator_test_result
```

### Select final response
```bash
from select_best_response_ourModel import select_highScore_response
select_highScore_response(ourModel_file="outputs/ESConv_causal_generator_test_result", save_file="outputs/ESConv_test_result_in_testset_highestScore.json")
select_highScore_response(ourModel_file="outputs/msc_causal_generator_test_result", save_file="outputs/msc_test_result_in_testset_highestScore.json")
```