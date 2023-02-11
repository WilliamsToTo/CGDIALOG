# <img src="img/logo.jpg" width="8%" alt="" align=center /> CGDIALOG (Under construction)

This repository contains the dataset and the pytorch implementations of the models from the paper [Less is More: Mitigate Spurious Correlations for Open-Domain Dialogue Response Generation Models by Causal Discovery]().

## Dataset
We leverage two public dialogue corpora (ESConv and MSC) to construct a corpus annotated with direct causes of responses called CGDIALOG.
The original annotated dataset can be found in `datasets/CGDIALOG`.
### Data Format

## Setup:
The code is based on PyTorch and HuggingFace `transformers`.
```bash 
pip install -r requirements.txt 
```

## How to use our models
You can download our models, then put to `models` folder:
- [ESConv_classifier](https://drive.google.com/drive/folders/109rlsiHP0o2-w2Dy0DVqwWPa6joYTNBo?usp=sharing)
- [MSC_classifier](https://drive.google.com/drive/folders/1_t9mPzQQFHhcbp1azQ11e6cse_Tz9vR0?usp=sharing)
- [ESConv_generator](https://drive.google.com/drive/folders/1RWvbklirSxaHjofJyIxYMu42Ge444WDv?usp=sharing)
- [MSC_generator](https://drive.google.com/drive/folders/1tBAvLN9W_dxQqNcAAxdblTUtVbFjaJUN?usp=sharing)


### Generate responses
```bash
python 
```