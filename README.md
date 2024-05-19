## ReContriever: Unsupervised Dense Retrieval with Relevance-Aware Contrastive Pre-Training ##

## Introduction ##
This is the official code for ACL 2023 Findings paper "Unsupervised Dense Retrieval with Relevance-Aware Contrastive Pre-Training".
Our code is mainly built upon the official Github Repository of [Facebookresearch/Contriever](https://github.com/facebookresearch/contriever).

## Python Env ##
```
conda create -n recontriever python=3.9
conda activate recontriever
pip install -r requirements.txt
```

## Pretrained Models ##
| Models |  Link  |
|---|---|
|Contriever_Reproduced  | [Yibin-Lei/Contriever-Reproduced](https://huggingface.co/Yibin-Lei/Contriever-Reproduced) | 
|ReContriever | [Yibin-Lei/ReContriever](https://huggingface.co/Yibin-Lei/ReContriever) | 

You can use them with:
```
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Yibin-Lei/Contriever-Reproduced")
model = AutoModel.from_pretrained("Yibin-Lei/Contriever-Reproduced")
```


## Data Preprocessing ##
Please refer to the readme of [Facebookresearch/Contriever](https://github.com/facebookresearch/contriever), which provides a detailed guide for data preprocessing.

## Pretraining ##
We provide the scripts to pre-train ReContriever and Contriever in *./pretrain_scripts* with 16 A100 GPUs.

Our one-document-multiple-pair strategy is implemented in *./src/data.py"*.

Our relevance-aware contrastive loss is implemented in *./src/releance_aware.py*

## Evaluation
For BEIR evaluation, simply run 
```bash
python eval_beir.py --model_name_or_path $your_model_path$ --dataset $data_name$
```

For open-domain QA retrieval tasks, we use the evaluation scripts provided by the [oriram/spider](https://github.com/oriram/spider).
