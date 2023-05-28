## Code for ReContriever (Unsupervised Dense Retrieval with Relevance-Aware Contrastive Pre-Training) ##

The repository contains the code to pre-train the ReContriever and Contriever models. 
Our code is mainly built upon the official Github Repository [Facebookresearch/Contriever](https://github.com/facebookresearch/contriever).

## Data Preprocessing ##
Please refer to the readme of [Facebookresearch/Contriever](https://github.com/facebookresearch/contriever), which provide a detailed guide for data preprocessing.

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