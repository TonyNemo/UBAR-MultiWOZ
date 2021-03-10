# UBAR
This is the code and data for the AAAI 2021 paper "UBAR: Towards Fully End-to-End Task-Oriented Dialog System with GPT-2". [arxiv](https://arxiv.org/pdf/2012.03539.pdf)


## Abstract
This paper presents our task-oriented dialog system UBAR
which models task-oriented dialogs on a dialog session level.
Specifically, UBAR is acquired by fine-tuning the large pretrained unidirectional language model GPT-2 on the sequence
of the entire dialog session which is composed of user utterance, belief state, database result, system act, and system response of every dialog turn. Additionally, UBAR is evaluated
in a more realistic setting, where its dialog context has access
to user utterances and all content it generated such as belief
states, system acts, and system responses. Experimental results on the MultiWOZ datasets show that UBAR achieves
state-of-the-art performances in multiple settings, improving
the combined score of response generation, policy optimization, and end-to-end modeling by 4.7, 3.5, and 9.4 points respectively. Thorough analyses demonstrate that the sessionlevel training sequence formulation and the generated dialog
context are essential for UBAR to operate as a fully end-toend task-oriented dialog system in real life. We also examine
the transfer ability of UBAR to new domains with limited
data and provide visualization and a case study to illustrate
the advantages of UBAR in modeling on a dialog session
level.

We also provide technical appendix in ```Appendix.pdf```.
## Requirements
- CUDA 10.1
- Python 3.6
- PyTorch 1.5
- spaCy
- transformers 2.11

We use the tokenization tool in SpaCy which can be installed through:
```
python -m spacy download en_core_web_sm
```


## Data Preprocessing
The original data files are put under data/multi-woz, which includes:
- data.json: The orignal MultiWOZ 2.0 data released by researchers in University of Cambridge available [here](https://github.com/budzianowski/multiwoz)
- annotated_user_da_with_span_full.json: A preprocessed and fully annotated version of MultiWOZ 2.0 data released by developers of Convlab available [here](https://github.com/ConvLab/ConvLab/tree/master/data/multiwoz/annotation)

We mainly show the implementation process on MultiWOZ 2.0. If you want to implement later versions you'll have to get the dataset yourself and change the corresponding paths in the scripts or use the scripts postfixed with "21". 

Get ready for preprocessng:
```
python data_analysis.py
python preprocess.py
```
## Training
Our implementation supports training on CPU or a single GPU.
```
python train.py -mode train -cfg gpt_path=distilgpt2 lr=1e-4 warmup_steps=2000 gradient_accumulation_steps=16 batch_size=2 epoch_num=60 exp_no=best_model
```

Our best model is saved in "experiments/all_0729_sd11_lr0.0001_bs2_ga16/epoch43_trloss0.56_gpt2", which is released at 
- [Baidu Wangpan, Access code：miaa ](https://pan.baidu.com/s/1GXnsGgwp2j66TqyxkOSbgA)
- [Google drive](https://drive.google.com/file/d/1uZOhZl3oKXf66DCCZIE2O7Aax3OJzfvl/view?usp=sharing)

### Train UBAR-DST
```
python train_DST.py -mode train -cfg lr=6e-5 epoch_num=50 gradient_accumulation_steps=12 warmup_steps=0  use_true_prev_bspn=False use_true_prev_aspn=True use_true_pv_resp=True
```

## Evaluation

### Response Generation
```
path='YOUR_EXPERIMENT_PATH'
python train.py -mode test -cfg eval_load_path=$path use_true_prev_bspn=True use_true_prev_aspn=True use_true_db_pointer=True use_true_prev_resp=False use_true_curr_bspn=True use_true_curr_aspn=True use_all_previous_context=True cuda_device=0
```


### Policy Optimization (Act and Response Generation)

```
path='YOUR_EXPERIMENT_PATH'
python train.py -mode test -cfg eval_load_path=$path use_true_prev_bspn=True use_true_prev_aspn=False use_true_db_pointer=True use_true_prev_resp=False use_true_curr_bspn=True use_true_curr_aspn=False use_all_previous_context=True cuda_device=0
```

### End-to-end Modeling (Belief state, Act and Response Generation)
```
path='YOUR_EXPERIMENT_PATH'
python train.py -mode test -cfg eval_load_path=$path use_true_prev_bspn=False use_true_prev_aspn=False use_true_db_pointer=False use_true_prev_resp=False use_true_curr_bspn=False use_true_curr_aspn=False use_all_previous_context=True cuda_device=0
```
Important note: To fairly compare with other methods in the end-to-end setting, the groundtruth belief states are used to query for DB results. To use generated belief states, set use_true_bspn_for_ctr_eval to False.

### Dialog State Tracking
```
path='YOUR_EXPERIMENT_PATH'
python train_DST.py -mode test -cfg eval_load_path=$path use_true_prev_bspn=False use_true_prev_aspn=True use_true_prev_resp=True use_true_db_pointer=False
```


### Evaluation settings
- use_true_prev_bspn: use the ground truth previous turns' belief span as context.
- use_true_prev_aspn: use the ground truth previous turns' action span as context.
- use_true_db_pointer: use the ground truth database search results as context.
- use_true_prev_resp: use the ground truth previous turns' response as context.
- use_true_curr_bspn: use the ground truth current turn's belief span.
- use_true_curr_aspn: use the ground truth current turn's belief span.
- use_all_previous_context: use all previous turns as context. 
- use_true_bspn_for_ctr_eval: use the ground truth belief span to query DB results.




## Acknowledgement
This code is adapted and modified upon the released code [[github]]() of previous AAAI 2020 paper "Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses under the Same Context"[[paper link]](https://arxiv.org/abs/1911.10484). 

We appreciate their open-sourcing such high-quality code, which is very helpful to our research. 

And of course thanks HuggingFace for their wonderful transformers implementation.
