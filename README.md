# \<Insert Paper Title\>

Welcome to the official repository for our reproducibility research of "**FairCLIP: HArnessing Fairness in Vision-Language Learning**", an approach to tackle bias in medical data using the CLIP framework. The original FairCLIP paper introduces a method using the Sinkhorn distance in order to minimize the distance between the population distribution and the group distribution of a sensitive attribute.

The code in this repository has been used to research the reproducability of the experiments conducted in the original FairCLIP paper. Furthermore, the repository contains debugged code as well as code for our contributions. Our contributions include:
<ul>
  <li>Extensive analysis of the Sinkhorn distances.</li>
  <li><b>FairCLIP+</b>, a model that takes into account multiple sensitive attributes.</li>
  <li>A look into the generalization of FairCLIP to other datasets.</li>
</ul>


## Table of Content
- [Installation Guide](#installation-guide)
  - [Datasets](#datasets)
- [How to Run](#how-to-run)
  -  [Finetuning](#finetuning)
  - [Model Evaluation](#model-evaluation)
  - [Distance Evaluation](#distance-evaluation)
  - [Linear Probing](#linear-probing)
  - [Lambda Test](#lambda-test)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Debugged Code](#debugged-code)
- [FairCLIP+](#fairclip)
- [Sinkhorn Distance](#sinkhorn-distance)


## Installation Guide
Download the repository as a .zip or clone the repository using:
```console
foo@bar:~$ git clone git@github.com:stanfris/FairCLIP_Reproducibility.git 
```
Install the correct version of the used packages from the .yml file using the following command:
```console
foo@bar:~$ conda env create -f fairclip.yml
```
Upon installation of the environment, it can be (de)activated using:
```console
foo@bar:~$ conda activate fairclip
(fairclip) foo@bar:~$ conda deactivate fairclip
```
The environment can be deleted using:
```console
foo@bar:~$ conda remove -n fairclip --all
```
Additional packages can be installed using pip:
```console
foo@bar:~$ pip install <package>
```

### Datasets
The Harvard-FairVLMed dataset can be requested [here](https://drive.google.com/drive/folders/1bkeifigwOAfnsLvup9mJOSNeA3WsvA2l?usp=drive_link).
The FairFace dataset can be downloaded [here]()

## How to Run
Now that the environment has been correctly installed, it is time to run the code. For each of the experiments we will provide the basic scripts needed to run the code. The scripts we used to run experiments on the snellius supercomputer can be found [here](https://github.com/stanfris/FairCLIP_Reproducibility/tree/main/experiments). These scripts can easily be turned into shell scripts to allow execution on other machines.

### Finetuning
To run finetuning code, use the following script (don't forget to change the paths according to your filesystem):
```bash
DATASET_DIR=../data/Harvard-FairVLMed
RESULT_DIR=../results/
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
NUM_EPOCH=10
MODALITY_TYPE='slo_fundus'
ATTRIBUTE_TYPE=race # Options: race | gender | ethnicity | language
SUMMARIZED_NOTE_FILE=gpt-4_summarized_notes.csv
LR=1e-5
BATCH_SIZE=32
LAMBDA=1e-7
BATCH_SIZE_FAIR=32

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_FairCLIP.csv

python3 ./finetune_FairCLIP.py \
		--dataset_dir ${DATASET_DIR} \
		--result_dir ${RESULT_DIR}fairCLIP/newdistance/${ATTRIBUTE_TYPE}/glaucoma_FairCLIP_${MODEL_ARCH}_${BATCH_SIZE} \
		--lr ${LR} \
		--batch_size ${BATCH_SIZE} \
		--perf_file ${PERF_FILE} \
		--model_arch ${MODEL_ARCH} \
		--attribute ${ATTRIBUTE_TYPE} \
		--batchsize_fairloss ${BATCH_SIZE_FAIR} \
		--lambda_fairloss ${LAMBDA} \
		--summarized_note_file ${SUMMARIZED_NOTE_FILE} \
```

### Model Evaluation
To run model evaluation code, use the following script (again, don't forget to change paths, especially the path to the model checkpoint):
```bash
DATASET_DIR=../data/Harvard-FairVLMed
RESULT_DIR=../results
MODEL_ARCH=vit-b16  # Options: vit-b16 | vit-l14
MODALITY_TYPE='slo_fundus'
LR=1e-5
BATCH_SIZE=32
ATTRIBUTE_TYPE=language # Options: race | gender | ethnicity | language

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}_CLIP_eval.csv

python3 ./evaluate_CLIP.py \
		--dataset_dir ${DATASET_DIR} \
		--result_dir ${RESULT_DIR}/CLIP_finetuning/ \
		--lr ${LR} \
		--perf_file ${PERF_FILE} \
		--model_arch ${MODEL_ARCH} \
		--pretrained_weights ${RESULT_DIR}/CLIP_finetuning/glaucoma_CLIP_${MODEL_ARCH}313_seed313_auc0.6833/clip.pth
```

### Distance Evaluation
To run distance evaluation code, use the following script for CLIP:
```bash
DATASET_DIR=../data/Harvard-FairVLMed
RESULT_DIR=../results
MODEL_ARCH=vit-b16  # Options: vit-b16 | vit-l14
MODALITY_TYPE='slo_fundus'
LR=1e-5
BATCH_SIZE=32
ATTRIBUTE_TYPE=ethnicity # Options: race | gender | ethnicity | language

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_FairerCLIP_eval_distance.csv

python3 ./evaluate_CLIP_with_distance.py \
		--dataset_dir ${DATASET_DIR} \
		--result_dir ${RESULT_DIR}/CLIP_finetuning \
		--lr ${LR} \
		--perf_file ${PERF_FILE} \
		--model_arch ${MODEL_ARCH} \
		--pretrained_weights ${RESULT_DIR}/CLIP_finetuning/glaucoma_CLIP_${MODEL_ARCH}313_seed313_auc0.6833/clip.pth
```
For FairCLIP and FairCLIP+, change the model checkpoint and hyperparameters according to the model.

### Linear Probing
To run the linear probing code, use the following script:
```bash
DATA_DIR=../data/Harvard-FairVLMed
PRETRAIN_CHKPT=ViT-L/14
CHKPT_NAME=CLIP_seed3231
EXP_NAME=../results/linear_probing

CFG_PATH=../LAVIS/lavis/projects/blip2/train/pretrain_stage1.yaml
FEATS_TYPE=image # [image, multimodal]
MODEL_TYPE=clip # [clip, blip2]
SUMMARY_TYPE=gpt-4
BATCH_SIZE=512
EPOCHS=500
LR=0.1
WDECAY=0.
SEED=3231

cd ../mae
python3 main_linprobe.py \
            --model_type ${MODEL_TYPE} \
            --vl_feats_type ${FEATS_TYPE} \
            --vision_encoder_weights clip \
            --summary_type ${SUMMARY_TYPE} \
            --batch_size ${BATCH_SIZE} \
            --model vit_large_patch16 \
            --cls_token \
	          --finetune ${PRETRAIN_CHKPT} \
            --save_checkpoint_name ${CHKPT_NAME} \
            --epochs ${EPOCHS} \
            --blr ${LR} \
            --weight_decay ${WDECAY} \
            --data_path ${DATA_DIR} \
            --output_dir ${EXP_NAME} \
            --log_dir ${EXP_NAME} \
            --nb_classes 2 \
            --blip_feats_select avgpool \
            --cfg-path ${CFG_PATH} \
            --seed ${SEED}
```
Change the ```CHKPNT_NAME``` and ```SEED``` according to your pre-trained models.

### Lambda Test
In order to run the Lambda test, use the following script:
```bash

```

### Hyperparameter Tuning
In order to run the hyperparameter tuning, use the following script:
```bash

```

## Debugged Code
In the original code found in the [FairCLIP Repository](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP) contains some bugs which we have fixed.

In the original code, during training the model is evaluated on the test set, which should have been the validation set. We have fixed this by changing
```python
for batch in test_dataloader:
```
to
```python
 for batch in val_dataloader:
```

Secondly, the model is never changed between train and evaluation mode. This is simply fixed by adding the lines
```python
model.train()
model.eval()
```
at the right places.

Thirdly, the original code weighs the sinkhorn loss with a Lambda value. If this Lambda value gets too larger (>= 1e-5), the code crashes. This issue arises since the model is fp16 when training on a GPU and fp32 when training on CPU. The model can work with all lambda values by changing
```python
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model)
```
to 
```python
model.float()
```
and changing
```python
if device == "cpu":
    optimizer.step()
else : 
    convert_models_to_fp32(model)
    optimizer.step()
    clip.model.convert_weights(model)
```
to 
```python
optimizer.step()
```

Finally, we found the code to incorrectly use ```torch.backends.cudnn.deterministic``` and ```torch.backends.cudnn.benchmark```.
In the original code, ```torch.backends.cudnn.deterministc``` is set to ```False``` which makes it hard for the experiments to be exactly reproducible. ```torch.backends.cudnn.benchmark``` was set ```True``` which we changed to ```False``` since it would otherwise overwrite ```torch.backends.cudnn.deterministic```.

## FairCLIP+


## Sinkhorn Distance

