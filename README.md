# On the reproducibility of "FairCLIP: Harnessing Fairness in Vision-Language Learning"

Welcome to the official repository for our reproducibility research of "**FairCLIP: Harnessing Fairness in Vision-Language Learning**", an approach to tackle bias in medical data using the CLIP framework. The original FairCLIP paper introduces a method using the Sinkhorn distance in order to minimize the distance between the population distribution and the group distribution of a sensitive attribute.

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
- [Sinkhorn Distance](#sinkhorn-distance)
- [FairCLIP+](#fairclip)
- [Debugged Code (For Submission)](#debugged-code-for-submission)
- [Acknowledgements](#acknowledgements)


## Installation Guide
Download the repository as a ```.zip``` or clone the repository.
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
The Harvard-FairVLMed dataset can be requested [here](https://drive.google.com/drive/folders/1bkeifigwOAfnsLvup9mJOSNeA3WsvA2l?usp=drive_link).<br>
The FairFace dataset can be downloaded [here](https://github.com/joojs/fairface).
Both datasets should be located in the data folder, such that this folder has the following structure:
```
data
├──Harvard-FairVLMed
|  ├── data_summary.csv
|  ├── gpt-4_summarized_notes.csv
|  ├── original_notes.csv
|  ├── Test
|  ├── Training
|  └── Validation
├──fairface
   ├── fairface_label_train.csv
   ├── ...
```

## How to Run
Now that the environment has been correctly installed, it is time to run the code. For each of the experiments we will provide the basic scripts needed to run the code. The scripts we used to run experiments cna be found in the experiments folder. These scripts can easily be turned into shell scripts to allow execution on other machines (don't forget to change all paths according to your filenames and filesystem).

### Finetuning
To run finetuning code, use the following script:
```bash
cd ./FairCLIP/

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
To run model evaluation code, use the following script:
```bash
cd ./FairCLIP/

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
To run distance evaluation code, use the following script for CLIP-FT:
```bash
cd ./experiments/distance/

DATASET_DIR=../../data/Harvard-FairVLMed
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
BATCH_SIZE=32
SEED=42
CHECKPOINT=../../results/(path to model architecture)/clip.pth
RESULT_DIR=../../results/(path to model architecture)
OUT=distances.pickle

srun python3 distance_test.py --seed ${SEED} --batch_size ${BATCH_SIZE} --model_arch ${MODEL_ARCH} --dataset_dir ${DATASET_DIR} --checkpoint ${CHECKPOINT} --out ${OUT} --results_dir ${RESULT_DIR}
```
For FairCLIP and FairCLIP+, change the model CHECKPOINT and RESULT_DIR according to where the model parameters are stored

### Linear Probing
To run the linear probing code, use the following script:
```bash
cd ../mae

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

The results for linear probing are saved in `EXP_NAME/CHKPT_NAME.pickle`. The results can be parsed into a single csv file compatible with the output generated from the finetuning of the CLIP models, using the file `convert_linear_probing_results.py` in `src`. You can use the following example code:

```bash
BASE_DIR=../results/linear_probing
FILES=(CLIP_FT_seed(seed 1).pickle CLIP_FT_seed2859.pickle CLIP_FT_seed(seed 2).pickle CLIP_seed(seed 3).pickle CLIP_seed(seed 4).pickle CLIP_seed(seed 5).pickle)
OUTPUT_FILE=../results/linear_probing/lp_results_double_check.csv

python3 convert_linear_probing_results.py --base_dir ${BASE_DIR} --files "${FILES[@]}" --out ${OUTPUT_FILE}
```



### Lambda Test
In order to run the Lambda test, use the following script:
```bash
DATASET_DIR=../data/Harvard-FairVLMed
RESULT_DIR=.
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
NUM_EPOCH=10
MODALITY_TYPE='slo_fundus'
ATTRIBUTE_TYPE=race # Options: race | gender | ethnicity | language
SUMMARIZED_NOTE_FILE=gpt-4_summarized_notes.csv
LR=1e-5
BATCH_SIZE=32
BATCH_SIZE_FAIR=32

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_FairCLIP_lambdas.csv

for i in {0..9}; do
    LAMBDA=$(bc -l <<< "10^(-$i)")
    for _ in {1..3}; do
        python3 lambda_experiment.py \
		    --dataset_dir ${DATASET_DIR} \
		    --result_dir ${RESULT_DIR}/results_lambda/ \
		    --lr ${LR} \
		    --num_epochs ${NUM_EPOCH} \
		    --batch_size ${BATCH_SIZE} \
		    --perf_file ${PERF_FILE} \
		    --model_arch ${MODEL_ARCH} \
		    --attribute ${ATTRIBUTE_TYPE} \
		    --batchsize_fairloss ${BATCH_SIZE_FAIR} \
		    --lambda_fairloss ${LAMBDA} \
		    --summarized_note_file ${SUMMARIZED_NOTE_FILE}
        wait $!
    done
done
```

### Hyperparameter Tuning
In order to run the hyperparameter tuning, use the following script:
```bash
DATASET_DIR=../data/Harvard-FairVLMed
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
NUM_EPOCH=10
MODALITY_TYPE='slo_fundus'
SUMMARIZED_NOTE_FILE=gpt-4_summarized_notes.csv
BATCH_SIZE=32
BATCH_SIZE_FAIR=32

python3 ./HPO_FairerCLIP.py \
		--dataset_dir ${DATASET_DIR} \
                --num_epochs ${NUM_EPOCH} \
		--batch_size ${BATCH_SIZE} \
		--model_arch ${MODEL_ARCH} \
		--batchsize_fairloss ${BATCH_SIZE_FAIR} \
		--summarized_note_file ${SUMMARIZED_NOTE_FILE}
```

## Sinkhorn Distance
The Sinkhorn distance has been adapted from the [geomloss](https://www.kernel-operations.io/geomloss/) library as follows:
```python
from geomloss import SamplesLoss

loss_for_FairCLIP = SamplesLoss(loss="sinkhorn", p=2, blur=args.sinkhorn_blur)

# Given (n, 1) and (m, 1) tensors x and y, compute
# loss = loss_for_FairCLIP(x, y)
```

## FairCLIP+
FairCLIP+ is our extension on FairCLIP to see what the effect is of training on multiple sensitive attributes instead of only one. These attributes can be weighted using a weightlist that can be passed as a command-line flag:
```python
parser.add_argument(
  "--weightslist",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=float,
  default=[0.25, 0.25, 0.25, 0.25],  # default if nothing is provided
)
```

In order for this to work, we have altered the code for calculating the loss. We loop over all ```group_dataloader``` and take into account all dataloaders for which ```weightslist[attributeid] != 0```. The total Sinkhorn distance for the complete group is then weighted using weightslist as follows:
```python
total_loss = total_loss + loss(correlations_with_batch[:, None],
                               correlations_with_group[:, None])
total_sinkhorn_loss += weightslist[attributeid]*total_loss
```

This loss is then weighted by ```args.lambda_fairloss```, added to the ```total_loss```, and scaled by ```args.accum_iter```:
```python
total_loss = (loss_img(logits_per_image, ground_truth) +
	      loss_txt(logits_per_text, ground_truth))/2

total_sinkhorn_loss = loss_fairer_CLIP(all_attribute_dataloaders, loss_for_FairCLIP, logits_per_image, logits_per_text, model, device, args.weightslist)

total_loss += args.lambda_fairloss * total_sinkhorn_loss

total_loss /= args.accum_iter
```

## Debugged Code (For Submission)
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
In the original code, ```torch.backends.cudnn.deterministc``` is set to ```False``` which makes it hard for the experiments to be exactly reproducible. ```torch.backends.cudnn.benchmark``` was set to ```True``` which we changed to ```False``` since it would otherwise overwrite ```torch.backends.cudnn.deterministic```.

# Acknowledgements
The code in this repository is mostly based on the official code for [FairCLIP](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP).
