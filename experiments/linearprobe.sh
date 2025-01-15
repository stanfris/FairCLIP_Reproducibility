#!/bin/bash

# directories
DATA_DIR=../data/Harvard-FairVLMed
PRETRAIN_CHKPT=../results/glaucoma_CLIP_vit-l14_seed1542_auc0.6424/clip_ep002.pth
EXP_NAME=tmp_linprobe

# hyperparameters
CFG_PATH=../LAVIS/lavis/projects/blip2/train/pretrain_stage1.yaml
FEATS_TYPE=image # [image, multimodal]
MODEL_TYPE=clip # [clip, blip2]
SUMMARY_TYPE=gpt-4
BATCH_SIZE=512
EPOCHS=3
LR=0.1
WDECAY=0.0

# Run your code
# cd ../mae
python3 main_linprobe.py \
            --model_type ${MODEL_TYPE} \
            --vl_feats_type ${FEATS_TYPE} \
            --vision_encoder_weights clip \
            --summary_type ${SUMMARY_TYPE} \
            --batch_size ${BATCH_SIZE} \
            --model vit_large_patch16 \
            --cls_token \
            --finetune ${PRETRAIN_CHKPT} \
            --epochs ${EPOCHS} \
            --blr ${LR} \
            --weight_decay ${WDECAY} \
            --data_path ${DATA_DIR} \
            --output_dir $EXP_NAME \
            --log_dir $EXP_NAME \
            --nb_classes 2 \
            --blip_feats_select avgpool \
            --cfg-path ${CFG_PATH}
