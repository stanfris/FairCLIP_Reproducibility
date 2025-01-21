from src import logger
from src.modules import *
import os
import numpy as np
import random
import argparse
import time
import json
import pandas as pd
from collections import Counter
from geomloss import SamplesLoss

import clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import warnings
import sys
sys.path.append('.')

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='CLIP Training/Fine-Tuning')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--dataset_dir', default='./data', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--eval_set', default='test',
                    type=str, help='options: val | test')
parser.add_argument('--summarized_note_file', default='', type=str)
parser.add_argument('--text_source', default='note',
                    type=str, help='options: note | label')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--model_arch', default='vit-b16',
                    type=str, help='options: vit-b16 | vit-l14')
parser.add_argument('--pretrained_weights', default='', type=str)
parser.add_argument('--attribute', default='race', type=str,
                    help='race|gender|ethnicity|language')
parser.add_argument('--batchsize_fairloss', default=64, type=int)
parser.add_argument('--lambda_fairloss', default=1e-4, type=float)
parser.add_argument('--tmp_hp', default=1e-4, type=float)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    logger.log(f'===> random seed: {args.seed}')

    result_dir = args.result_dir + f"{args.seed}"

    logger.configure(dir=result_dir, log_suffix='eval')

    with open(os.path.join(result_dir, f'args_eval.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # the number of groups in each attribute
    groups_in_attrs = [3, 2, 2, 3]
    attr_to_idx = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}
    idx_to_attr_to_group = {0: {0: "asian", 1: "black", 2: "white"}, 1: {0: "female", 1: "male"}, 2: {0: "non-hispanic", 1: "hispanic"}, 3: {0: "english", 1: "spanish", 2: "other"}}

    model_arch_mapping = {'vit-b16': 'ViT-B/16', 'vit-l14': 'ViT-L/14'}

    best_global_perf_file = os.path.join(
        os.path.dirname(result_dir), f'best_{args.perf_file}')
    acc_head_str = ''
    auc_head_str = ''
    dpd_head_str = ''
    eod_head_str = ''
    esacc_head_str = ''
    esauc_head_str = ''
    group_disparity_head_str = ''
    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            for i in range(len(groups_in_attrs)):
                auc_head_str += ', '.join(
                    [f'auc_attr{i}_group{x}' for x in range(groups_in_attrs[i])]) + ', '
            dpd_head_str += ', '.join(
                [f'dpd_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            eod_head_str += ', '.join(
                [f'eod_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esacc_head_str += ', '.join(
                [f'esacc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esauc_head_str += ', '.join(
                [f'esauc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '

            group_disparity_head_str += ', '.join(
                [f'std_group_disparity_attr{x}, max_group_disparity_attr{x}' for x in range(len(groups_in_attrs))]) + ', '

            with open(best_global_perf_file, 'w') as f:
                f.write(
                    f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')

    # If using GPU then use mixed precision training.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Must set jit=False for training
    model, preprocess = clip.load(
        model_arch_mapping[args.model_arch], device=device, jit=False)

    test_dataset = fair_vl_med_dataset(
        args.dataset_dir, preprocess, subset='Test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, drop_last=False)

    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(
        test_dataset)
    logger.log(f'group size on race in test set: {group_size_on_race}')
    logger.log(f'group size on gender in test set: {group_size_on_gender}')
    logger.log(
        f'group size on ethnicity in test set: {group_size_on_ethnicity}')

    if device == "cpu":
        model.float()
    else:
        # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model)

    loss_for_FairCLIP = SamplesLoss(
        loss="sinkhorn", p=2, blur=args.tmp_hp)  # 0.05

    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)
        model.load_state_dict(checkpoint['model_state_dict'])

    for epoch in range(1):
        # iterate over test dataset
        all_probs = []
        all_labels = []
        all_attrs = []
        correlations_batch_total = []
        correlations_attributes = [{"asian": [], "black": [], "white": []}, {"female": [], "male": []}, {"non-hispanic": [], "hispanic": []}, {"english": [], "spanish": [], "other":[]}]
        for batch in test_dataloader:
            images, texts, label_and_attributes = batch

            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            attributes = label_and_attributes[:, 1:].tolist

            ground_truth = torch.arange(
                len(images), dtype=torch.long, device=device)

            similarity = logits_per_image @ logits_per_text.T
            correlations_with_batch = similarity.diag().float().tolist()
            correlations_batch_total += correlations_with_batch

            for idx, batch_correlation in enumerate(correlations_with_batch):
                idx_attributes = label_and_attributes[idx]
                for i in range(len(groups_in_attrs)):
                    correlations_attributes[i][idx_to_attr_to_group[i][[idx_attributes[i]]]].append(batch_correlation)

        # after the entire dataloader has been passed, calculate distances (unnormalized)
        distances = [{"asian": 0, "black": 0, "white": 0}, {"female": 0, "male": 0}, {"non-hispanic": 0, "hispanic": 0}, {"english": 0, "spanish": 0, "other": 0}]
        correlations_batch_total = torch.FloatTensor(correlations_batch_total)
        correlations_batch_total = correlations_batch_total.to(device)
        for attribute_id, num_groups in enumerate(groups_in_attrs):
            for group_id in range(num_groups):
                correlations_group = correlations_attributes[attribute_id][idx_to_attr_to_group[attribute_id][group_id]]
                correlations_group = torch.FloatTensor(correlations_group).to(device)
                distance = loss_for_FairCLIP(correlations_batch_total[:, None], correlations_group[:, None])
                distances[attribute_id][idx_to_attr_to_group[attribute_id][group_id]] = distance
                logger.log(f"Attribute: {attribute_id}, group: {idx_to_attr_to_group[attribute_id][group_id]}: distance: {distance}")
