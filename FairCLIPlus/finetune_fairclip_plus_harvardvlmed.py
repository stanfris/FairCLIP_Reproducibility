from src import logger
from src.modules import *
import os
import numpy as np
import random
import argparse
import json
import pandas as pd
from geomloss import SamplesLoss
from FairCLIPlus import FairCLIPlus, train_model

import clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import sys
sys.path.append('.')


parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')

parser.add_argument('--model_arch', default='ViT-B/16',
                    type=str, help='options: ViT-B/16 | ViT-L/14')
parser.add_argument('--pretrained_weights', default='', type=str)
parser.add_argument(
  "--attributeslist",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=["race", "gender", "ethnicity", "language"],  # default if nothing is provided
)
parser.add_argument(
  "--weightslist",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=float,
  default=[0.5, 0.5, 0.0, 0.0],  # default if nothing is provided
)
parser.add_argument(
  "--groups_per_attr",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[3, 2, 2, 3],  # default if nothing is provided
)

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
parser.add_argument('--batchsize_fairloss', default=64, type=int)
parser.add_argument('--lambda_fairloss', default=1e-4, type=float)
parser.add_argument('--sinkhorn_blur', default=1e-4, type=float)

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--dataset_dir', default='./data', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--summarized_note_file', default='', type=str)
parser.add_argument('--text_source', default='note',
                    type=str, help='options: note | label')
parser.add_argument('--perf_file', default='', type=str)


if __name__ == '__main__':
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    logger.log(f'===> random seed: {args.seed}')

    result_dir = args.result_dir + f"{args.seed}"

    logger.configure(dir=result_dir, log_suffix='train')

    with open(os.path.join(result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # creates final log file
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
            for i in range(len(args.groups_per_attr)):
                auc_head_str += ', '.join(
                    [f'auc_attr{i}_group{x}' for x in range(args.groups_per_attr[i])]) + ', '
            dpd_head_str += ', '.join(
                [f'dpd_attr{x}' for x in range(len(args.groups_per_attr))]) + ', '
            eod_head_str += ', '.join(
                [f'eod_attr{x}' for x in range(len(args.groups_per_attr))]) + ', '
            esacc_head_str += ', '.join(
                [f'esacc_attr{x}' for x in range(len(args.groups_per_attr))]) + ', '
            esauc_head_str += ', '.join(
                [f'esauc_attr{x}' for x in range(len(args.groups_per_attr))]) + ', '

            group_disparity_head_str += ', '.join(
                [f'std_group_disparity_attr{x}, max_group_disparity_attr{x}' for x in range(len(args.groups_per_attr))]) + ', '

            with open(best_global_perf_file, 'w') as f:
                f.write(
                    f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')

    # Initializing losses
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_for_FairCLIP = SamplesLoss(
        loss="sinkhorn", p=2, blur=args.sinkhorn_blur)

    # Initializing model and optimizer
    attributes_weights = dict()
    for i, attr in enumerate(args.attributeslist):
        attributes_weights[attr] = args.weightslist[i]

    fair_clip_plus = FairCLIPlus(attributes_weights, args.model_arch, device, loss_img, loss_txt,
                                loss_for_FairCLIP, args.lr, args.weight_decay, args.lambda_fairloss, args.pretrained_weights)
    fair_clip_plus = fair_clip_plus.to(device)

    # Loading data
    train_dataset = fair_vl_med_dataset(args.dataset_dir, fair_clip_plus.preprocess, subset='Training',
                                        text_source=args.text_source, summarized_note_file=args.summarized_note_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=False)

    val_dataset = fair_vl_med_dataset(
        args.dataset_dir, fair_clip_plus.preprocess, subset='Validation')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, drop_last=False)

    test_dataset = fair_vl_med_dataset(
        args.dataset_dir, fair_clip_plus.preprocess, subset='Test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, drop_last=False)

    all_attribute_dataloaders = dict()
    for attr_index, attr in enumerate(args.attributeslist):
        # get different dataloaders for each group inside an attribute (for example: male, female; or: English, Spanish)
        group_dataloaders = dict()
        for group_idx in range(args.groups_per_attr[attr_index]):
            tmp_dataset = fair_vl_group_dataset(args.dataset_dir, fair_clip_plus.preprocess,
                                                text_source='note', summarized_note_file=args.summarized_note_file,
                                                attribute=attr, thegroup=group_idx)
            tmp_dataloader = DataLoader(tmp_dataset, batch_size=args.batchsize_fairloss, shuffle=True,
                                        num_workers=args.workers, pin_memory=True, drop_last=False)
            group_dataloaders[group_idx] = endless_loader(tmp_dataloader)
        all_attribute_dataloaders[attr] = group_dataloaders

    best_auc, best_acc, best_ep, best_auc_groups, best_dpd_groups, \
            best_eod_groups, best_es_acc, best_es_auc, \
            best_between_group_disparity = train_model(fair_clip_plus,
                                                       train_dataloader, val_dataloader,
                                                       all_attribute_dataloaders, result_dir)

    # Log to corresponding file
    if args.perf_file != '':
        if os.path.exists(best_global_perf_file):
            with open(best_global_perf_file, 'a') as f:

                esacc_head_str = ', '.join(
                    [f'{x:.4f}' for x in best_es_acc]) + ', '
                esauc_head_str = ', '.join(
                    [f'{x:.4f}' for x in best_es_auc]) + ', '

                auc_head_str = ''
                for i in range(len(best_auc_groups)):
                    auc_head_str += ', '.join(
                        [f'{x:.4f}' for x in best_auc_groups[i]]) + ', '

                group_disparity_str = ''
                for i in range(len(best_between_group_disparity)):
                    group_disparity_str += ', '.join(
                        [f'{x:.4f}' for x in best_between_group_disparity[i]]) + ', '

                dpd_head_str = ', '.join(
                    [f'{x:.4f}' for x in best_dpd_groups]) + ', '
                eod_head_str = ', '.join(
                    [f'{x:.4f}' for x in best_eod_groups]) + ', '

                path_str = f'{result_dir}_seed{args.seed}_auc{best_auc:.4f}'
                f.write(f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')

    os.rename(result_dir,
              f'{result_dir}_seed{args.seed}_auc{best_auc:.4f}')
