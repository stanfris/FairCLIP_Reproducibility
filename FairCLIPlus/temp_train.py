import torch.utils
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


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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

    return parser


class FairCLIPPlusLoss(nn.Module):
    def __init__(self, loss_img, loss_txt, fairness_weight, attributes, distance_loss):
        super().__init__()

        self.attributes = attributes
        self.fairness_weight = fairness_weight
        self.loss_img = loss_img
        self.loss_txt = loss_txt
        self.distance_loss = distance_loss

    def forward(self, logits_per_image, logits_per_text, features_per_attribute, correlations_with_batch, ground_truth):
        loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text, ground_truth)) / 2

        # Compute fairness-related loss per attribute
        if self.fairness_weight == 0:
            return loss

        distance_loss = 0

        for attribute_name, attr_weight in self.attributes.items():
            if attr_weight == 0:
                continue
            attribute_loss = 0
            for _, (group_im_logits, group_txt_logits) in features_per_attribute[attribute_name].items():
                correlations_with_group = group_im_logits.diag().float()

                attribute_loss += self.distance_loss(correlations_with_batch[:, None],
                                                     correlations_with_group[:, None])
            distance_loss += attribute_loss * attr_weight

        loss = loss + self.fairness_weight * distance_loss

        return loss



def train_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: FairCLIPPlusLoss,
        train_dataset: DataLoader,
        attribute_group_dataset: DataLoader,
        device: torch.device
        ):
    # train loop
    avg_train_loss = 0
    for batch in train_dataset:
        images, texts, _ = batch
        images = images.to(device)
        texts = texts.to(device)

        optimizer.zero_grad()

        logits_per_image, logits_per_text = model(images, texts)


        # Compute similarity scores
        correlations_with_batch = logits_per_image.diag().float()

        # compute similarity scores for batches
        logits_per_attr = dict()

        for attribute_name, group_dls in attribute_group_dataset.items():
            logits_per_group = dict()

            for group_idx, group_dl in group_dls.items():
                group_images, group_texts, _ = next(group_dl)
                group_images = group_images.to(device)
                group_texts = group_texts.to(device)

                with torch.no_grad():
                    group_image_logits, group_text_logits = model(group_images, group_texts)

                logits_per_group[group_idx] = (group_image_logits, group_text_logits)

            logits_per_attr[attribute_name] = logits_per_group


        ground_truth = torch.arange(len(logits_per_image), dtype=torch.long, device=device)
        loss = loss_fn(logits_per_image, logits_per_text, logits_per_attr, correlations_with_batch, ground_truth)

        avg_train_loss += loss.detach().item()

        loss.backward()
        optimizer.step()

    avg_train_loss /= len(train_dataset)

    return model, avg_train_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_dataset: DataLoader,
    device: torch.device
):
    # Keep track of all evaluation methods
    all_probs = []
    all_labels = []
    all_attrs = []
    eval_avg_loss = 0

    for batch in eval_dataset:
        images, texts, label_and_attributes = batch

        images = images.to(device)
        texts = texts.to(device)

        labels = label_and_attributes[:, 0].to(device)
        attributes = label_and_attributes[:, 1:].to(device)

        # Zero-shot prediction
        class_text_feats = []

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=1, keepdim=True)

        for i in range(texts.shape[1]):
            text_features = model.encode_text(texts[:, i, :])
            text_features /= text_features.norm(dim=1, keepdim=True)
            class_text_feats.append(text_features[:, None, :])

        # concatentate class_text_feats along the second dimension
        class_text_feats = torch.cat(class_text_feats, dim=1)

        # computes a softmax over all similarity diagonals
        # over all individual features in the batch.
        # Then computes a softmax over these values
        vl_prob, _ = compute_vl_prob(
            image_features, class_text_feats)

        all_probs.append(vl_prob[:, 1].cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_attrs.append(attributes.cpu().numpy())

        # apply binary cross entropy loss
        loss = F.binary_cross_entropy(
            vl_prob[:, 1].float(), labels.float())
        eval_avg_loss += loss.item()


    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_attrs = np.concatenate(all_attrs, axis=0)
    eval_avg_loss /= len(eval_dataset)

    eval_results = evaluate_comprehensive_perf(all_probs, all_labels, all_attrs.T)

    return eval_avg_loss, eval_results, all_probs, all_labels, all_attrs



def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: FairCLIPPlusLoss,
    epochs: int,
    train_dataset: DataLoader,
    eval_dataset: DataLoader,
    attribute_group_dataset: DataLoader,
    device: torch.device,
    logger,
    result_dir: str
):
    best_epoch = 0
    best_loss = 1000000
    best_auc_groups = None
    best_acc_groups = None
    best_pred_gt_by_attr = None
    best_auc = sys.float_info.min
    best_acc = sys.float_info.min
    best_es_acc = sys.float_info.min
    best_es_auc = sys.float_info.min
    best_dpd_groups = None
    best_eod_groups = None
    best_between_group_disparity = None

    for epoch in range(epochs):
        model.train()
        model, avg_train_loss = train_step(model, optimizer, loss_fn, train_dataset, attribute_group_dataset, device)

        # model.eval()
        eval_avg_loss, eval_results, all_probs, all_labels, all_attrs = evaluate(model, eval_dataset, device)

        overall_acc, eval_es_acc, overall_auc, eval_es_auc, \
            eval_aucs_by_attrs, eval_dpds, eval_eods, \
                between_group_disparity = eval_results


        logger.log(f'===> epoch[{epoch:03d}/{epochs:03d}], training loss: {avg_train_loss:.4f}, eval loss: {eval_avg_loss:.4f}')

        if best_auc <= overall_auc:
            best_auc = overall_auc
            best_acc = overall_acc
            best_epoch = epoch
            best_auc_groups = eval_aucs_by_attrs
            best_dpd_groups = eval_dpds
            best_eod_groups = eval_eods
            best_es_acc = eval_es_acc
            best_es_auc = eval_es_auc
            best_between_group_disparity = between_group_disparity

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': eval_avg_loss,
            }, os.path.join(result_dir, "best_model.pth"))

        if result_dir is not None:
            np.savez(os.path.join(result_dir, f'pred_gt_ep{epoch:03d}.npz'),
                     val_pred=all_probs, val_gt=all_labels, val_attr=all_attrs)

        logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_epoch}')
        logger.log(
            f'---- best AUC by groups and attributes at epoch {best_epoch}')
        logger.log(best_auc_groups)

        logger.logkv('epoch', epoch)
        logger.logkv('trn_loss', round(avg_train_loss, 4))

        logger.logkv('eval_loss', round(eval_avg_loss, 4))
        logger.logkv('eval_acc', round(overall_acc, 4))
        logger.logkv('eval_auc', round(overall_auc, 4))

        for ii in range(len(eval_es_acc)):
            logger.logkv(f'eval_es_acc_attr{ii}', round(eval_es_acc[ii], 4))
        for ii in range(len(eval_es_auc)):
            logger.logkv(f'eval_es_auc_attr{ii}', round(eval_es_auc[ii], 4))
        for ii in range(len(eval_aucs_by_attrs)):
            for iii in range(len(eval_aucs_by_attrs[ii])):
                logger.logkv(f'eval_auc_attr{ii}_group{iii}', round(
                    eval_aucs_by_attrs[ii][iii], 4))

        for ii in range(len(between_group_disparity)):
            logger.logkv(f'eval_auc_attr{ii}_std_group_disparity', round(
                between_group_disparity[ii][0], 4))
            logger.logkv(f'eval_auc_attr{ii}_max_group_disparity', round(
                between_group_disparity[ii][1], 4))

        for ii in range(len(eval_dpds)):
            logger.logkv(f'eval_dpd_attr{ii}', round(eval_dpds[ii], 4))
        for ii in range(len(eval_eods)):
            logger.logkv(f'eval_eod_attr{ii}', round(eval_eods[ii], 4))

        logger.dumpkvs()

    return (
        best_epoch,
        best_loss,
        best_auc_groups,
        best_acc_groups,
        best_pred_gt_by_attr,
        best_auc,
        best_acc,
        best_es_acc,
        best_es_auc,
        best_between_group_disparity,
        best_dpd_groups,
        best_eod_groups
    )





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')
    parser = init_parser(parser)
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

    # Initializing model and optimizer
    attributes_weights = dict()
    for i, attr in enumerate(args.attributeslist):
        attributes_weights[attr] = args.weightslist[i]

    # load CLIP
    model, preprocess = clip.load(args.model_arch, device)
    model.float()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.1, 0.1), eps=1e-6, weight_decay=args.weight_decay)

    # Load pretrained weights if provided
    if args.pretrained_weights:
        checkpoint = torch.load(args.pretrained_weights)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.float()
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.to(device)


    # Initializing losses
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    distance_loss = SamplesLoss(loss="sinkhorn", p=2, blur=args.sinkhorn_blur, scaling=0.95)
    fairclip_loss = FairCLIPPlusLoss(
        loss_img=loss_img,
        loss_txt=loss_txt,
        fairness_weight=args.lambda_fairloss,
        attributes=attributes_weights,
        distance_loss=distance_loss
    )

    # Loading data
    train_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Training',
                                        text_source=args.text_source, summarized_note_file=args.summarized_note_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=False)

    val_dataset = fair_vl_med_dataset(
        args.dataset_dir, preprocess, subset='Validation')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, drop_last=False)

    all_attribute_dataloaders = dict()
    for attr_index, attr in enumerate(args.attributeslist):
        # get different dataloaders for each group inside an attribute (for example: male, female; or: English, Spanish)
        group_dataloaders = dict()
        for group_idx in range(args.groups_per_attr[attr_index]):
            tmp_dataset = fair_vl_group_dataset(args.dataset_dir, preprocess,
                                                text_source='note', summarized_note_file=args.summarized_note_file,
                                                attribute=attr, thegroup=group_idx)
            tmp_dataloader = DataLoader(tmp_dataset, batch_size=args.batchsize_fairloss, shuffle=True,
                                        num_workers=args.workers, pin_memory=True, drop_last=False)
            group_dataloaders[group_idx] = endless_loader(tmp_dataloader)
        all_attribute_dataloaders[attr] = group_dataloaders

    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(
        train_dataset)
    logger.log(f'group size on race in training set: {group_size_on_race}')
    logger.log(f'group size on gender in training set: {group_size_on_gender}')
    logger.log(
        f'group size on ethnicity in training set: {group_size_on_ethnicity}')
    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(
        val_dataset)
    logger.log(f'group size on race in validation set: {group_size_on_race}')
    logger.log(f'group size on gender in validation set: {group_size_on_gender}')
    logger.log(
        f'group size on ethnicity in validation set: {group_size_on_ethnicity}')

    # train model
    (
        best_epoch,
        best_loss,
        best_auc_groups,
        best_acc_groups,
        best_pred_gt_by_attr,
        best_auc,
        best_acc,
        best_es_acc,
        best_es_auc,
        best_between_group_disparity,
        best_dpd_groups,
        best_eod_groups
    ) = train(
        model, optimizer, fairclip_loss,
        args.num_epochs, train_dataloader, val_dataloader,
        all_attribute_dataloaders, device, logger, result_dir)

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
                f.write(f'{best_epoch}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')

    os.rename(result_dir,
              f'{result_dir}_seed{args.seed}_auc{best_auc:.4f}')

