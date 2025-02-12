import sys
import os
import random
import argparse
import optuna

import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from torch import optim

import clip

from geomloss import SamplesLoss

from src import logger
from src.modules import (
    compute_vl_prob,
    endless_loader,
    evaluate_comprehensive_perf,
    fair_vl_group_dataset,
    fair_vl_med_dataset,
    set_random_seed
)

from wandb_logger import WandbLogger


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

    parser.add_argument('--seed', default=42, type=int,
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
    parser.add_argument('--project', type=str)
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
        base_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text, ground_truth)) / 2

        # Compute fairness-related loss per attribute
        if self.fairness_weight == 0:
            return base_loss

        distance_loss = 0

        for attribute_name, attr_weight in self.attributes.items():
            if attr_weight == 0:
                continue
            attribute_loss = 0

            for _, group_scaled_similarities in features_per_attribute[attribute_name].items():
                attribute_loss += self.distance_loss(correlations_with_batch[:, None],
                                                     group_scaled_similarities[:, None])
            distance_loss += attribute_loss * attr_weight

        loss = base_loss + self.fairness_weight * distance_loss

        return loss, base_loss.detach().item(), distance_loss.detach().item()


def train_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: FairCLIPPlusLoss,
        train_dataset: DataLoader,
        attribute_group_dataset: DataLoader,
        device: torch.device,
        ):
    # train loop
    avg_train_loss = 0
    avg_train_clip_loss = 0
    avg_train_distance = 0

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
                    group_scaled_similarities, _ = model(group_images, group_texts)

                logits_per_group[group_idx] = group_scaled_similarities.diag().float()

            logits_per_attr[attribute_name] = logits_per_group

        ground_truth = torch.arange(len(logits_per_image), dtype=torch.long, device=device)
        loss, clip_loss, distance_loss_val = loss_fn(logits_per_image, logits_per_text, logits_per_attr, correlations_with_batch, ground_truth)

        loss.backward()
        optimizer.step()

        avg_train_loss += loss.detach().item()
        avg_train_clip_loss += clip_loss
        avg_train_distance += distance_loss_val

    avg_train_loss /= len(train_dataset)
    avg_train_clip_loss /= len(train_dataset)
    avg_train_distance /= len(train_dataset)

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
):
    best_auc = sys.float_info.min

    for _ in range(epochs):
        model.train()
        model, _ = train_step(model, optimizer, loss_fn, train_dataset, attribute_group_dataset, device)

        model.eval()
        _, eval_results, _, _, _ = evaluate(model, eval_dataset, device)

        _, _, overall_auc, _, _, _, _, _ = eval_results


        if best_auc <= overall_auc:
            best_auc = overall_auc

    return best_auc


def objective(trial):
    parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')
    parser = init_parser(parser)
    args = parser.parse_args()

    args.lr = trial.suggest_float("lr", 1e-8, 1e-2, log=True)
    args.lambda_fairloss = trial.suggest_float("lambda_fairloss", 0.1, 1000, step=100)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

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
        if args.weightslist[attr_index] == 0:
            all_attribute_dataloaders[attr] = dict()
            continue

        group_dataloaders = dict()
        for group_idx in range(args.groups_per_attr[attr_index]):
            tmp_dataset = fair_vl_group_dataset(args.dataset_dir, preprocess,
                                                text_source='note', summarized_note_file=args.summarized_note_file,
                                                attribute=attr, thegroup=group_idx)
            tmp_dataloader = DataLoader(tmp_dataset, batch_size=args.batchsize_fairloss, shuffle=True,
                                        num_workers=args.workers, pin_memory=True, drop_last=False)
            group_dataloaders[group_idx] = endless_loader(tmp_dataloader)
        all_attribute_dataloaders[attr] = group_dataloaders

    best_auc = train(model, optimizer, fairclip_loss,
                     args.num_epochs, train_dataloader, val_dataloader,
                     all_attribute_dataloaders, device)

    return best_auc

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150)

    trial = study.best_trial
    params_str = '\n'.join([f"{key}: {value}" for key, value in trial.params.items()])
    msg = f'''Best Trial
    ============================
    Best AUC: {trial.value:.4f}
    Best Hyperparameters:\n''' + params_str
    print(msg)