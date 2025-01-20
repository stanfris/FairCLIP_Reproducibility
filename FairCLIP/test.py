from src import logger
from src.modules_new_data import *
import os
import numpy as np
import random
import argparse
import time
import json
import pandas as pd
from collections import Counter

import clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt


def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/16', device=device, jit=False)

    val_dataset = fairface_dataset(
        '../data/fairface/', preprocess, subset='Validation', summarized_notes_file_val='fairface_label_val.csv')
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                num_workers=4, pin_memory=True, drop_last=False)
    
    clip.model.convert_weights(model)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.transformer.parameters(), "lr": 0.0001},
        {"params": model.visual.parameters(), "lr": 0.0001},
    ], lr=0.0001, betas=(0.1, 0.1), eps=1e-6, weight_decay=0)

    accuracies = []

    for batch in val_dataloader:
        images, texts, label_and_attributes = batch

        images = images.to(device)
        texts = texts.to(device)
        combined_labels = label_and_attributes[:, 0].to(device)
        attributes = label_and_attributes[:, 1:].to(device)

        class_text_feats = []
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=1, keepdim=True)

            for i in range(texts.shape[1]):
                text_features = model.encode_text(texts[:, i, :])
                text_features /= text_features.norm(dim=1, keepdim=True)
                class_text_feats.append(text_features[:, None, :])
            # concatentate class_text_feats along the second dimension
            class_text_feats = torch.cat(class_text_feats, dim=1)

        all_logits = []
        for i in range(class_text_feats.shape[1]):
            similarity = (image_features @ class_text_feats[:, i, :].T)
            # extract the diagonal of the matrix
            logits = similarity.diag()
            all_logits.append(logits)

        all_logits = torch.stack(all_logits, dim=1)
        predictions = torch.argmax(all_logits, dim=1)
        print(predictions)
        break
        # Calculate accuracy
        accuracy = (predictions == combined_labels).float().mean().item()
        accuracies.append(accuracy)

    print(f"Accuracy: {np.mean(accuracies)}")
    print(f"Std: {np.std(accuracies)}")
    print(f"Min: {np.min(accuracies)}")
    print(f"Max: {np.max(accuracies)}")
    print(f"Median: {np.median(accuracies)}")
    print(f"25th percentile: {np.percentile(accuracies, 25)}")
    print(f"75th percentile: {np.percentile(accuracies, 75)}")


