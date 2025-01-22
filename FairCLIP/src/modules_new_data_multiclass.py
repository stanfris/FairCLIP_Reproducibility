import os
import numpy as np
import random
from PIL import Image
import math
import copy
import pandas as pd

import re

import clip

import torch
import torch.nn as nn

from torchvision.models import *
import torch.nn.functional as F

from sklearn.metrics import *
from fairlearn.metrics import *

from natsort import natsorted

class fairface_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='../../data/fairface/', preprocess=None, subset='Training', summarized_notes_file_train='fairface_label_train.csv', summarized_notes_file_val='fairface_label_val.csv', ruleout_unknown=False, group_loader=False, attribute=0, thegroup=0):
        self.age_mapping = {
            "0-2": 0,
            "3-9": 1,
            "10-19": 2,
            "20-29": 3,
            "30-39": 4,
            "40-49": 5,
            "50-59": 6,
            "60-69": 7,
            "more than 70": 8
        }
        self.race_mapping = {
            "East Asian": 0,
            "Indian": 1,
            "Black": 2,
            "White": 3,
            "Middle Eastern": 4,
            "Southeast Asian": 5,
            "Latino_Hispanic": 6
        }
        self.age_mapping_inv = {v: k for k, v in self.age_mapping.items()}
        self.race_mapping_inv = {v: k for k, v in self.race_mapping.items()}
        self.preprocess = preprocess
        self.subset = subset
        self.ruleout_unknown = ruleout_unknown
        if subset=='Training':
            self.dataset_dir = os.path.join(dataset_dir, 'train/')
        else:
            self.dataset_dir = os.path.join(dataset_dir, 'val/')
        self.files = natsorted(os.listdir(self.dataset_dir))[:1100]

        self.summarized_notes = {}

        # check if the split file exists
        if subset=='Training':
            df = pd.read_csv(os.path.join(dataset_dir, summarized_notes_file_train)).iloc[:1100]
            self.data = df
            self.dataset_dir = os.path.join(dataset_dir, 'train/')

            if group_loader:
                tmp_files = []
                for file in self.files:
                    if attribute == 'age':
                        group = self.age_mapping.get(self.data[self.data.file == "train/" + file]["age"].item())
                    else:
                        group = self.race_mapping.get(self.data[self.data.file == "train/" + file]["race"].item())
                    if group == thegroup:
                        tmp_files.append(file)
                self.files = tmp_files
        else:
            print("Loading validation")
            df = pd.read_csv(os.path.join(dataset_dir, summarized_notes_file_val)).iloc[:1100]
            self.data = df
            if group_loader:
                tmp_files = []
                for file in self.files:
                    if attribute == 0:
                        group = self.age_mapping.get(self.data[self.data.file == "val/" + file]["age"].item())
                    else:
                        group = self.race_mapping.get(self.data[self.data.file == "val/" + file]["race"].item())
                    if group == thegroup:
                        tmp_files.append(file)
                self.files = tmp_files
            
        



    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.files[idx])
        img = Image.open(image_path)
        final_image = self.preprocess(img)
        row = self.data.loc[idx].to_dict()
        race_label = int(self.race_mapping.get(row['race']))
        if self.subset == 'Training':
            note = "Image of a person that is " + row['age']+ " years old, and their race is: " + row['race'] + "They are " + row['gender']
            token = clip.tokenize(note)
            token = token.squeeze()
        else:
            eastasian = 'A picture of an East Asian person'
            eastasian = clip.tokenize(eastasian)
            indian = 'A picture of an Indian person'
            indian= clip.tokenize(indian)
            black = 'A picture of a Black person'
            black=clip.tokenize(black)
            white = 'A picture of a White person'
            white=clip.tokenize(white)
            middle_eastern = 'A picture of a Middle Eastern person'
            middle_eastern = clip.tokenize(middle_eastern)
            southeast_asian = 'A picture of a Southeast Asian person'
            southeast_asian = clip.tokenize(southeast_asian)
            latino_hispanic = 'A picture of a Latino Hispanic person'
            latino_hispanic = clip.tokenize(latino_hispanic)
            # concatenate two tensors together, the final tensor will be at size of 2, 77
            token = torch.cat([eastasian, indian, black, white, middle_eastern, southeast_asian, latino_hispanic], dim=0)
        # extract labels from df
        

        age_label = int(self.age_mapping.get(row["age"]))
        gender_label = int(row["gender"]=='Male')
        
        label_and_attributes = torch.tensor([race_label, race_label])

        return final_image, token, label_and_attributes


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def compute_vl_prob(img_feats, class_txt_feats):
    # img_feats: [batch_size, 512]
    # class_txt_feats: [batch_size, num_class, 512]

    all_logits = []
    for i in range(class_txt_feats.shape[1]):
        similarity = (img_feats @ class_txt_feats[:, i, :].T)
        # extract the diagonal of the matrix
        logits = similarity.diag()
        all_logits.append(logits)

    all_logits = torch.stack(all_logits, dim=1)

    # compute the probability by applying softmax along the second dimension
    vl_prob = torch.softmax(all_logits, dim=1)

    return vl_prob, all_logits

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if len(output.shape) == 1:
        acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
        return acc.item()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]
        output = torch.tensor(output).to(torch.float32)
        _, pred = output.topk(maxk, dim=1)
        target = torch.tensor(target).to(torch.int64)
        target = target.view(batch_size, 1).repeat(1, maxk)
        
        correct = (pred == target)
  
        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item()
            accuracy /= batch_size # [0, 1.]
            topk_accuracy.append(accuracy)
        
        return topk_accuracy[0]

def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    print(y)
    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        auc_val = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(y, num_classes)
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')

    return auc_val

def auc_score(pred_prob, y):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if np.unique(y).shape[0]>2:
        AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    else:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        AUC = auc(fpr, tpr)
    
    return AUC

def num_to_onehot(nums, num_classes):
    nums = nums.astype(int)
    return np.eye(num_classes)[nums]

def prob_to_label(pred_prob):
    # Find the indices of the highest probabilities for each sample
    max_prob_indices = np.argmax(pred_prob, axis=1)

    # Create one-hot vectors for each sample
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1

    return one_hot_vectors

def numeric_to_one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=np.int32)

    if num_classes is None:
        num_classes = np.max(y) + 1
    
    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1
    
    return one_hot_array

def multiclass_demographic_parity(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(pred_one_hot[:,i],
                                gt_one_hot[:,i],
                                sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_equalized_odds(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(pred_one_hot[:,i],
                            gt_one_hot[:,i],
                            sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_demographic_parity_(pred_prob, y, attrs):

    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    attrs_set = np.unique(attrs)
    y_pred = np.argmax(pred_prob, axis=1)

    mc_dpd = 0
    for i in range(pred_prob.shape[1]):
        tmp_preds = (y_pred==i).astype(int)
        tmp_not_preds = 1 - tmp_preds

        dp_by_attrs = []
        for j in attrs_set:
            idx = attrs==j
            tmp = np.abs(tmp_preds.mean().item() - tmp_preds[idx].mean().item()) + np.abs(tmp_not_preds.mean().item() - tmp_not_preds[idx].mean().item())
            dp_by_attrs.append(tmp)
        mc_dpd += np.mean(dp_by_attrs).item()

    mc_dpd = mc_dpd / pred_prob.shape[1]
        
    return mc_dpd

def auc_score_multiclass(pred_prob, y, num_of_class=3, eps=0.01):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    sensitivity_at_diff_specificity = [-1]*4
    y_onehot = num_to_onehot(y, num_of_class)
    fpr, tpr, thresholds = roc_curve(y_onehot.ravel(), pred_prob.ravel())
    for i in range(len(fpr)):
        cur_fpr = fpr[i]
        cur_tpr = tpr[i]
        if np.abs(cur_fpr-0.2) <= eps:
            sensitivity_at_diff_specificity[0] = cur_tpr
        if np.abs(cur_fpr-0.15) <= eps:
            sensitivity_at_diff_specificity[1] = cur_tpr
        if np.abs(cur_fpr-0.1) <= eps:
            sensitivity_at_diff_specificity[2] = cur_tpr
        if np.abs(cur_fpr-0.05) <= eps:
            sensitivity_at_diff_specificity[3] = cur_tpr
    AUC = auc(fpr, tpr)
    
    return AUC, sensitivity_at_diff_specificity

def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    if len(output.shape) >= 2:
        overall_acc = np.sum(np.argmax(output, axis=1) == target)/target.shape[0]
    else:
        overall_acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if len(pred_group.shape) >= 2:
            acc = np.sum(np.argmax(pred_group, axis=1) == gt_group)/gt_group.shape[0]
        else:
            acc = np.sum((pred_group >= 0.5).astype(float) == gt_group)/gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_acc)
    es_acc = (overall_acc / (alpha*tmp + 1))
    
    return es_acc

def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    
    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(target, output)
        overall_auc = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(target, num_classes)
        overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')

    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if num_classes == 2:
            fpr, tpr, thresholds = roc_curve(gt_group, pred_group)
            group_auc = auc(fpr, tpr)
        elif num_classes > 2:
            y_onehot = num_to_onehot(gt_group, num_classes)
            group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')
        
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_auc)
    es_auc = (overall_auc / (alpha*tmp + 1))

    return es_auc


def evaluate_comprehensive_perf(preds, gts, attrs=None, num_classes=2):
    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    eods = []
    between_group_disparity = []

    overall_acc = accuracy(preds, gts, topk=(1,))
    
    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i, :]  # Attribute i

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)

        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append(
                compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes)
            )
        aucs_by_attrs.append(aucs_by_group)

        std_disparity, max_disparity = compute_between_group_disparity(
            aucs_by_group, overall_auc
        )
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = preds.argmax(axis=1) if num_classes > 2 else (preds >= 0.5).astype(float)

        if num_classes == 2:
            dpd = demographic_parity_difference(
                gts, pred_labels, sensitive_features=attr
            )
            eod = equalized_odds_difference(
                gts, pred_labels, sensitive_features=attr
            )
        else:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            eod = multiclass_equalized_odds(preds, gts, attr)

        dpds.append(dpd)
        eods.append(eod)

    return (
        overall_acc,
        esaccs_by_attrs,
        overall_auc,
        esaucs_by_attrs,
        aucs_by_attrs,
        dpds,
        eods,
        between_group_disparity,
    )

def compute_between_group_disparity(auc_list, overall_auc):
    return np.std(auc_list) / overall_auc, (np.max(auc_list)-np.min(auc_list)) / overall_auc

def compute_between_group_disparity_half(auc_list, overall_auc):
    return np.std(auc_list) / np.abs(overall_auc-0.5), (np.max(auc_list)-np.min(auc_list)) / np.abs(overall_auc-0.5)

def endless_loader(dataloader):
    while True:
        for data in dataloader:
            yield data