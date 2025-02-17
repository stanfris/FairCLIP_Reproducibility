from typing import Callable, Dict, Tuple

import argparse
import sys

sys.path.append('../../FairCLIP/src')
from modules import fair_vl_med_dataset, set_random_seed


from pathlib import Path

import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip

from geomloss import SamplesLoss


GROUPS_IN_ATTRS = [3, 2, 2, 3]
ATTR_TO_IDX = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}
IDX_TO_ATTR = {val: key for key, val in ATTR_TO_IDX.items()}
IDX_TO_GROUP = {0: {0: "asian", 1: "black", 2: "white"}, 1: {0: "female", 1: "male"}, 2: {0: "non-hispanic", 1: "hispanic"}, 3: {0: "english", 1: "spanish", 2: "other"}}
MODEL_ARCH_MAPPING = {'vit-b16': 'ViT-B/16', 'vit-l14': 'ViT-L/14'}


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--dataset_dir', default='./data', type=str)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--summarized_note_file', default="gpt-4_summarized_notes.csv", type=str)
    parser.add_argument('--model_arch', required=True, type=str, help='options: vit-b16 | vit-l14')
    parser.add_argument('--checkpoint', type=str, help="Model checkpoint path")
    parser.add_argument('--out', type=str, help="Output (pickle) file")

    return parser


@torch.no_grad()
def get_all_similarities(model: nn.Module, data_loader: DataLoader, device: str, standardize: bool = False, normalize: bool = False) -> Tuple[torch.FloatTensor, Dict[Tuple[int, int], torch.FloatTensor]]:
    model.eval()

    correlations = []
    correlation_group_indices = {}

    for batch in data_loader:
        images, texts, label_and_attributes = batch
        images = images.to(device)
        texts = texts.to(device)

        images = images.to(device)
        texts = texts.to(device)
        values, _ = model(images, texts)
        similarity = values.diag().float().tolist()
        # similarity = image_features.diag().float().tolist()
        correlations += similarity
        attributes = label_and_attributes[:, 1:].tolist()

        for sample_idx, sample_attributes in enumerate(attributes):
            for attr_idx, attr_group in enumerate(sample_attributes):
                key = (attr_idx, attr_group)

                if key not in correlation_group_indices:
                    correlation_group_indices[key] = []

                correlation_group_indices[key].append(similarity[sample_idx])

    # convert to tensors
    correlations = torch.FloatTensor(correlations)
    converted_correlation_group_indices = {}

    for key, values in correlation_group_indices.items():
        conv_values = torch.FloatTensor(values)

        if standardize:
            conv_values = (conv_values - torch.mean(conv_values)) / torch.std(conv_values)
        converted_correlation_group_indices[key] = conv_values

    if standardize:
        correlations = (correlations - torch.mean(correlations)) / torch.std(correlations)

    return correlations, converted_correlation_group_indices


def compute_distances(
        all_correlations: torch.FloatTensor,
        group_correlations: Dict[Tuple[int, int], torch.FloatTensor],
        distance_fn: Callable
        ) -> Dict[Tuple[int, int], float]:

    distances = {}

    for group, attr_group_correlations in group_correlations.items():
        distances[group] = distance_fn(all_correlations[:, None], attr_group_correlations[:, None])

    return distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLIP Training/Fine-Tuning')
    init_parser(parser)
    args = parser.parse_args()

    set_random_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_ARCH_MAPPING[args.model_arch], device=device, jit=False)

    if args.checkpoint is not None:
        model_checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(model_checkpoint['model_state_dict'])

    distance_fn = SamplesLoss(loss="sinkhorn", p=2, blur=1e-4, diameter=2, scaling=0.95)
    # changed blur to 0.1 due to the documentation
    # distance_fn = SamplesLoss(loss="gaussian", p=2, blur=0.1, diameter=2, scaling=0.95)

    test_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Test', present_as_training=True, summarized_note_file=args.summarized_note_file)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    # compute distances
    correlations, group_correlations = get_all_similarities(model, test_dataloader, device)
    distances = compute_distances(correlations, group_correlations, distance_fn)

    # print results
    attribute_groups_labels = sorted(group_correlations.keys())

    for attr_group in attribute_groups_labels:
        attr_idx, group_idx = attr_group
        attribute_name = IDX_TO_ATTR[attr_idx]
        group_name = IDX_TO_GROUP[attr_idx][group_idx]
        distance = distances[attr_group]
        print(f"Attribute: {attribute_name}, group: {group_name}, distance: {distance}")


    if args.out is not None and args.results_dir is not None:
        out_path = Path(args.results_dir) / args.out
        out_data = {"all": correlations, "groups": group_correlations, "distances": distances}

        with open(out_path, "wb") as f:
            pickle.dump(out_data, f)

        print(f"Wrote similarity scores to {out_path}")
