from typing import Literal
import torch
import clip
from torch.utils.data import DataLoader
from src.modules import *
from src import logger

class FairCLIPlus(torch.nn.Module):
    """
    FairCLIPlus is a modified version of CLIP that incorporates fairness-aware objectives.
    It applies additional loss functions to mitigate bias in image-text representations.
    """
    def __init__(self, attributes: dict[str, float],
                 model_architecture: Literal['RN50', 'ViT-B/16', 'ViT-L/14'],
                 device: Literal['cuda', 'cpu'],
                 loss_img: callable,
                 loss_txt: callable,
                 distance_loss: callable,
                 fairness_weight: float,
                 pretrained_weights: str = ""):
        """
        Initializes the FairCLIPlus model.

        Args:
            attributes (dict[str, float]): Dictionary mapping attribute names to their weights.
            model_architecture (Literal['RN50', 'ViT-B/16', 'ViT-L/14']):
                                    Specifies the CLIP model architecture to use.
            device (Literal['cuda', 'cpu']): Device on which the model will be run.
            loss_img (callable): Loss function for image representations.
            loss_txt (callable): Loss function for text representations.
            distance_loss (callable): Loss function for measuring distance between correlations.
            fairness_weight (float): Indicates how heavy the distance should count
            pretrained_weights (str, optional): Path to pretrained model weights. Defaults to "".
        """
        super().__init__()

        self.device = device
        self.attributes = attributes
        self.fairness_weight = fairness_weight

        # Defining losses for the fairCLIP objective
        self.loss_img = loss_img
        self.loss_txt = loss_txt
        self.distance_loss = distance_loss

        # Load the specified CLIP model
        self.model, self.preprocess = clip.load(model_architecture, device)
        self.model.float()

        # Load pretrained weights if provided
        if pretrained_weights:
            checkpoint = torch.load(pretrained_weights)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.float()

    def forward(self, images, texts):
        """
        Performs a forward pass through the CLIP model.

        Args:
            images (torch.Tensor): Batch of image tensors.
            texts (torch.Tensor): Batch of text tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits for image-text similarity.
        """
        images = images.to(self.device)
        texts = texts.to(self.device)
        logits_per_image, logits_per_text = self.model(images, texts)
        return logits_per_image, logits_per_text

    def fairCLIPlus_loss(self, logits_per_image_batch, logits_per_text_batch,
                        logits_per_attr: dict[str, dict[int, (torch.Tensor, torch.Tensor)]]):
        """
        Computes the fairness-aware loss for the FairCLIPlus model.

        Args:
            logits_per_image_batch (torch.Tensor): Logits for images in the batch.
            logits_per_text_batch (torch.Tensor): Logits for texts in the batch.
            logits_per_attr (dict[str, dict[str, (torch.Tensor, torch.Tensor)]]):
                Nested dictionary mapping attributes to image-text logits.

        Returns:
            torch.Tensor: The total computed loss incorporating both CLIP and fairness objectives.
        """
        ground_truth = torch.arange(
                len(logits_per_image_batch), dtype=torch.long, device=self.device)
        total_loss_clip = (self.loss_img(logits_per_image_batch, ground_truth) +
                        self.loss_txt(logits_per_text_batch, ground_truth)) / 2

        # Compute similarity scores
        similarity = logits_per_image_batch @ logits_per_text_batch.T
        correlations_with_batch = similarity.diag().float()

        # Compute fairness-related loss per attribute
        # TODO: implement standardize
        if self.fairness_weight == 0:
            return total_loss_clip

        distance_loss = 0
        for attribute_name, attr_weight in self.attributes.items():
            if attr_weight == 0:
                continue
            attribute_loss = 0
            for _, (group_im_logits, group_txt_logits) in logits_per_attr[attribute_name].items():
                similarity = group_im_logits @ group_txt_logits.T
                correlations_with_group = similarity.diag().float()
                #TODO: change interface for distance loss function so that it is more generalizable
                attribute_loss += self.distance_loss(correlations_with_batch[:, None],
                                                     correlations_with_group[:, None])
            distance_loss += attribute_loss * attr_weight

        total_loss = total_loss_clip + self.fairness_weight * distance_loss
        return total_loss


def train_model(model: FairCLIPlus, optimizer, batch_dataset_train: DataLoader,
                batch_dataset_val: DataLoader, attribute_group_dl: dict[str, dict[int, DataLoader]],
                result_dir: str, output_model_name:str = "clip.pth", epochs: int = 10,
                verbose: bool = True, start_epoch: int = 0):
    """
    TODO
    """
    # Default values
    best_auc = 0
    best_acc = 0
    best_ep = -1
    best_auc_groups = None
    best_dpd_groups = None
    best_eod_groups = None
    best_es_acc = 0
    best_es_auc = 0
    best_between_group_disparity = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        # sanity check
        if not model.model.training:
            raise Exception("Internal CLIP model still in evaluation mode")

        # train loop
        for batch in batch_dataset_train:
            optimizer.zero_grad()

            images, texts, _ = batch

            logits_per_image, logits_per_text = model(images, texts)

            logits_per_attr = dict()

            for attribute_name, group_dls in attribute_group_dl.items():
                logits_per_group = dict()
                for group_idx, group_dl in group_dls.items():
                    group_images, group_texts, _ = next(group_dl)

                    # We don't want this model to see heaps more data -
                    # this would make it an unfair comparison
                    with torch.no_grad():
                        group_image_logits, group_text_logits = model(group_images, group_texts)

                    logits_per_group[group_idx] = (group_image_logits, group_text_logits)

                logits_per_attr[attribute_name] = logits_per_group

            loss = model.fairCLIPlus_loss(logits_per_image, logits_per_text, logits_per_attr)

            loss.backward()
            optimizer.step()

        # Eval loop
        model.eval()
        # Sanity check:
        if model.model.training:
            raise Exception("Internal CLIP model still in training mode")

        # Keep track of all evaluation methods
        all_probs = []
        all_labels = []
        all_attrs = []
        eval_avg_loss = 0
        for batch in batch_dataset_val:
            images, texts, label_and_attributes = batch

            labels = label_and_attributes[:, 0].to(model.device)
            attributes = label_and_attributes[:, 1:].to(model.device)

            # Zero-shot prediction
            class_text_feats = []
            with torch.no_grad():
                image_features = model.model.encode_image(images)
                image_features /= image_features.norm(dim=1, keepdim=True)

                for i in range(texts.shape[1]):
                    text_features = model.model.encode_text(texts[:, i, :])
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

            # Check if model should be updated:

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_attrs = np.concatenate(all_attrs, axis=0)
        eval_avg_loss /= len(batch_dataset_val)

        overall_acc, eval_es_acc, overall_auc, eval_es_auc, \
            eval_aucs_by_attrs, eval_dpds, eval_eods, \
                between_group_disparity = evalute_comprehensive_perf(
                                            all_probs, all_labels, all_attrs.T)

        if best_auc <= overall_auc:
            best_auc = overall_auc
            best_acc = overall_acc
            best_ep = epoch
            best_auc_groups = eval_aucs_by_attrs
            best_dpd_groups = eval_dpds
            best_eod_groups = eval_eods
            best_es_acc = eval_es_acc
            best_es_auc = eval_es_auc
            best_between_group_disparity = between_group_disparity

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': eval_avg_loss,
            }, os.path.join(result_dir, output_model_name))

        # save npz values of each epoch
        if result_dir is not None:
            np.savez(os.path.join(result_dir, f'pred_gt_ep{epoch:03d}.npz'),
                     val_pred=all_probs, val_gt=all_labels, val_attr=all_attrs)

        # log all to output
        if verbose:
            logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
            logger.log(
                f'---- best AUC by groups and attributes at epoch {best_ep}')
            logger.log(best_auc_groups)

            logger.logkv('epoch', epoch)

            logger.logkv('eval_loss', round(eval_avg_loss, 4))
            logger.logkv('eval_acc', round(overall_acc, 4))
            logger.logkv('eval_auc', round(overall_auc, 4))

            for ii, _ in enumerate(eval_es_acc):
                logger.logkv(f'eval_es_acc_attr{ii}', round(eval_es_acc[ii], 4))
            for ii, _ in enumerate(eval_es_auc):
                logger.logkv(f'eval_es_auc_attr{ii}', round(eval_es_auc[ii], 4))
            for ii, _ in enumerate(eval_aucs_by_attrs):
                for iii, _ in enumerate(eval_aucs_by_attrs[ii]):
                    logger.logkv(f'eval_auc_attr{ii}_group{iii}', round(
                        eval_aucs_by_attrs[ii][iii], 4))

            for ii, _ in enumerate(between_group_disparity):
                logger.logkv(f'eval_auc_attr{ii}_std_group_disparity', round(
                    between_group_disparity[ii][0], 4))
                logger.logkv(f'eval_auc_attr{ii}_max_group_disparity', round(
                    between_group_disparity[ii][1], 4))

            for ii, _ in enumerate(eval_dpds):
                logger.logkv(f'eval_dpd_attr{ii}', round(eval_dpds[ii], 4))
            for ii, _ in enumerate(eval_eods):
                logger.logkv(f'eval_eod_attr{ii}', round(eval_eods[ii], 4))

            logger.dumpkvs()

    # Done with all epochs
    return best_auc, best_acc, best_ep, best_auc_groups, best_dpd_groups, \
            best_eod_groups, best_es_acc, best_es_auc, best_between_group_disparity
