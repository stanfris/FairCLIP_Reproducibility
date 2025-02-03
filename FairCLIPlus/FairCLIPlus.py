from typing import Literal
import torch
import clip

class FairCLIPlus(torch.nn.Module):
    """
    FairCLIPlus is a modified version of CLIP that incorporates fairness-aware objectives.
    It applies additional loss functions to mitigate bias in image-text representations.
    """
    def __init__(self, attributes: dict[str, float],
                 model_architecture: Literal['RN50', 'ViT-B/16', 'ViT-L/14'],
                 device: Literal['cuda', 'cpu'],
                 loss_img,
                 loss_txt,
                 distance_loss,
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
            pretrained_weights (str, optional): Path to pretrained model weights. Defaults to "".
        """
        super().__init__()

        self.device = device
        self.attributes = attributes

        # Defining losses for the fairCLIP objective
        self.loss_img = loss_img
        self.loss_txt = loss_txt
        self.distance_loss = distance_loss

        # Load the specified CLIP model
        self.model, self.preprocess = clip.load(model_architecture, device)

        # Load pretrained weights if provided
        if pretrained_weights:
            checkpoint = torch.load(pretrained_weights)
            self.model.load_state_dict(checkpoint['model_state_dict'])

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

    def fairCLIPlus_loss(self, logits_per_image_batch, logits_per_text_batch, loss_img, loss_txt,
                        logits_per_attr: dict[dict[str, (torch.Tensor, torch.Tensor)]]):
        """
        Computes the fairness-aware loss for the FairCLIPlus model.

        Args:
            logits_per_image_batch (torch.Tensor): Logits for images in the batch.
            logits_per_text_batch (torch.Tensor): Logits for texts in the batch.
            loss_img (callable): Loss function for images.
            loss_txt (callable): Loss function for texts.
            logits_per_attr (dict[dict[str, (torch.Tensor, torch.Tensor)]]):
                Nested dictionary mapping attributes to image-text logits.

        Returns:
            torch.Tensor: The total computed loss incorporating both CLIP and fairness objectives.
        """
        ground_truth = torch.arange(
                len(logits_per_image_batch), dtype=torch.long, device=self.device)
        total_loss_clip = (loss_img(logits_per_image_batch, ground_truth) +
                        loss_txt(logits_per_text_batch, ground_truth)) / 2

        # Compute similarity scores
        similarity = logits_per_image_batch @ logits_per_text_batch.T
        correlations_with_batch = similarity.diag().float()

        # Compute fairness-related loss per attribute
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

        total_loss = total_loss_clip + distance_loss
        return total_loss
