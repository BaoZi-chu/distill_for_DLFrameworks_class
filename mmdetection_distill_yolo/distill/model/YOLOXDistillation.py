import torch
from mmdet.structures import SampleList  # Custom data structure from MMDetection
from mmengine import is_list_of  # Utility function to check list types
from torch import Tensor
import torch.nn.functional as F
from mmdet.registry import MODELS  # Registry from MMDetection to manage model architectures
from collections import OrderedDict

from mmdet.models import YOLOX  # Import the base YOLOX model

from .DistillKLDivLoss import DistillKLDivLoss  # Import a custom distillation loss class
from typing import Tuple, Union, Dict

@MODELS.register_module()  # Register this class as a module in MMEngine's model registry
class YOLOXDistillation(YOLOX):
    """
    A YOLOX model class modified for knowledge distillation, which allows a student model to learn
    from a teacher model's outputs to improve its predictions.

    Attributes:
        alpha (float): Weighting factor for combining distillation loss with standard loss.
        temperature (float): Temperature scaling applied in distillation loss to soften probability distributions.
        distill (bool): Flag to enable/disable distillation.
    """
    def __init__(self, *args, alpha=0.5, temperature=4.0, distill=True, **kwargs):
        """
        Initializes the YOLOXDistillation model with distillation settings.
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = None  # Placeholder for the teacher model
        self.alpha = alpha  # Weight for distillation loss
        self.temperature = temperature  # Temperature for softening logits in distillation
        self.distill = distill  # Flag to enable/disable distillation

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        """
        Calculates the combined loss for the model, including both the task-specific losses and
        distillation losses if distillation is enabled.

        Args:
            batch_inputs (Tensor): Input images for the batch.
            batch_data_samples (SampleList): Ground truth data samples associated with the batch.

        Returns:
            Union[dict, list]: A dictionary containing all computed losses.
        """
        assert self.teacher_model is not None, 'please provide teacher model'
        student_feature_maps = self.extract_feat(batch_inputs)  # Extract features using the student model
        losses = self.bbox_head.loss(student_feature_maps, batch_data_samples)  # Compute base losses
        
        # Compute distillation losses by comparing student and teacher feature maps
        with torch.no_grad():  # Disable gradients for teacher model to save computation
            teacher_feature_maps = self.teacher_model.extract_feat(batch_inputs)  # Extract features using the teacher model

        # Add distillation losses for each feature map
        losses['distillation_feature_map_0_loss'] = F.mse_loss(student_feature_maps[0], teacher_feature_maps[0])
        losses['distillation_feature_map_1_loss'] = F.mse_loss(student_feature_maps[1], teacher_feature_maps[1])
        losses['distillation_feature_map_2_loss'] = F.mse_loss(student_feature_maps[2], teacher_feature_maps[2])

        return losses

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parses the raw output losses from the network, calculating the total loss and preparing
        variables for logging.

        Args:
            losses (Dict[str, torch.Tensor]): Dictionary of loss names and their corresponding tensor values.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total computed loss and a dictionary for logging purposes.
        """
        log_vars = []
        # Iterate through losses to calculate total and log individual losses
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        # Calculate total and distillation-specific losses
        loss = sum(value for key, value in log_vars if 'loss' in key and 'distillation' not in key)
        distillation_loss = sum(value for key, value in log_vars if 'loss' in key and 'distillation' in key)
        # Combine losses according to the alpha weighting
        loss = self.alpha * distillation_loss + (1 - self.alpha) * loss
        log_vars.insert(0, ['loss', loss])  # Insert total loss at the beginning for logging
        log_vars = OrderedDict(log_vars)  # Convert list to OrderedDict for consistent logging order

        return loss, log_vars
