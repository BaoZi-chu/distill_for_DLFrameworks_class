import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKLDivLoss(nn.Module):
    """
    A custom loss module for knowledge distillation using the KL divergence.

    This loss function calculates the divergence between the soft predictions of the student
    and the teacher models, scaled by a temperature parameter to control the softness of
    probability distributions and a weight to control the contribution of this loss in the
    total training loss.

    Attributes:
        temperature (float): The temperature factor to apply to logits. Higher values
                             produce softer probability distributions.
        loss_weight (float): The factor by which the loss will be scaled.
    """
    def __init__(self, temperature=1.0, loss_weight=1.0):
        """
        Initializes the DistillKLDivLoss with optional temperature and loss weight.

        Args:
            temperature (float, optional): Controls the softness of the output distributions.
                                           Defaults to 1.0.
            loss_weight (float, optional): Controls the weight of this loss in total loss.
                                           Defaults to 1.0.
        """
        super(DistillKLDivLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, student_outputs, teacher_outputs, loss_weight=1.0):
        """
        Forward pass for calculating the distillation loss.

        Args:
            student_outputs (torch.Tensor): The logits output by the student model.
            teacher_outputs (torch.Tensor): The logits output by the teacher model.
            loss_weight (float, optional): Dynamic adjustment to the loss's contribution
                                           during training. Defaults to 1.0.

        Returns:
            torch.Tensor: The calculated distillation loss.
        """
        self.loss_weight = loss_weight  # Update the loss weight if provided during the forward pass
        student_logits = student_outputs / self.temperature  # Scale student outputs
        teacher_logits = teacher_outputs / self.temperature  # Scale teacher outputs

        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),  # Softmax with log for the student logits
            F.softmax(teacher_logits, dim=-1),      # Softmax for the teacher logits
            reduction='batchmean'                   # Mean over the batch for scaling the loss
        ) / student_logits.shape[0]                # Normalize the loss by the batch size

        return self.loss_weight * loss * (self.temperature ** 2)  # Scale the loss by weight and squared temperature
