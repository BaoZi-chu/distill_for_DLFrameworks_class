import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKLDivLoss(nn.Module):
    def __init__(self, temperature=1.0, loss_weight=1.0):
        super(DistillKLDivLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, student_outputs, teacher_outputs,loss_weight=1.0):
        self.loss_weight = loss_weight
        student_logits = student_outputs / self.temperature
        teacher_logits = teacher_outputs / self.temperature
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )/student_logits.shape[0]

        return self.loss_weight * loss * (self.temperature ** 2)
