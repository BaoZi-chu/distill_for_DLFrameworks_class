import torch
from mmdet.structures import SampleList
from mmengine import is_list_of
from torch import Tensor
import torch.nn.functional as F
from mmdet.registry import MODELS
from collections import OrderedDict

from mmdet.models import YOLOX

from .DistillKLDivLoss import DistillKLDivLoss
from typing import Tuple, Union,Dict


@MODELS.register_module()
class YOLOXDistillation(YOLOX):
    def __init__(self, *args, alpha=0.5, temperature=4.0,distill=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.teacher_model = None
        self.alpha = alpha
        self.temperature = temperature
        self.distill = distill

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        assert self.teacher_model is not None, 'please provide teacher model '
        student_feature_maps = self.extract_feat(batch_inputs)
        # student_bbox = self.bbox_head(student_feature_maps)
        # iou_loss = torch.nn.SmoothL1Loss()
        losses = self.bbox_head.loss(student_feature_maps, batch_data_samples)
        # 计算蒸馏损失
        with torch.no_grad():
            teacher_feature_maps = self.teacher_model.extract_feat(batch_inputs)
            # teacher_bbox = self.teacher_model.bbox_head(teacher_feature_maps)

        losses['distillation_feature_map_0_loss'] = F.mse_loss(student_feature_maps[0],teacher_feature_maps[0])
        losses['distillation_feature_map_1_loss'] = F.mse_loss(student_feature_maps[1], teacher_feature_maps[1])
        losses['distillation_feature_map_2_loss'] = F.mse_loss(student_feature_maps[2], teacher_feature_maps[2])
        # kl_loss_fun = DistillKLDivLoss(temperature=self.temperature)
        # losses['distillation_kl_loss'] = kl_loss_fun(student_feature_maps[2],teacher_feature_maps[2])

        return losses

    def parse_losses(
            self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses and distillation losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key and 'distillation' not in key)
        distillation_loss = sum(value for key, value in log_vars if 'loss' in key and 'distillation' in key)
        loss = self.alpha * distillation_loss + (1 - self.alpha) * loss
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars
