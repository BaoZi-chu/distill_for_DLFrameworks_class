from mmdet.apis import init_detector
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import torch.nn.functional as F
import torch
from typing import Dict, Optional, Sequence, Union

DATA_BATCH = Optional[Union[dict, tuple, list]]
@HOOKS.register_module()
class DistillHook(Hook):
    def __init__(self,teacher_model_config=None, teacher_checkpoint=None,priority=48):
        super().__init__()
        self.teacher_model_config = teacher_model_config
        self.teacher_checkpoint = teacher_checkpoint
    def before_train(self, runner):
        runner.model.teacher_model = init_detector(self.teacher_model_config, self.teacher_checkpoint)
        runner.model.teacher_model.eval()
    def after_train(self, runner) -> None:
        runner.model.teacher_model = None


    # def after_train_iter(self, runner,batch_idx: int,
    #                      data_batch: DATA_BATCH = None,
    #                      outputs: Optional[dict] = None):
    #     student_outputs = outputs['loss']
    #     # images = runner.data_batch['img']
    #     with torch.no_grad():
    #         inputs = torch.stack(data_batch['inputs'])
    #
    #         # 确保 inputs 类型与模型权重类型一致
    #         inputs = inputs.to(next(self.teacher_model.parameters()).device).float()
    #         teacher_outputs = self.teacher_model(inputs)
    #     # Compute distillation loss
    #     loss_distill = self.compute_distill_loss(student_outputs, teacher_outputs)
    #     runner.outputs['loss'] = runner.outputs['loss'] + loss_distill
    #
    # def compute_distill_loss(self, student_outputs, teacher_outputs):
    #     loss = F.kl_div(F.log_softmax(student_outputs / self.temperature, dim=1),
    #                     F.softmax(teacher_outputs / self.temperature, dim=1),
    #                     reduction='batchmean') * (self.temperature ** 2)
    #     return self.alpha * loss