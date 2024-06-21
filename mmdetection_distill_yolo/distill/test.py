import torch
import numpy as np
import random

from mmdet.apis import init_detector
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from model.YOLOXDistillation import YOLOXDistillation
from model.distill_hook import DistillHook


def set_random_seed(seed, deterministic=False):
    """设置随机种子以保证实验的可重复性。

    Args:
        seed (int): 要设置的随机种子值。
        deterministic (bool, optional): 是否设置确定性选项，以确保结果的可重复性。默认值为 False。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_teacher_model():
    teacher_cfg = '../mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py'
    teacher_weights = '../../checkpoint/yolox_distill/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
    teacher_cfg = Config.fromfile(teacher_cfg)
    teacher_cfg.work_dir = '../../checkpoint/yolox_distill/yolox_s_8x8_300e_coco_distill'
    runner = Runner.from_cfg(teacher_cfg)
    runner.model = init_detector(teacher_cfg, teacher_weights)
    return runner
def test_student_model():
    cfg = Config.fromfile('config/yolox_s_8x8_300e_coco_distill.py')
    checkpoint_file = '../../checkpoint/yolox_distill/epoch_10.pth'
    # 设置工作目录，用于保存训练过程中的检查点和日志
    cfg.work_dir = '../../checkpoint/yolox_distill/'

    # 根据配置文件创建Runner对象
    runner = Runner.from_cfg(cfg)
    runner.model = init_detector(cfg, checkpoint_file)
    return runner
if __name__ == '__main__':


    runner = test_student_model()
    # 设置随机种子以保证实验的可重复性
    set_random_seed(0)
    runner.test()

