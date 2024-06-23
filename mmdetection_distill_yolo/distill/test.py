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
        
    说明:
        本函数设置全局随机种子，并可选地启用 CUDA 确定性选项，这会牺牲一定性能以确保结果一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True  # 开启 PyTorch 的确定性模式
        torch.backends.cudnn.benchmark = False     # 禁用 CUDA 的基准测试功能，以保证计算的确定性

def test_teacher_model():
    """初始化并返回用于测试的教师模型的 Runner 对象。
    
    Returns:
        Runner: 配置好的教师模型 Runner。
    """
    # 载入教师模型配置文件和权重
    teacher_cfg = '../mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py'
    teacher_weights = '../../checkpoint/yolox_distill/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
    teacher_cfg = Config.fromfile(teacher_cfg)
    teacher_cfg.work_dir = '../../checkpoint/yolox_distill/yolox_s_8x8_300e_coco_distill'

    # 创建 Runner 对象并加载模型
    runner = Runner.from_cfg(teacher_cfg)
    runner.model = init_detector(teacher_cfg, teacher_weights)
    return runner

def test_student_model():
    """初始化并返回用于测试的学生模型的 Runner 对象。
    
    Returns:
        Runner: 配置好的学生模型 Runner。
    """
    # 载入学生模型配置文件和权重
    cfg = Config.fromfile('config/yolox_s_8x8_300e_coco_distill.py')
    checkpoint_file = '../../checkpoint/yolox_distill/epoch_10.pth'
    cfg.work_dir = '../../checkpoint/yolox_distill/'  # 设置工作目录

    # 创建 Runner 对象并加载模型
    runner = Runner.from_cfg(cfg)
    runner.model = init_detector(cfg, checkpoint_file)
    return runner

if __name__ == '__main__':
    # 初始化学生模型并进行测试
    runner = test_student_model()
    set_random_seed(0)  # 设置随机种子
    runner.test()  # 执行模型测试
