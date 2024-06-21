# CUDA setting
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
import random

from mmdet.apis import init_detector
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner import Runner
from model.YOLOXDistillation import YOLOXDistillation
from model.distill_hook import DistillHook

def set_random_seed(seed, deterministic=False):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    cfg = Config.fromfile('config/yolox_s_8x8_300e_coco_distill.py')
    # 设置工作目录
    cfg.work_dir = '../../checkpoint/yolox_distill'
    runner = Runner.from_cfg(cfg)
    checkpoint_file = '../../checkpoint/yolox_distill/epoch_2.pth'
    runner.model = init_detector(cfg, checkpoint_file)
    set_random_seed(0)
    # start training
    runner.train()
    # start test
    runner.test()