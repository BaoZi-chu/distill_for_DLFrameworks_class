# Import necessary libraries
import os
import torch
import numpy as np
import random
from mmdet.apis import init_detector
from mmengine.config import Config
from mmengine.runner import Runner
from model.YOLOXDistillation import YOLOXDistillation  
from model.distill_hook import DistillHook  

def set_random_seed(seed, deterministic=False):
    """
    Sets the random seed for all used computational libraries to ensure reproducibility.
    
    Args:
        seed (int): The random seed.
        deterministic (bool): If True, makes operations deterministic but may reduce performance. Default is False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True  # Ensures consistent results on the cost of performance
        torch.backends.cudnn.benchmark = False     # Disables the benchmarking feature that speeds up training

if __name__ == '__main__':
    # Set environment variables for CUDA device order and visibility to ensure specific GPU is used
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Load the model configuration from a predefined file
    cfg = Config.fromfile('config/yolox_s_8x8_300e_coco_distill.py')
    
    # Set the working directory to save checkpoints and logs
    cfg.work_dir = '../../checkpoint/yolox_distill'
    
    # Initialize the training runner with the loaded configuration
    runner = Runner.from_cfg(cfg)
    
    # Load a checkpoint file to resume training or for initialization
    checkpoint_file = '../../checkpoint/yolox_distill/epoch_2.pth'
    runner.model = init_detector(cfg, checkpoint_file)
    
    # Set the random seed for reproducibility
    set_random_seed(0, deterministic=False)  # Deterministic flag can be toggled based on performance requirement
    
    # Start the training process
    runner.train()
    
    # Start the testing process to evaluate the model
    runner.test()
