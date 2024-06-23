from mmdet.apis import init_detector  # Import initialization function for detectors
from mmengine.registry import HOOKS  # Import the registry for hooks
from mmengine.hooks import Hook  # Base class for hooks
import torch.nn.functional as F  # Import functional API for torch, although not used here
import torch  # Import PyTorch, not used directly in this snippet
from typing import Dict, Optional, Sequence, Union  # Import typing classes for type annotations

# Type alias for data batches, can be either dict, tuple, or list, and optionally None
DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()  # Register this class as a module within hooks
class DistillHook(Hook):
    """
    Hook for distillation during training. This hook manages a teacher model
    that provides additional supervision to the main model during training.

    Attributes:
        teacher_model_config (Optional[Dict]): Configuration for the teacher model.
        teacher_checkpoint (Optional[str]): Path to the teacher model's checkpoint.
        priority (int): Priority of the hook, used to determine execution order of hooks.
    """
    def __init__(self, teacher_model_config=None, teacher_checkpoint=None, priority=48):
        """
        Initializes the DistillHook.

        Args:
            teacher_model_config (Optional[Dict]): Configuration dictionary for the teacher model.
            teacher_checkpoint (Optional[str]): Path to the checkpoint to load the teacher model.
            priority (int): Priority level of the hook, defaults to 48.
        """
        super().__init__()
        self.teacher_model_config = teacher_model_config
        self.teacher_checkpoint = teacher_checkpoint

    def before_train(self, runner):
        """
        Action to perform before training starts. Initializes the teacher model and sets it to eval mode.

        Args:
            runner: The training runner instance.
        """
        runner.model.teacher_model = init_detector(self.teacher_model_config, self.teacher_checkpoint)
        runner.model.teacher_model.eval()  # Set the teacher model to evaluation mode

    def after_train(self, runner) -> None:
        """
        Action to perform after training completes. Cleans up by removing the teacher model.

        Args:
            runner: The training runner instance.
        """
        runner.model.teacher_model = None  # Dereference the teacher model to free memory
