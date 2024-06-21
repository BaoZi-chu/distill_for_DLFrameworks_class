import torch
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F


# 自定义的COCODataset类，用于加载和处理COCO数据集
class COCODataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        # root: 图片文件夹路径
        # annotation: 注释文件路径
        # transforms: 图像变换
        self.root = root
        self.coco = COCO(annotation)  # 加载COCO注释文件
        self.ids = list(self.coco.imgs.keys())  # 获取所有图片的ID
        self.transforms = transforms  # 图像变换函数

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]  # 获取图片ID
        ann_ids = coco.getAnnIds(imgIds=img_id)  # 获取该图片的所有注释ID
        annotations = coco.loadAnns(ann_ids)  # 加载所有注释信息
        img_info = coco.loadImgs(img_id)[0]  # 获取图片信息
        path = img_info['file_name']  # 获取图片文件名

        # 加载图像
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        # 获取边界框坐标并转换格式
        boxes = []
        labels = []
        for ann in annotations:
            x_min = ann['bbox'][0]
            y_min = ann['bbox'][1]
            x_max = x_min + ann['bbox'][2]
            y_max = y_min + ann['bbox'][3]

            # 过滤掉无效的边界框
            if ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])

        # 转换为 tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 创建目标字典，包含边界框、标签和图片ID
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        # 如果指定了图像变换，则应用变换
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target  # 返回图像和对应的目标

    def __len__(self):
        return len(self.ids)    # 返回数据集中图片的数量

# 自定义的CocoDetectionTransforms类，继承自CocoDetection类，增加了自定义变换功能
class CocoDetectionTransforms(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super(CocoDetectionTransforms, self).__init__(img_folder, ann_file)
        self._transforms = transforms   # 自定义变换函数

    def __getitem__(self, idx):
        img, target = super(CocoDetectionTransforms, self).__getitem__(idx)
        image_id = self.ids[idx]    # 获取图片ID
        target = self._transform_target(target, image_id)   # 转换目标
        if self._transforms is not None:
            img, target = self._transforms(img, target) # 应用自定义变换
        return img, target  # 返回图像和转换后的目标

    # 将目标转换为指定格式
    def _transform_target(self, target, image_id):
        boxes = []
        labels = []
        for obj in target:
            x_min, y_min, width, height = obj["bbox"]
            x_max = x_min + width
            y_max = y_min + height
            if width > 0 and height > 0:  # 过滤无效边界框
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(obj["category_id"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return {"boxes": boxes, "labels": labels, "image_id": torch.tensor([image_id])}

# 获取图像变换函数
def get_transform():
    def transform(img, target):
        img = F.to_tensor(img)  # 将图像转换为tensor
        return img, target  # 返回转换后的图像和目标
    return transform