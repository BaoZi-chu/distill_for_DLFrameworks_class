import torch
from torch.utils.data import DataLoader, Subset
from coco2017.coco_dataset import CocoDetectionTransforms, get_transform
import sys
import os
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.teacher_model import get_teacher_model
from models.student_model import get_student_model
from models.distillation_loss import calculate_loss
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from torchvision.transforms import functional as F

# 提取 collate_fn 到全局作用域
def collate_fn(batch):
    return tuple(zip(*batch))

# 可视化样本图像及其目标边界框
def visualize_sample(images, targets):
    fig, ax = plt.subplots(1, len(images), figsize=(20, 5))
    if len(images) == 1:
        ax = [ax]
    for idx, image in enumerate(images):
        img = image.cpu().permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        ax[idx].imshow(img)
        for box in targets[idx]['boxes']:
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='red')
            ax[idx].add_patch(rect)
    plt.show()


def main():
    # 定义数据转换
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 设置训练数据路径和注释文件路径
    train_data_root = '../../data/coco2017/train2017'
    train_ann_file = '../../data/coco2017/annotations/instances_train2017.json'


    # 加载COCO数据集，并应用数据转换
    train_dataset = CocoDetectionTransforms(img_folder=train_data_root, ann_file=train_ann_file,
                                            transforms=get_transform())

    # 仅使用10%的数据进行训练
    np.random.seed(42)  # 固定随机种子
    indices = np.random.choice(len(train_dataset), size=int(0.001 * len(train_dataset)), replace=False)
    train_subset = Subset(train_dataset, indices)

    # 创建数据加载器
    train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # 获取教师模型和学生模型
    teacher_model = get_teacher_model()
    student_model = get_student_model(num_classes=91)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.to(device)
    student_model.to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    num_epochs = 4
    temperature = 3.0
    initial_alpha = 0.8
    final_alpha = 0.4
    alpha_step = (initial_alpha - final_alpha) / num_epochs

    # 训练循环
    for epoch in range(num_epochs):
        student_model.train()
        alpha = max(final_alpha, initial_alpha - epoch * alpha_step)

        progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, targets) in enumerate(progress_bar):
            try:
                images = [image.to(device) for image in images]
                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

                # 在第一个批次上可视化样本图像及其目标边界框
                if batch_idx == 0:
                    visualize_sample(images, targets)

                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_model.eval()    # 设置教师模型为评估模式
                    teacher_outputs, teacher_features = teacher_model(images, targets)

                student_outputs_train, student_features = student_model(images, targets)

                # 计算总损失
                loss = calculate_loss(teacher_features, student_features, student_outputs_train, temperature, alpha)

                loss.backward()     # 反向传播
                optimizer.step()    # 优化器更新

                progress_bar.set_postfix(loss=loss.item())
            except Exception as e:
                print(f"Error during training: {str(e)}. Continuing...")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 保存学生模型的状态字典
    torch.save(student_model.state_dict(), '../../checkpoint/faster_rcnn_distill/student_model_distilled.pth')
    print("学生模型已保存至 '../../checkpoint/faster_rcnn_distill/student_model_distilled.pth'")

    print("模型蒸馏训练完成。")


if __name__ == '__main__':
    main()
