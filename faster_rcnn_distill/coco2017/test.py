import torch
from torch.utils.data import DataLoader, Subset
from coco2017.coco_dataset import CocoDetectionTransforms, get_transform
import sys
import os
from models.teacher_model import get_teacher_model, get_teacher_model_eval
from models.student_model import get_student_model
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from pycocotools.cocoeval import COCOeval
import json
from torchvision.transforms import functional as F
from pycocotools.coco import COCO

# 函数用于整理批次数据
def collate_fn(batch):
    return tuple(zip(*batch))


# 可视化样本图像、目标和预测结果
def visualize_sample(images, targets, predictions, score_threshold=0.0):
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

        for box, score in zip(predictions[idx]['boxes'], predictions[idx]['scores']):
            if score > score_threshold:
                x_min, y_min, x_max, y_max = box.cpu().numpy()
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='blue')
                ax[idx].add_patch(rect)
        print(f"Predictions for image {idx}: {predictions[idx]['boxes']}, Scores: {predictions[idx]['scores']}")
    plt.show()

# 模型评估函数
def evaluate_model(type, model, data_loader, device, visualize=False, score_threshold=0.05):
    model.eval()
    coco_results = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [image.to(device) for image in images]
            outputs = model(images)
            if type == "student":
                outputs = outputs[0]
            for target, output in zip(targets, outputs):
                if len(target) == 0:
                    continue
                image_id = target['image_id'].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                if visualize:
                    visualize_sample(images, targets, outputs, score_threshold=score_threshold)

                for box, score, label in zip(boxes, scores, labels):
                    if score >= score_threshold:
                        coco_results.append({
                            "image_id": int(image_id),
                            "category_id": int(label),
                            "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                            "score": float(score)
                        })

    return coco_results

# 评估和总结函数
def evaluate_and_summarize(coco_gt, results_json_path):
    coco_dt = coco_gt.loadRes(results_json_path)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats

def main():
    # COCO数据集路径
    coco_root = '../data/coco2017'
    ann_file = os.path.join(coco_root, 'annotations/instances_val2017.json')
    img_folder = os.path.join(coco_root, 'val2017')

    # 数据集和DataLoader
    dataset = CocoDetectionTransforms(img_folder, ann_file, transforms=get_transform())
    np.random.seed(42)
    indices = np.random.choice(len(dataset), size=int(0.1 * len(dataset)), replace=False)
    val_subset = Subset(dataset, indices)
    val_data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 设备设置
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 加载预训练模型
    teacher_model = get_teacher_model_eval()
    teacher_model.to(device)

    # 评估教师模型
    teacher_results = evaluate_model("teacher",teacher_model, val_data_loader, device, score_threshold=0.05)

    # 保存教师模型的评估结果
    with open('teacher_results.json', 'w') as f:
        json.dump(teacher_results, f, indent=4)

    # 评估和总结教师模型
    teacher_stats = evaluate_and_summarize(COCO(ann_file), 'teacher_results.json')
    print(f"Teacher Model AP: {teacher_stats[0]:.4f}")

    # 评估学生模型
    student_model = get_student_model(num_classes=91)
    student_model.load_state_dict(torch.load('.../../checkpoint/faster_rcnn_distill/student_model_distilled.pth'))
    student_model.to(device)
    student_results = evaluate_model("student",student_model, val_data_loader, device, score_threshold=0.05)

    # 保存学生模型的评估结果
    with open('student_results.json', 'w') as f:
        json.dump(student_results, f, indent=4)

    # 评估和总结学生模型
    student_stats = evaluate_and_summarize(COCO(ann_file), 'student_results.json')
    print(f"Student Model AP: {student_stats[0]:.4f}")

    print("Evaluation completed.")

if __name__ == '__main__':
    main()
