import torch
import torch.nn.functional as F

# 特征归一化函数，使用层归一化
def normalize_features(x):
    return F.layer_norm(x, normalized_shape=x.shape[1:])

# 蒸馏损失计算函数，基于特征图的KL散度
def distillation_loss_feature(student_features, teacher_features, temperature):
    loss = 0.0
    # 如果输入为字典，则提取其值
    student_features = student_features.values() if isinstance(student_features, dict) else student_features
    teacher_features = teacher_features.values() if isinstance(teacher_features, dict) else teacher_features

    # 逐层计算学生模型和教师模型特征图的蒸馏损失
    for sf, tf in zip(student_features, teacher_features):
        sf = normalize_features(sf) / temperature
        tf = normalize_features(tf) / temperature
        sf_log_softmax = F.log_softmax(sf, dim=-1)
        tf_softmax = F.softmax(tf, dim=-1)
        num_elements = sf.numel()
        loss += F.kl_div(sf_log_softmax, tf_softmax, reduction='batchmean') * (temperature ** 2) / num_elements
    return loss

def calculate_loss(teacher_features, student_features, student_outputs_train, temperature, alpha):
    # 计算标准损失
    standard_loss = sum(loss for loss in student_outputs_train.values())
    # 计算蒸馏损失
    distill_loss_value = distillation_loss_feature(student_features, teacher_features, temperature)
    # 合并损失
    loss = alpha * standard_loss + (1 - alpha) * distill_loss_value
    return loss

