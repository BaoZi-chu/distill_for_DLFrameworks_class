_base_ = '../../mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py'
# 定义教师模型和学生模型
teacher_cfg = '../mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py'
teacher_weights= '../../checkpoint/yolox_distill/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
# student_cfg = _base_

# 修改数据集路径和类别
img_scale = (640, 640)  # width, height
data_root = '../../data/coco2017/'
dataset_type = 'CocoDataset'
backend_args = None
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]
train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,

        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args),
    pipeline=train_pipeline)
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator


model = dict(
    type='YOLOXDistillation',
    # backbone=dict(
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     norm_eval=True,
    #     init_cfg=None),

    temperature=4.0,
    alpha=0.7,
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.5,
        widen_factor=0.25,
        out_indices=(2, 3, 4),
        init_cfg=None),  # 不使用预训练权重
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[64, 128, 256],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=80,  # COCO 数据集有 80 个类别
        in_channels=128,
        feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=1,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='DistillHook',
        teacher_model_config=teacher_cfg,
        teacher_checkpoint=teacher_weights,
        priority=48
        ),
    # dict(
    #     type='EMAHook',
    #     ema_type='ExpMomentumEMA',
    #     momentum=0.0001,
    #     update_buffers=True,
    #     priority=49),

]

# training settings
max_epochs = 8
num_last_epochs = 1
interval = 1

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
