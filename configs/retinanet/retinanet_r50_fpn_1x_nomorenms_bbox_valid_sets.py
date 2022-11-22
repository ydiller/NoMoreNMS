deepsets_cfg = dict(
                type='DeepsetsHead',
                loss_mse=dict(
                    type='MSELoss', loss_weight=0.0),
                loss_ce=dict(
                    type='CrossEntropyLoss', loss_weight=1.0),
                # 'forward_box' |'forward_centroids' | 'forward_normalized'
                bbox_prediction_type='forward_normalized',
                # 'bbox' | 'bbox_spacial' | 'bbox_spacial_vis' | 'bbox_spacial_vis_label'
                input_type='bbox_spacial_vis',
                ds1=1000,
                ds2=600,
                ds3=300,
                set_size=32,
                reg=5,
                include_ds4=1,
                top_c=3,
                max_num=512,
                iou_thresh=0.5,
                indim=13 + 1024,  # 1117
                dim_input=128,
                dim_output=5,
                dim_hidden=16,
                num_inds=16,
                num_heads=1,
                l1_weight=0.02,
                giou_weight=2,
                ap_weight=1,
                giou_coef=0.3)
if deepsets_cfg['input_type'] == 'bbox':
    deepsets_cfg['indim'] = 5
    deepsets_cfg['dim_input'] = 5
elif deepsets_cfg['input_type'] == 'bbox_spacial':
    deepsets_cfg['indim'] = 13
    deepsets_cfg['dim_input'] = 13
elif deepsets_cfg['input_type'] == 'bbox_spacial_vis':
    deepsets_cfg['indim'] = 13 + 1024
    deepsets_cfg['dim_input'] = 256
elif deepsets_cfg['input_type'] == 'bbox_spacial_vis_label':
    deepsets_cfg['indim'] = 13 + 1024 + 80
    deepsets_cfg['dim_input'] = 256

# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        with_wandb=0,
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        with_wandb=0,
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/COCO/COCO2017/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    val_samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017_2k.json',
        img_prefix=data_root + 'images/train2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017_100.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1000, metric='bbox', by_epoch=False, save_best='auto')

# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    by_epoch=False,
    policy='poly',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    min_lr=0.00001)
runner = dict(type='MyRunner', max_epochs=12)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/data/pretrained_models/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
work_dir = 'work_dirs/retinanet_r50_fpn_1x_nomorenms_bbox_valid_sets'
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
