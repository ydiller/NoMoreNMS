deepsets_cfg = dict(
                type='DeepsetsHead',
                loss_mse=dict(
                    type='MSELoss', loss_weight=0.0),
                loss_ce=dict(
                    type='CrossEntropyLoss', loss_weight=1.0),
                ds1=1000,
                ds2=600,
                ds3=300,
                set_size=32,
                reg=5,
                include_ds4=1,
                top_c=3,
                max_num=512,
                iou_thresh=0.5,
                indim=13,  # 1117
                dim_input=13,
                dim_output=5,
                dim_hidden=16,
                num_inds=16,
                num_heads=4,
                l1_weight=2,
                giou_weight=2,
                giou_coef=0.3)

# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,  # original 1
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0)),
    roi_head=dict(
        type='DeepsetsRoIHeadBboxValidSets',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),
            loss_bbox=dict(type='L1Loss', loss_weight=0.0))),
        # deepsets_head=dict(
        #         type='DeepsetsHead',
        #         loss_mse=dict(
        #             type='MSELoss', loss_weight=0.0),
        #         loss_ce=dict(
        #             type='CrossEntropyLoss', loss_weight=1.0),
        #         ds1=1000,
        #         ds2=600,
        #         ds3=300,
        #         set_size=16,
        #         reg=5,
        #         include_ds4=1
        #     ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False,
            deepsets_config=deepsets_cfg,
            with_wandb=1),
        ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            # nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05),
            max_per_img=1000,
            deepsets_config=deepsets_cfg,
            with_wandb=1),
        )
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )

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
    # dict(type='Rotate', prob=0.5, level=2),
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
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',  # 50k, 10k, 5k, 2k, 500
        img_prefix=data_root + 'images/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017_100.json',  # 100, 2
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017_100.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline,
        samples_per_gpu=6))

evaluation = dict(interval=500, metric='bbox', by_epoch=False, save_best='auto')
# evaluation = dict(interval=1, metric='bbox')

# optimizer
# optimizer = dict(type='Adam', lr=0.00001, weight_decay=0.00000001)  # set selection
# optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0000001)  # bbox prediction l1. mse
# optimizer = dict(type='Adam', lr=0.00001, weight_decay=0.000001)  # bbox prediction giou, map 42
# optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.001)  # bbox prediction giou
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.00001)  # bbox prediction giou with normalization
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     by_epoch=False,
#     policy='poly',
#     warmup='linear',
#     warmup_iters=200,
#     warmup_ratio=0.001)
lr_config = dict(
    by_epoch=False,
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    # step=[1, 3, 5, 8],
    # gamma=0.1
    step=20000,
    gamma=0.75)
# runner = dict(type='EpochBasedRunner', max_epochs=12)
runner = dict(type='MyRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# custom hooks
# custom_imports = dict(imports=['mmdet.core.utils.custom_eval_hook'], allow_failed_imports=False)
custom_hooks = [
    # dict(type='CustomEvalHook', interval=5, metric='bbox', by_epoch=False),
    dict(type='NumClassCheckHook')
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/data/pretrained_models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# resume_from = 'work_dirs/faster_rcnn_r50_fpn_1x_nomorenms/epoch_1_08.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
work_dir = 'work_dirs/faster_rcnn_r50_fpn_1x_nomorenms_giou'
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

