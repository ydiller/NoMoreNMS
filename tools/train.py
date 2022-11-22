# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes, freeze_layers


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--with_wandb', type=int, default=0)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--dim_input', type=int, default=None)
    parser.add_argument('--dim_output', type=int, default=None)
    parser.add_argument('--l1_weight', type=float, default=None)
    parser.add_argument('--giou_weight', type=int, default=None)
    parser.add_argument('--ap_weight', type=int, default=None)
    parser.add_argument('--giou_coef', type=float, default=None)
    parser.add_argument('--set_size', type=int, default=None)
    parser.add_argument('--dim_hidden', type=int, default=None)
    parser.add_argument('--num_inds', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--bbox_prediction_type', type=str, default=None)
    parser.add_argument('--input_type', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='faster')
    parser.add_argument('--flips', type=int, default=2)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    if args.with_wandb:
        cfg.model.train_cfg.rcnn.with_wandb = args.with_wandb
    if args.lr:
        cfg.optimizer.lr = args.lr
    if args.wd is not None:
        cfg.optimizer.weight_decay = args.wd
    if args.bs:
        cfg.data.samples_per_gpu = args.bs
    if args.set_size:
        cfg.model.train_cfg.rcnn.deepsets_config.set_size = args.set_size
    if args.l1_weight is not None:
        cfg.model.train_cfg.rcnn.deepsets_config.l1_weight = args.l1_weight
    if args.giou_weight is not None:
        cfg.model.train_cfg.rcnn.deepsets_config.giou_weight = args.giou_weight
    if args.ap_weight is not None:
        cfg.model.train_cfg.rcnn.deepsets_config.ap_weight = args.ap_weight
    if args.giou_coef:
        cfg.model.train_cfg.rcnn.deepsets_config.giou_coef = args.giou_coef
    if args.dim_hidden:
        cfg.model.train_cfg.rcnn.deepsets_config.dim_hidden = args.dim_hidden
    if args.dim_output:
        cfg.model.train_cfg.rcnn.deepsets_config.dim_output = args.dim_output
    if args.num_inds:
        cfg.model.train_cfg.rcnn.deepsets_config.num_inds = args.num_inds
    if args.num_heads:
        cfg.model.train_cfg.rcnn.deepsets_config.num_heads = args.num_heads
    if args.num_heads:
        cfg.model.train_cfg.rcnn.deepsets_config.num_heads = args.num_heads
    if args.bbox_prediction_type:
        cfg.model.train_cfg.rcnn.deepsets_config.bbox_prediction_type = args.bbox_prediction_type
    if args.input_type:
        cfg.model.train_cfg.rcnn.deepsets_config.input_type = args.input_type
        if cfg.model.train_cfg.rcnn.deepsets_config.input_type == 'bbox':
            cfg.model.train_cfg.rcnn.deepsets_config.indim = 5
            cfg.model.train_cfg.rcnn.deepsets_config.dim_input = 5
        elif cfg.model.train_cfg.rcnn.deepsets_config.input_type == 'bbox_spacial':
            cfg.model.train_cfg.rcnn.deepsets_config.indim = 13
            cfg.model.train_cfg.rcnn.deepsets_config.dim_input = 13
        elif cfg.model.train_cfg.rcnn.deepsets_config.input_type == 'bbox_spacial_vis':
            cfg.model.train_cfg.rcnn.deepsets_config.indim = 13 + 1024
            cfg.model.train_cfg.rcnn.deepsets_config.dim_input = 256
        elif cfg.model.train_cfg.rcnn.deepsets_config.input_type == 'bbox_spacial_vis_label':
            cfg.model.train_cfg.rcnn.deepsets_config.indim = 13 + 1024 + 80
            cfg.model.train_cfg.rcnn.deepsets_config.dim_input = 256
    if args.dim_input:
        cfg.model.train_cfg.rcnn.deepsets_config.dim_input = args.dim_input
    if args.num_workers:
        cfg.data.workers_per_gpu = args.num_workers
    if args.flips == 2:
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        cfg.data.train.pipeline = [
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
    elif args.flips == 1:
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        cfg.data.train.pipeline = [
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal']),
                # dict(type='Rotate', prob=0.5, level=2),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ]
    else:
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        cfg.data.train.pipeline = [
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.0, direction=['horizontal']),
                # dict(type='Rotate', prob=0.5, level=2),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ]
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # val_dataset.pipeline = cfg.data.train.pipeline  # coco
        val_dataset.pipeline = cfg.data.train.dataset.pipeline  # pascalvoc
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    print('train deepsets only')

    model = freeze_layers(model, model_type=args.model_type)
    # get number of parameters
    # import numpy as np
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
