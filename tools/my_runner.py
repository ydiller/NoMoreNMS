# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
import torch
import mmcv
import wandb
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info
import copy
import logging
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod

import torch
from torch.optim import Optimizer
import mmcv
from mmcv.parallel import is_module_wrapper
from mmcv.runner.checkpoint import load_checkpoint
from mmcv.runner.dist_utils import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.log_buffer import LogBuffer
from mmcv.runner.priority import Priority, get_priority
from mmcv.runner.utils import get_time_str


@RUNNERS.register_module()
class MyRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None,
                 with_wandb=None):
        if batch_processor is not None:
            if not callable(batch_processor):
                raise TypeError('batch_processor must be callable, '
                                f'but got {type(batch_processor)}')
            warnings.warn(
                'batch_processor is deprecated, please implement '
                'train_step() and val_step() in the model instead.',
                DeprecationWarning)
            # raise an error is `batch_processor` is not None and
            # `model.train_step()` exists.
            if is_module_wrapper(model):
                _model = model.module
            else:
                _model = model
            if hasattr(_model, 'train_step') or hasattr(_model, 'val_step'):
                raise RuntimeError(
                    'batch_processor and model.train_step()/model.val_step() '
                    'cannot be both available.')
        else:
            assert hasattr(model, 'train_step')

        # check the type of `optimizer`
        if isinstance(optimizer, dict):
            for name, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(
                        f'optimizer must be a dict of torch.optim.Optimizers, '
                        f'but optimizer["{name}"] is a {type(optim)}')
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                f'optimizer must be a torch.optim.Optimizer object '
                f'or dict or None, but got {type(optimizer)}')

        # check the type of `logger`
        if not isinstance(logger, logging.Logger):
            raise TypeError(f'logger must be a logging.Logger object, '
                            f'but got {type(logger)}')

        # check the type of `meta`
        if meta is not None and not isinstance(meta, dict):
            raise TypeError(
                f'meta must be a dict or None, but got {type(meta)}')

        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta
        self.with_wandb = with_wandb
        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')

        self._max_epochs = max_epochs
        self._max_iters = max_iters
        # TODO: Redesign LogBuffer, it is not flexible and elegant enough
        self.log_buffer = LogBuffer()

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'MyHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')


    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.with_wandb:
                    wandb.log({"CE val loss": sum(self.log_buffer.val_history['loss_deepsets_ce'])/
                                              len(self.log_buffer.val_history['loss_deepsets_ce']),
                               "val ds_acc": sum(self.log_buffer.val_history['ds_acc'])/
                                  len(self.log_buffer.val_history['ds_acc']),
                               "val iou_error": sum(self.log_buffer.val_history['iou_error'])/len(self.log_buffer.val_history['iou_error']),
                               "val max score predictions": sum(self.log_buffer.val_history['ds_pred_on_max'])/
                                                            len(self.log_buffer.val_history['ds_pred_on_max'])
                               })
        else:  # single gpu
            if self.with_wandb:
                wandb.log({"CE val loss": sum(self.log_buffer.val_history['loss_deepsets_ce']) /
                                          len(self.log_buffer.val_history['loss_deepsets_ce']),
                           "val ds_acc": sum(self.log_buffer.val_history['ds_acc']) /
                                         len(self.log_buffer.val_history['ds_acc']),
                           "val iou_error": sum(self.log_buffer.val_history['iou_error']) / len(
                               self.log_buffer.val_history['iou_error']),
                           "val max score predictions": sum(self.log_buffer.val_history['ds_pred_on_max']) /
                                                        len(self.log_buffer.val_history['ds_pred_on_max'])})


    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='end2end_epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


# @RUNNERS.register_module()
# class Runner(MyRunner):
#     """Deprecated name of EpochBasedRunner."""
#
#     def __init__(self, *args, **kwargs):
#         warnings.warn(
#             'Runner was deprecated, please use EpochBasedRunner instead',
#             DeprecationWarning)
#         super().__init__(*args, **kwargs)
