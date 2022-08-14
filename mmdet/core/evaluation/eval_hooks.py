# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import os.path as osp
import torch
import mmcv
import wandb
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class EvalHook(BaseEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(EvalHook, self).__init__(*args, **kwargs)

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):  # original
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)  # original - key_score is None
        if self.save_best:
            self._save_ckpt(runner, key_score)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if runner.with_wandb:
                    wandb.log({"mAP": key_score['bbox_mAP']})
        else:  # single gpu
            if runner.with_wandb:
                if self.save_best:
                    wandb.log({"mAP": key_score})
                else:
                    wandb.log({"mAP": key_score['bbox_mAP']})

    def _save_ckpt(self, runner, key_score):
        """Save the best checkpoint.

        It will compare the score according to the compare function, write
        related information (best score, best checkpoint path) and save the
        best checkpoint into ``work_dir``.
        """
        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        best_score = runner.meta['hook_msgs'].get(
            'best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            if runner.with_wandb:
                wandb.log({"best/mAP": best_score})
            runner.meta['hook_msgs']['best_score'] = best_score

            if self.best_ckpt_path and self.file_client.isfile(
                    self.best_ckpt_path):
                self.file_client.remove(self.best_ckpt_path)
                runner.logger.info(
                    (f'The previous best checkpoint {self.best_ckpt_path} was '
                     'removed'))

            best_ckpt_name = f'best_{self.key_indicator}_{best_score}_{current}.pth'
            self.best_ckpt_path = self.file_client.join_path(
                self.out_dir, best_ckpt_name)
            runner.meta['hook_msgs']['best_ckpt'] = self.best_ckpt_path

            runner.save_checkpoint(
                self.out_dir, best_ckpt_name, create_symlink=False)
            runner.logger.info(
                f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(
                f'Best {self.key_indicator} is {best_score:0.4f} '
                f'at {cur_time} {cur_type}.')

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            # If the performance of model is pool, the `eval_res` may be an
            # empty dict and it will raise exception when `self.save_best` is
            # not None. More details at
            # https://github.com/open-mmlab/mmdetection/issues/6265.
            if not eval_res:
                warnings.warn(
                    'Since `eval_res` is an empty dict, the behavior to save '
                    'the best checkpoint will be skipped in this evaluation.')
                return None

            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return eval_res
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.
class DistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(DistEvalHook, self).__init__(*args, **kwargs)

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    # def _do_evaluate(self, runner):  # original
    #     """perform evaluation and save ckpt."""
    #     # Synchronization of BatchNorm's buffer (running_mean
    #     # and running_var) is not supported in the DDP of pytorch,
    #     # which may cause the inconsistent performance of models in
    #     # different ranks, so we broadcast BatchNorm's buffers
    #     # of rank 0 to other ranks to avoid this.
    #     if self.broadcast_bn_buffer:
    #         model = runner.model
    #         for name, module in model.named_modules():
    #             if isinstance(module,
    #                           _BatchNorm) and module.track_running_stats:
    #                 dist.broadcast(module.running_var, 0)
    #                 dist.broadcast(module.running_mean, 0)
    #
    #     if not self._should_evaluate(runner):
    #         return
    #
    #     tmpdir = self.tmpdir
    #     if tmpdir is None:
    #         tmpdir = osp.join(runner.work_dir, '.eval_hook')
    #
    #     from mmdet.apis import multi_gpu_test
    #     results = multi_gpu_test(
    #         runner.model,
    #         self.dataloader,
    #         tmpdir=tmpdir,
    #         gpu_collect=self.gpu_collect)
    #     if runner.rank == 0:
    #         print('\n')
    #         runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
    #         key_score = self.evaluate(runner, results)
    #
    #         if self.save_best:
    #             self._save_ckpt(runner, key_score)


    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        # temporary cancel evaluation to prevent crush.

        # tmpdir = self.tmpdir
        # if tmpdir is None:
        #     tmpdir = osp.join(runner.work_dir, '.eval_hook')
        # if runner.rank == 0:
        #     print('\n')
        #     from mmdet.apis import single_gpu_test
        #     results = single_gpu_test(runner.model, self.dataloader, show=False)
        #     runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        #     key_score = self.evaluate(runner, results)
        #     if self.save_best:
        #         self._save_ckpt(runner, key_score)


