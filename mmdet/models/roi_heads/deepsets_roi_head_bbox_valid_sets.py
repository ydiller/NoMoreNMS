# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import wandb
import datetime
import cv2 as cv
import torch.nn.functional as F
import numpy as np
import sys
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from tools.set_transformer2 import SetTransformer
from .deepsets_forward import *

DS_TYPE = 'hard'  # soft | hard


class PermEqui2_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        self.weight = self.Gamma.weight
        self.bias = self.Gamma.bias

    def forward(self, x):
        # if x.ndim == 1:
        #     x = x.unsqueeze(0)
        xm = x.mean(0, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        # return self.Gamma(x) - self.Lambda(xm)
        return x


class SoftCrossEntropyLoss():
    def __init__(self):
        super().__init__()
        # self.weights = weights

    def forward(self, y_hat, y, iou_thr):
        p = F.log_softmax(y_hat, 0)
        # p = F.softmax(y_hat, 0)
        # w_labels = self.weights*y
        # loss = -(y[valid_inds]*p[valid_inds].T).mean()
        # loss = -(p[valid_inds]).mean()
        # loss = -torch.log(torch.sum(p[valid_mask]))
        # valid_mask = y > iou_thr
        # valid_inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        # valid_mask = valid_mask.type(torch.float)/len(valid_mask.nonzero(as_tuple=False))
        # loss = -torch.mean(valid_mask*p.T)
        loss = -torch.mean(y * p.T)
        return loss

    # def forward(self, y_hat, y, iou_thr):
    #    p = F.log_softmax(y*y_hat.T, 1)
    #    loss = -p[torch.argmax(y_hat)]
    #    return loss


class GIOUuLoss():
    def __init__(self):
        super().__init__()

    def forward(self, pr_bboxes, gt_bboxes, giou_coef=1.0, reduction='mean'):
        """
    gt_bboxes: tensor (-1, 4) xyxy
    pr_bboxes: tensor (-1, 4) xyxy
    loss proposed in the paper of giou
    """
        gt_area = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        pr_area = (pr_bboxes[:, 2] - pr_bboxes[:, 0]) * (pr_bboxes[:, 3] - pr_bboxes[:, 1])

        # iou
        lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
        rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
        TO_REMOVE = 1
        # wh = (rb - lt + TO_REMOVE).clamp(min=0)  # original
        wh = (rb - lt).clamp(min=0)  # width height of intersection
        inter = wh[:, 0] * wh[:, 1]
        union = gt_area + pr_area - inter
        iou = inter / union
        # enclosure
        lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
        rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
        # wh = (rb - lt + TO_REMOVE).clamp(min=0)
        wh = (rb - lt).clamp(min=0)
        enclosure = wh[:, 0] * wh[:, 1]
        giou = iou - giou_coef * (
                    enclosure - union) / (enclosure + 0.00001) # not original - multiply by 0.4 to reduce inner box phenomena
        loss = 1. - giou
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'none':
            pass
        if loss < 0:
            print(f"negative giou loss. pred = {pr_bboxes}\n gt= {gt_bboxes} \n iou: {iou} \n "
                  f"enclusure: {enclosure} \n union= {union}")
            loss = 1.
        return loss


@HEADS.register_module()
class DeepsetsRoIHeadBboxValidSets(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 reg=0.8,
                 include_ds4=1,
                 loss_mse=dict(
                     type='MSELoss',
                     loss_weight=1.0),
                 loss_l1=dict(
                     type='L1Loss',
                     loss_weight=1.0),
                 loss_ce=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_ap=dict(
                     type='APLoss')):
        super(DeepsetsRoIHeadBboxValidSets, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.reg = reg
        self.include_ds4 = include_ds4
        self.loss_mse = build_loss(loss_mse)
        self.loss_ce = build_loss(loss_ce)
        self.loss_l1 = build_loss(loss_l1)
        self.loss_ap = build_loss(loss_ap)
        self.loss_giou = GIOUuLoss()
        self.set_statistics = {
            "sets": 0,
            "valid_sets": 0,
            "gt_num": 0,
            "num_imgs": 0,
            "double_gt": 0}
        # self.std_statistics = {
        #     "x1": 0,
        #     "y1": 0,
        #     "x2": 0,
        #     "y2": 0,
        #     "counter": 0}
        # self.ds1 = PermEqui2_mean(indim, ds1)
        # self.ds2 = PermEqui2_mean(ds1, ds2)
        # self.ds3 = PermEqui2_mean(ds2, ds3)
        # if not include_ds4:
        #     self.ds4 = PermEqui2_mean(ds3, 1)
        # else:
        # self.ds4 = PermEqui2_mean(ds3, int(ds3/2))
        # self.bn4 = torch.nn.BatchNorm1d(int(ds3/2))
        # self.ds5 = PermEqui2_mean(int(ds3/2), 1)
        self.curr_cfg = train_cfg if train_cfg else test_cfg
        self.bbox_prediction_type = globals()[self.curr_cfg.deepsets_config.bbox_prediction_type]
        self.input_type = self.curr_cfg.deepsets_config.input_type
        self.set_size = self.curr_cfg.deepsets_config.set_size
        self.indim = self.curr_cfg.deepsets_config.indim
        self.dim_input = self.curr_cfg.deepsets_config.dim_input
        self.dim_output = self.curr_cfg.deepsets_config.dim_output
        self.l1_weight = self.curr_cfg.deepsets_config.l1_weight
        self.giou_weight = self.curr_cfg.deepsets_config.giou_weight
        self.ap_weight = self.curr_cfg.deepsets_config.ap_weight
        self.giou_coef = self.curr_cfg.deepsets_config.giou_coef
        self.ln1 = torch.nn.Linear(self.indim, self.dim_input)
        self.dim_hidden = self.curr_cfg.deepsets_config.dim_hidden
        self.num_inds = self.curr_cfg.deepsets_config.num_inds
        self.num_heads = self.curr_cfg.deepsets_config.num_heads
        # self.ln2 = torch.nn.Linear(1024, 6)
        # self.dropout = nn.Dropout(0.25)
        # self.ln3 = torch.nn.Linear(256, 128)
        # self.bn1 = torch.nn.BatchNorm1d(128)
        # self.bn2 = torch.nn.BatchNorm1d(256)
        # self.bn3 = torch.nn.BatchNorm1d(128)
        # self.ds1 = PermEqui2_mean(self.dim_input, self.dim_output)
        # self.ds2 = PermEqui2_mean(128, 128)
        # self.ds3 = PermEqui2_mean(128, 1)
        # self.set_transformer = SetTransformer(dim_input=self.dim_input, num_outputs=self.dim_output, dim_output=1, num_inds=16, dim_hidden=128,  # original
        #                                       num_heads=4, ln=False, mode="dense")
        # self.set_transformer = SetTransformer(dim_input=self.dim_input, num_outputs=self.dim_output, dim_output=1, num_inds=16, dim_hidden=128,  # original
        #                                       num_heads=4, ln=False, mode="dense")
        self.set_transformer = SetTransformer(dim_input=self.dim_input, num_outputs=self.dim_output, dim_output=1,
                                              num_inds=self.num_inds, dim_hidden=self.dim_hidden,
                                              num_heads=self.num_heads, ln=False, mode="dense")  # 13 features
        # self.set_transformer2 = SetTransformer(dim_input=5, num_outputs=1, dim_output=1,
        #         num_inds=self.num_inds, dim_hidden=5, num_heads=self.num_heads, ln=False, mode="dense")
        # self.ln4 = torch.nn.Linear(self.dim_output, 1)
        self.ln5 = torch.nn.Linear(self.dim_output, 4)
        self.ln6 = torch.nn.Linear(self.set_size, 1)
        self.ln7 = torch.nn.Linear(5, 1)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'],)
        return outs

    def register_grad(self, grad_input):
        print('a')
        assert all(t is None or torch.all(~torch.isnan(t)) for t in
                   grad_input), f" grad_input={grad_input}"

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            # losses.update(bbox_results['loss_bbox'])
            # if torch.distributed.is_initialized():
            #     if torch.distributed.get_rank() == 0 and self.curr_cfg.with_wandb:
            #         wandb.log({"Bbox loss": bbox_results['loss_bbox']['loss_bbox'],
            #                    "Class CE loss:": bbox_results['loss_bbox']['loss_cls'],
            #                    "Acc": bbox_results['loss_bbox']['acc']})
            # else:
            #     if self.curr_cfg.with_wandb:
            #         wandb.log({"Bbox loss": bbox_results['loss_bbox']['loss_bbox'],
            #                    "Class CE loss:": bbox_results['loss_bbox']['loss_cls'],
            #                    "Acc": bbox_results['loss_bbox']['acc']})
        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
        ######## deepsets ######
        # torch.autograd.set_detect_anomaly(True)
        num_proposals_per_img = tuple(len(res.bboxes) for res in sampling_results)
        rois = bbox2roi([res.bboxes for res in sampling_results])
        feats = bbox_results["last_layer_feats"]
        rois = rois.split(num_proposals_per_img, 0)
        scores = bbox_results["cls_score"].split(num_proposals_per_img, 0)
        bboxes = bbox_results["bbox_pred"].split(num_proposals_per_img, 0)
        feats = feats.split(num_proposals_per_img, 0)
        device = bboxes[0].get_device()
        ds_cfg = self.curr_cfg.deepsets_config
        loss_deepsets = dict()
        loss_deepsets['loss_deepsets_total'] = torch.zeros(1, requires_grad=True).cuda(device=device)
        loss_deepsets['giou'] = torch.zeros(1, requires_grad=True).cuda(device=device)
        loss_deepsets['l1'] = torch.zeros(1, requires_grad=True).cuda(device=device)
        loss_deepsets['bce'] = torch.zeros(1, requires_grad=True).cuda(device=device)
        loss_deepsets['valid_set_acc'] = torch.zeros(1, requires_grad=True).cuda(device=device)
        valid_img_num = 0
        for i in range(num_imgs):
            bbox, score = self.bbox_head.get_bboxes(
                rois[i],
                scores[i],
                bboxes[i],
                img_metas[i]['img_shape'],
                img_metas[i]['scale_factor'],
                rescale=False,
                cfg=None)
            # bbox, score = bbox.detach(), score.detach()
            sets, preds, predicted_scores, set_labels, set_bboxes, set_scores, centroids_per_set, normalization_data = \
                self.bbox_prediction_type(self, bbox, score, feats[i], self.input_type,
                                          img_shape=img_metas[i]['img_shape'],gt_labels=gt_labels[i], ds_cfg=ds_cfg,
                                          device=device, score_thr=0.05)
            # valid_preds, valid_set_bboxes, valid_set_scores, valid_ious, valid_centroids, valid_normalization_data, \
            # score_target, gt_box_per_set, gt_valid_sets = \
            #     self._get_target(sets, set_labels, set_bboxes, centroids_per_set, normalization_data, gt_bboxes[i],
            #                      gt_labels[i], preds, device=device)
            gt_box_per_set, gt_valid_sets, score_target, gt_valid_preds_no_matching\
                = self._get_target_refactored(gt_bboxes[i], gt_labels[i], preds, predicted_scores, set_labels, device=device)
            loss_deepsets_i = self._loss(preds, predicted_scores, gt_box_per_set, gt_valid_sets,
                                         score_target, set_labels, gt_valid_preds_no_matching,
                                         img_shape=img_metas[i]['img_shape'], device=device,
                                         lambda_l1=self.l1_weight, lambda_iou=self.giou_weight, lambda_ap=self.ap_weight)
            if loss_deepsets_i['loss_deepsets_total'] is not None:
                loss_deepsets['loss_deepsets_total'] += loss_deepsets_i['loss_deepsets_total']
                loss_deepsets['giou'] += loss_deepsets_i['giou']
                loss_deepsets['l1'] += loss_deepsets_i['l1']
                loss_deepsets['bce'] += loss_deepsets_i['bce']
                loss_deepsets['valid_set_acc'] += loss_deepsets_i['valid_set_acc']
                valid_img_num += 1
            # wandb bounding box
            # pred_boxes_log = []
            # non_valid_boxes_log = []
            # gt_boxes_log = []
            # img = cv.imread(img_metas[i]['filename'])
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # img = cv.resize(img, (img_metas[i]['img_shape'][1], img_metas[i]['img_shape'][0]))
            # for _s, _set in enumerate(valid_set_bboxes):
            #     for _b, b in enumerate(_set):
            #         box_data = {
            #             "position": {
            #                 "minX": b[0].item(),
            #                 "maxX": b[2].item(),
            #                 "minY": b[1].item(),
            #                 "maxY": b[3].item()},
            #             "class_id": _s,
            #             "domain": "pixel",
            #             "box_caption": f'{_s} ({valid_set_scores[_s][_b].item():.2f})',
            #             # "box_caption:": "%d (%.2f)" % (_s, sets[_s][_b][-1].item()),
            #             "scores": {"score": valid_set_scores[_s][_b].item(),
            #                        "set id": _s}
            #         }
            #         pred_boxes_log.append(box_data)
            # for _s, _set in enumerate(non_valid_set_bboxes):
            #     for _b, b in enumerate(_set):
            #         box_data = {
            #             "position": {
            #                 "minX": b[0].item(),
            #                 "maxX": b[2].item(),
            #                 "minY": b[1].item(),
            #                 "maxY": b[3].item()},
            #             "class_id": _s,
            #             "domain": "pixel",
            #             "box_caption": f'{_s} ({non_valid_set_scores[_s][_b].item():.2f})',
            #             # "box_caption:": "%d (%.2f)" % (_s, sets[_s][_b][-1].item()),
            #             "scores": {"score": non_valid_set_scores[_s][_b].item(),
            #                        "set id": _s}
            #         }
            #         non_valid_boxes_log.append(box_data)
            # for _b, b in enumerate(gt_bboxes[i]):
            #     box_data = {
            #         "position": {
            #             "minX": b[0].item(),
            #             "maxX": b[2].item(),
            #             "minY": b[1].item(),
            #             "maxY": b[3].item()},
            #         "class_id": gt_labels[i][_b].item(),
            #         "domain": "pixel",
            #         "box_caption": f'{gt_labels[i][_b].item()}'
            #     }
            #     gt_boxes_log.append(box_data)
            # if self.curr_cfg.with_wandb:
            #     wandb.log({"image": wandb.Image(img,
            #                          boxes={"predictions": {"box_data": pred_boxes_log},
            #                                 "gts": {"box_data": gt_boxes_log},
            #                                 "non valid boxes": {"box_data": non_valid_boxes_log}})})
            ###
        for key, value in loss_deepsets.items():
            if valid_img_num:
                loss_deepsets[key] = value / valid_img_num
            else:
                loss_deepsets[key] = value / num_imgs  # no loss, prevent dividing by zero
        losses.update(loss_deepsets)

        ### temp
        # p = [torch.argmax(x) for x in preds]
        # z = [torch.argmax(x[:, -1]) for x in sets]
        # eq = [p == z for p, z in zip(p, z)]
        # max_score_rate = float(sum(eq)) / (len(eq)+0.000001)
        ###

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0 and self.curr_cfg.with_wandb:
                wandb.log({"Total loss": loss_deepsets["loss_deepsets_total"],
                           "GIOU loss": loss_deepsets["giou"],
                           "L1 loss": loss_deepsets["l1"],
                           "Score L1 loss": loss_deepsets["bce"],
                           "Valid Set Acc": loss_deepsets["valid_set_acc"],
                           "Preds mean": torch.mean(abs(preds))
                           })
        else:
            if self.curr_cfg.with_wandb:
                wandb.log({"Total loss": loss_deepsets["loss_deepsets_total"],
                           "GIOU loss": loss_deepsets["giou"],
                           "L1 loss": loss_deepsets["l1"],
                           "Score L1 loss": loss_deepsets["bce"],
                           "Valid Set Acc": loss_deepsets["valid_set_acc"],
                           "Preds mean": torch.mean(abs(preds))})
        # "max score rate": max_score_rate})

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, last_layer_feats = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, last_layer_feats=last_layer_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, sets = self.simple_test_deepsets_bbox_valid_sets(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results, det_bboxes, det_labels, sets
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
            -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels

    # def normalize(self, vector):
    #     return torch.div(vector, torch.max(vector + sys.float_info.epsilon, dim=0)[0])

    def normalize(self, vector):
        if vector.size(0) > 1:
            # vector -= torch.mean(vector, 0).repeat(vector.size(0), 1)
            # std = torch.std(vector, 0).repeat(vector.size(0), 1)
            mean = torch.mean(vector, 0)
            # std = torch.std(vector, 0) + 0.000001
            std = (vector - torch.mean(vector, 0)).norm(p=2, dim=0) + 0.000001
            # print(std)
            vector = torch.div(vector - mean, std)  # sys.float_info.epsilon
            # print(std.detach().cpu())
        return vector

    def input_preprocessing(self, bboxes, img_shape, device):
        x1 = bboxes[:, 0].unsqueeze(1) / img_shape[1]
        x2 = bboxes[:, 2].unsqueeze(1) / img_shape[1]
        y1 = bboxes[:, 1].unsqueeze(1) / img_shape[0]
        y2 = bboxes[:, 3].unsqueeze(1) / img_shape[0]
        width = (x2 - x1)
        height = (y2 - y1)
        aspect_ratio = torch.div(width, height + 0.000001)  # sys.float_info.epsilon)
        area = width * height
        # set_area_mean = area/torch.mean(area)
        # set_area_sum = area/torch.sum(area)
        # set_aspect_mean = aspect_ratio/torch.mean(aspect_ratio)
        # set_aspect_sum = aspect_ratio/torch.sum(aspect_ratio)
        # one_hot_classes = torch.zeros((1, num_classes)).repeat(len(_set), 1).cuda(device=device)
        # one_hot_classes[:, c] = 1
        centroids = torch.cat([x1 + width / 2, y1 + height / 2], dim=1).cuda(device=device)
        return x1, x2, y1, y2, width, height, aspect_ratio, area, centroids

    # def set_forward(self, x):
    # x = F.elu(self.ds1(x))
    # x = F.elu(self.ds2(x))
    # x = F.elu(self.ds3(x))
    # if not self.include_ds4:
    #     pred = F.elu(self.ds4(x))
    # else:
    #     x = F.elu(self.ds4(x))
    #     pred = self.ds5(x)
    # return pred

    def set_forward2(self, x):
        x = self.bn1(F.elu(self.ln1(x)))
        x = self.bn2(F.elu(self.ln2(x)))
        x = self.bn3(F.elu(self.ln3(x)))
        # x = F.elu(self.ds1(x))
        # x = F.elu(self.ds2(x))
        pred = self.ds3(x)
        return pred

    def set_forward3(self, x):
        x = F.elu(self.ln1(x))
        # x = F.elu(self.ln2(x))
        # x = F.elu(self.ln3(x))
        x = F.elu(self.set_transformer(x))
        pred = self.ln4(x)
        return pred

    def set_forward4(self, x):
        x = F.elu(self.set_transformer(x))
        pred = self.ln6(x.T)
        pred[-1] = torch.sigmoid(pred[-1])
        return pred

    def set_forward5(self, x):
        # for bbox normalization
        if self.input_type == 'bbox_spacial_vis' or self.input_type == 'bbox_spacial_vis_label':
            x = F.elu(self.ln1(x))
        pred = self.set_transformer(x)
        pred[-1] = torch.sigmoid(pred[-1])
        return pred

    def set_forward6(self, x):
        # for bbox predictions and centroids
        if self.input_type == 'bbox_spacial_vis' or self.input_type == 'bbox_spacial_vis_label':
            x = F.elu(self.ln1(x))
        pred = self.set_transformer(x)
        pred = torch.sigmoid(pred)
        return pred


    def _get_target_refactored(self,
                    gt_bboxes,
                    gt_labels,
                    preds,
                    scores,
                    set_labels,
                    device=0):
        """
        matching preds and gts.
        :returns: gt_target: box ccordinates of matched gt for each pred.
                  iou_target: iou(pred, matched_gt)
        """
        # for each prediction, caclculate iou with gt.
        # max iou(pred, gt) is the target of the prediction. if already matched, look for another gt.
        # train only on predictions with matched gt, and iou(pred, gt) > 0.5.
        zeros_float = torch.zeros(len(preds), dtype=torch.float32).cuda(device=device)
        zeros_int = torch.zeros(len(preds), dtype=torch.int64).cuda(device=device)
        gt_box_per_pred = torch.zeros((len(preds), 4), dtype=torch.float32).cuda(device=device)
        gt_valid_preds, gt_valid_preds_no_matching, iou_target = zeros_int, zeros_int, zeros_float
        if len(preds) == 0:
            return gt_box_per_pred, gt_valid_preds, iou_target, gt_valid_preds_no_matching
        iou_mat = bbox_overlaps(preds[:, :4], gt_bboxes)
        for c in torch.unique(set_labels):
            class_preds = preds[set_labels == c]
            class_gts = gt_bboxes[gt_labels == c]
            gtm_c = torch.zeros(len(class_gts))
            c_iou_mat = iou_mat[set_labels == c][:, gt_labels == c]
            for dind, d in enumerate(class_preds):
                m = -1
                p = torch.where(set_labels == c)[0][dind]
                iou = 0
                for gind, g in enumerate(class_gts):
                    gt_valid_preds_no_matching[p] = c_iou_mat[dind, gind] > 0.5
                    if gtm_c[gind] == 1:  # gt already matched
                        continue
                    if c_iou_mat[dind, gind] < iou:  # better match made
                        continue
                    iou = c_iou_mat[dind, gind]
                    m = gind # pred_index here
                gt_valid_preds[p] = iou > 0.5
                iou_target[p] = iou
                if m > -1:
                    gt_box_per_pred[p] = class_gts[m]
                    gtm_c[m] = 1

        return gt_box_per_pred, gt_valid_preds, iou_target, gt_valid_preds_no_matching

    def _loss(self, predicted_boxes, predicted_scores, gt_box_per_set, gt_valid_sets, score_target, set_labels,
              gt_valid_preds_no_matching, img_shape=None, device=0, lambda_iou=2, lambda_l1=0.5, lambda_ap=2):
        losses = dict()
        assert predicted_boxes is not None, "pred is None"
        _l1_loss = torch.zeros(1).cuda(device=device)
        _giou_loss = torch.zeros(1).cuda(device=device)

        if len(predicted_boxes) == 0 or not torch.sum(gt_valid_sets):
            losses['loss_deepsets_total'] = None
        else:
            _giou_loss = self.loss_giou.forward(predicted_boxes[gt_valid_preds_no_matching > 0, :4], gt_box_per_set[gt_valid_preds_no_matching > 0],
                                                self.giou_coef)
            _l1_loss = self.loss_l1.forward(predicted_boxes[gt_valid_preds_no_matching > 0, :4], gt_box_per_set[gt_valid_preds_no_matching > 0])

            ### full map
            _ap_loss_l = []
            # set_labels = torch.FloatTensor(set_labels).cuda(device=device)
            for label in torch.unique(set_labels):
                _ap_loss_iou = []
                label_mask = set_labels == label
                for iou in range(50, 100, 5):
                    iou_mask = score_target > iou/100
                    if torch.sum(iou_mask[label_mask]) > 0:
                        _ap_loss_iou.append(self.loss_ap.forward(predicted_scores[label_mask].unsqueeze(0), iou_mask[label_mask].unsqueeze(0)))
                    else:
                        _ap_loss_iou.append(torch.ones(1).cuda(device=device))
                _ap_loss_l.append(sum(_ap_loss_iou)/len(_ap_loss_iou))
            _ap_loss = sum(_ap_loss_l)/len(_ap_loss_l)
            losses['loss_deepsets_total'] = lambda_iou * _giou_loss \
                                            + lambda_l1 * _l1_loss / (np.mean(img_shape[0] + img_shape[1])) \
                                            + lambda_ap * _ap_loss
            losses['giou'] = lambda_iou * _giou_loss
            losses['l1'] = lambda_l1 * _l1_loss / (np.mean(img_shape[0] + img_shape[1]))
            losses['bce'] = lambda_ap * _ap_loss
            losses['valid_set_acc'] = torch.sum(torch.eq(predicted_scores>0.5, gt_valid_sets))/float(len(gt_valid_sets))
        return losses

    def _forward_deepsets_old(self, multi_bboxes, cls_score, last_layer_feats,
                          mode='train', ds_cfg=None, img_shape=None, gt_labels=None, device=0, score_thr=0.05):
        """
        Args:
            labels: bbox labels (assigned by comparing to gt)
        """
        num_classes = cls_score.shape[1] - 1
        preds = []
        inputs = []
        set_bboxes = []
        set_scores = []
        input_labels = []
        centroids_per_set = []
        normalization_data = []
        top_c = ds_cfg['top_c']
        max_num = ds_cfg['max_num']
        iou_threshold = ds_cfg['iou_thresh']
        if mode == 'train':
            classes = torch.unique(gt_labels)
        else:
            # classes = torch.unique(torch.topk(cls_score[:, :-1], top_c, dim=1)[1])  # top_classes_on_set
            classes = torch.range(0, num_classes - 1, dtype=int).cuda(device=device)
        zero = torch.zeros(1).cuda(device=device)
        one = torch.ones(1).cuda(device=device)
        for c in classes:  # c in 0 to data length - 1
            sets = []
            bboxes = multi_bboxes[:, c * 4:(c + 1) * 4]
            scores = cls_score[:, c]
            valid_mask = scores > score_thr
            valid_inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
            bboxes, scores = bboxes[valid_inds], scores[valid_inds]
            scores, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            feats = last_layer_feats[inds]
            ious = bbox_overlaps(bboxes, bboxes)
            is_clustered = torch.ones(ious.shape[0]).cuda(device=device)
            for j, row in enumerate(ious):
                _set = []
                if is_clustered[j] == 1:
                    selected_indices = torch.nonzero((row > iou_threshold) * is_clustered, as_tuple=False).squeeze()
                    is_clustered *= torch.where(row > iou_threshold, zero, one)  # selected indices = 0
                    if selected_indices.ndim == 0:
                        selected_indices = selected_indices.unsqueeze(0)
                    sets.append(selected_indices)
            x1_full, x2_full, y1_full, y2_full, width_full, height_full, aspect_ratio_full, area_full, centroids_full = \
                self.input_preprocessing(bboxes, img_shape, device)
            for s, _set in enumerate(sets):
                if mode == 'train' and len(_set) == 1:  # were set includes only one object
                    continue
                # if max(scores[sets[s]]) < 0.2:  # to enhance training
                #     continue
                # x1 = bboxes[sets[s], 0].unsqueeze(1)/img_shape[1]
                # x2 = bboxes[sets[s], 2].unsqueeze(1)/img_shape[1]
                # y1 = bboxes[sets[s], 1].unsqueeze(1)/img_shape[0]
                # y2 = bboxes[sets[s], 3].unsqueeze(1)/img_shape[0]
                # width = (x2 - x1)
                # height = (y2 - y1)
                # aspect_ratio = torch.div(width, height+sys.float_info.epsilon)
                # area = width*height
                # set_area_mean = area/torch.mean(area)
                # set_area_sum = area/torch.sum(area)
                # set_aspect_mean = aspect_ratio/torch.mean(aspect_ratio)
                # set_aspect_sum = aspect_ratio/torch.sum(aspect_ratio)
                # one_hot_classes = torch.zeros((1, num_classes)).repeat(len(_set), 1).cuda(device=device)
                # one_hot_classes[:, c] = 1
                # centroids = torch.cat([x1+width/2, y1+height/2], dim=1).cuda(device=device)
                x1 = x1_full[sets[s]]
                x2 = x2_full[sets[s]]
                y1 = y1_full[sets[s]]
                y2 = y2_full[sets[s]]
                width = width_full[sets[s]]
                height = height_full[sets[s]]
                aspect_ratio = aspect_ratio_full[sets[s]]
                area = area_full[sets[s]]
                centroids = centroids_full[sets[s]]
                set_dist_mean = torch.mean(torch.cdist(centroids, centroids, p=2), 1).unsqueeze(1)
                set_dist_sum = torch.sum(torch.cdist(centroids, centroids, p=2), 1).unsqueeze(1)
                set_ious_mean = torch.mean(ious[sets[s]][:, sets[s]], 1).unsqueeze(1)
                set_ious_sum = torch.sum(ious[sets[s]][:, sets[s]], 1).unsqueeze(1)
                current_scores = scores[sets[s]].unsqueeze(1)
                # zero mean center points
                # x1 -= torch.mean(centroids, 0)[0]
                # x2 -= torch.mean(centroids, 0)[0]
                # y1 -= torch.mean(centroids, 0)[1]
                # y2 -= torch.mean(centroids, 0)[1]
                # if mode == 'train' and (torch.std(x1) < 1e-5 or torch.std(y1) < 1e-5 or torch.std(x2) < 1e-5 or torch.std(y2) < 1e-5 \
                #         or torch.std(width) < 1e-5 or torch.std(height) < 1e-5 or torch.std(aspect_ratio) < 1e-5 \
                #         or torch.std(area) < 1e-5 or torch.std(scores[sets[s]].unsqueeze(1)) < 1e-5):
                # #     # print(f'len set: \n {len(_set)} \n zero std at y1: \n {y1.detach().cpu()} \n  x1:\n {x1.detach().cpu()}')
                #     print(f'zero std')
                #     continue
                # if mode == 'train' and (torch.std(x1) < 1e-5 or torch.std(y1) < 1e-5 or torch.std(x2) < 1e-5 or torch.std(y2) < 1e-5):
                #     continue
                normalization_data_per_set = {
                    'x1_mean': torch.mean(x1),
                    'x1_std': (x1 - torch.mean(x1, 0)).norm(p=2, dim=0),  # torch.std(x1),
                    'y1_mean': torch.mean(y1),
                    'y1_std': (y1 - torch.mean(y1, 0)).norm(p=2, dim=0),  # torch.std(y1),
                    'x2_mean': torch.mean(x2),
                    'x2_std': (x2 - torch.mean(x2, 0)).norm(p=2, dim=0),  # torch.std(x2),
                    'y2_mean': torch.mean(y2),
                    'y2_std': (y2 - torch.mean(y2, 0)).norm(p=2, dim=0),  # torch.std(y2),
                    'scores_mean': torch.mean(current_scores),
                    'scores_std': (current_scores - torch.mean(current_scores, 0)).norm(p=2, dim=0),
                    'set_size': len(_set),
                }
                # self.std_statistics['x1'] += torch.mean(x1)
                # self.std_statistics['y1'] += torch.mean(y1)
                # self.std_statistics['x2'] += torch.mean(x2)
                # self.std_statistics['y2'] += torch.mean(y2)
                # self.std_statistics['counter'] += 1
                # zero mean by coordinate
                # x1 = x1 - torch.mean(x1)
                # x2 = x2 - torch.mean(x2)
                # y1 = y1 - torch.mean(y1)
                # y2 = y2 - torch.mean(y2)
                # input = torch.cat([x1, y1, x2, y2, feats[sets[s]], width, height, aspect_ratio, area, set_ious_mean,
                #                    set_ious_sum, set_dist_mean, set_dist_sum, one_hot_classes, scores[sets[s]].unsqueeze(1)], dim=1)

                # input = torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, scores[sets[s]].unsqueeze(1)], dim=1)
                # print('y1: \n', y1.detach().cpu(), '\n y1 std: \n', torch.std(y1) )
                # print(torch.std(x1).detach().cpu(), torch.std(y1).detach().cpu(), torch.std(x2).detach().cpu(), torch.std(y2).detach().cpu())
                # print(torch.mean(x1).detach().cpu())
                # print((x1 - torch.mean(x1, 0)).norm(p=2, dim=0).detach().cpu())
                # print(torch.mean(y1).detach().cpu())
                # print((y1 - torch.mean(y1, 0)).norm(p=2, dim=0).detach().cpu())
                # print(torch.mean(x2).detach().cpu())
                # print((x2 - torch.mean(x2, 0)).norm(p=2, dim=0).detach().cpu())
                # print(torch.mean(y2).detach().cpu())
                # print((y2 - torch.mean(y2, 0)).norm(p=2, dim=0).detach().cpu())
                # normalized_part = self.normalize(
                #     torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, scores[sets[s]].unsqueeze(1)], dim=1))
                normalized_part = self.normalize(
                    torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean, set_ious_sum,
                            set_dist_mean, set_dist_sum], dim=1))
                # normalized_part = self.normalize(
                #     torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean, set_ious_sum,
                #             set_dist_mean, set_dist_sum, scores[sets[s]].unsqueeze(1)], dim=1))
                # normalized_part = self.normalize(
                #     torch.cat([x1, y1, x2, y2], dim=1))
                # x1 = (x1 - torch.mean(x1))/0.4
                # y1 = (y1 - torch.mean(y1)) / 0.4
                # x2 = (x2 - torch.mean(x2)) / 0.6
                # y2 = (y2 - torch.mean(y2)) / 0.6
                # input = torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean, set_ious_sum,
                #                    set_dist_mean, set_dist_sum, scores[sets[s]].unsqueeze(1)], dim=1)
                # input = normalized_part
                input = torch.cat([normalized_part, feats[sets[s]], scores[sets[s]].unsqueeze(1)], dim=1)
                # 13 features expanding
                # expaned_features = self.ln2(torch.cat([x1, y1, x2, y2, width, height, aspect_ratio, area, set_ious_mean,
                #                    set_ious_sum, set_dist_mean, set_dist_sum, scores[sets[s]].unsqueeze(1)], dim=1))
                # input = torch.cat([expaned_features, feats[sets[s]]], dim=1)

                _set_bboxes = bboxes[sets[s]]
                _set_scores = scores[sets[s]].unsqueeze(1)
                # if len(input) < self.set_size:
                #     # zero padding
                #     input = torch.cat([input, torch.zeros((self.set_size - input.shape[0]),
                #                                           input.shape[1]).cuda(device=device)], 0)
                #     _set_bboxes = torch.cat([_set_bboxes, torch.zeros((self.set_size - _set_bboxes.shape[0]),
                #                                                       _set_bboxes.shape[1]).cuda(device=device)], 0)
                #     _set_scores = torch.cat([_set_scores, torch.zeros((self.set_size - _set_scores.shape[0]),
                #                                                       _set_scores.shape[1]).cuda(device=device)], 0)
                # if len(input) > self.set_size:
                #     # _, top_scores_idx = input[:, -1].sort(descending=True)
                #     # input = input[top_scores_idx][:self.set_size]
                #     input = input[:self.set_size]
                #     _set_bboxes = _set_bboxes[:self.set_size]
                #     _set_scores = _set_scores[:self.set_size]
                # if mode == 'train':
                randperm = torch.randperm(len(input))
                input = input[randperm]
                set_bboxes.append(_set_bboxes[randperm])
                set_scores.append(_set_scores[randperm])
                # else:
                #     set_bboxes.append(_set_bboxes)
                #     set_scores.append(_set_scores)
                # set_bboxes.append(_set_bboxes)
                # set_scores.append(_set_scores)
                inputs.append(input)
                input_labels.append(c)
                normalization_data.append(normalization_data_per_set)
                centroids_per_set.append(torch.mean(centroids, 0))
                # t1 = datetime.datetime.now()
                # if input.shape[0] == 1:
                #     # pred = self.set_forward5(input).unsqueeze(0).unsqueeze(1)
                #     pred = self.set_forward5(input)
                # else:
                pred = self.set_forward5(input)
                pred = pred.unsqueeze(0)
                # pred = pred.T

                # de-normalization
                pred[:, 0] = ((pred[:, 0] * normalization_data_per_set['x1_std']) + normalization_data_per_set[
                    'x1_mean'])
                pred[:, 1] = ((pred[:, 1] * normalization_data_per_set['y1_std']) + normalization_data_per_set[
                    'y1_mean'])
                pred[:, 2] = ((pred[:, 2] * normalization_data_per_set['x2_std']) + normalization_data_per_set[
                    'x2_mean'])
                pred[:, 3] = ((pred[:, 3] * normalization_data_per_set['y2_std']) + normalization_data_per_set[
                    'y2_mean'])
                # de-normalize score
                # pred[:, 4] = ((pred[:, 4] * normalization_data_per_set['scores_std']) + normalization_data_per_set[
                #     'scores_mean'])

                preds.append(pred)
        if len(preds) > 0:
            preds_tensor = torch.stack(preds, dim=1).squeeze(0)
            # predicted_scores = self.set_forward5(preds_tensor)
            predicted_scores = preds_tensor[:, -1]
            preds_tensor_reshaped = torch.zeros((preds_tensor.shape[0], 4)).cuda(device=device)
            preds_tensor_reshaped[:, 2] = preds_tensor[:, 2] * img_shape[1]
            preds_tensor_reshaped[:, 3] = preds_tensor[:, 3] * img_shape[0]
            preds_tensor_reshaped[:, 0] = preds_tensor[:, 0] * img_shape[1]
            preds_tensor_reshaped[:, 1] = preds_tensor[:, 1] * img_shape[0]
            input_labels = torch.stack(input_labels)
            if predicted_scores.ndim == 0:
                predicted_scores = predicted_scores.unsqueeze(0)
        else:
            predicted_scores = torch.zeros(0).cuda(device=device)
            preds_tensor_reshaped = torch.empty(0).cuda(device=device)
            input_labels = torch.empty(0).cuda(device=device)
        return inputs, preds_tensor_reshaped, predicted_scores, input_labels, set_bboxes, set_scores, centroids_per_set, normalization_data

    def _get_target_old(self,
                    sets,
                    set_labels,
                    set_bboxes,
                    centroids_per_set,
                    normalization_data,
                    gt_bboxes,
                    gt_labels,
                    preds,
                    device=0):
        """
        returns deepsets labels.
        a. for each class on each set, find closest gt,
        b. calculate iou between set and gt box,
        c. normalize with max iou.
        """
        valid_preds = torch.empty((0, 5), dtype=torch.float32).cuda(device=device)
        valid_set_bboxes = []  # for wandb log
        valid_set_scores = []  # for wandb log
        valid_ious = []  # for error viz
        valid_centroids = []
        valid_normalization_data = []
        gt_box_per_set = torch.empty((0, 4), dtype=torch.float32).cuda(device=device)
        gt_valid_sets = torch.zeros(len(sets), dtype=torch.int64).cuda(device=device)
        score_target = torch.zeros(len(sets), dtype=torch.float32).cuda(device=device)
        for j, set in enumerate(sets):
            c = set_labels[j]
            assert len(set) > 0 and c in gt_labels
            gt_class_inds = torch.where(gt_labels == c)[0].squeeze()
            # takes boxes of relevant gt
            class_boxes = torch.index_select(gt_bboxes, 0, gt_class_inds)
            gt_iou = bbox_overlaps(set_bboxes[j], class_boxes)
            vals, row_idx = gt_iou.max(0)  # row idx: indices of set elements with max ious with each GT (1, num_gts)
            col_idx = vals.argmax(0)  # col idx: index of GT with highest iou with any set element
            if torch.max(gt_iou[:, col_idx]) < 0.5:
                gt_valid_sets[j] = 0
            else:
                gt_valid_sets[j] = 1
            score_target[j] = torch.max(gt_iou[:, col_idx])
            valid_set_bboxes.append(set_bboxes[j])
            valid_set_scores.append(sets[j][:, -1])
            valid_ious.append(gt_iou[:, col_idx])
            valid_centroids.append(centroids_per_set[j])
            valid_normalization_data.append(normalization_data[j])
            # gt_box_per_set.append(class_boxes[col_idx])
            valid_preds = torch.cat((valid_preds, preds[j]), 0)
            gt_box_per_set = torch.cat((gt_box_per_set, class_boxes[col_idx].unsqueeze(0)), 0)
            if gt_class_inds.ndim == 0:
                gt_class_inds = gt_class_inds.unsqueeze(0)

        return valid_preds, valid_set_bboxes, valid_set_scores, valid_ious, valid_centroids, valid_normalization_data, \
               score_target, gt_box_per_set, gt_valid_sets

    def _old_loss(self, pred, valid_centroids, valid_normalization_data, gt_box_per_set, gt_valid_sets, score_target, set_labels,
              img_shape=None, device=0, lambda_iou=2, lambda_l1=5):
        losses = dict()
        assert pred is not None, "pred is None"
        _l1_loss = torch.zeros(1).cuda(device=device)
        _giou_loss = torch.zeros(1).cuda(device=device)
        pred = pred.T
        pred_for_loss = torch.empty_like(pred)
        centroids_per_set = torch.empty((len(valid_centroids), 2)).cuda(device)
        ndps = torch.empty((len(valid_normalization_data), 9)).cuda(device)
        for i, _set in enumerate(valid_centroids):
            centroids_per_set[i] = valid_centroids[i]
            ndps[i][0] = valid_normalization_data[i]['x1_mean']
            ndps[i][1] = valid_normalization_data[i]['x1_std']
            ndps[i][2] = valid_normalization_data[i]['y1_mean']
            ndps[i][3] = valid_normalization_data[i]['y1_std']
            ndps[i][4] = valid_normalization_data[i]['x2_mean']
            ndps[i][5] = valid_normalization_data[i]['x2_std']
            ndps[i][6] = valid_normalization_data[i]['y2_mean']
            ndps[i][7] = valid_normalization_data[i]['y2_std']
            ndps[i][8] = valid_normalization_data[i]['set_size']

        pred_for_loss[:, 2] = ((pred[:, 2] * ndps[:, 5]) + ndps[:, 4]) * img_shape[1]
        pred_for_loss[:, 3] = ((pred[:, 3] * ndps[:, 7]) + ndps[:, 6]) * img_shape[0]
        pred_for_loss[:, 0] = ((pred[:, 0] * ndps[:, 1]) + ndps[:, 0]) * img_shape[1]
        pred_for_loss[:, 1] = ((pred[:, 1] * ndps[:, 3]) + ndps[:, 2]) * img_shape[0]
        if len(pred) == 0 or not torch.sum(gt_valid_sets):
            losses['loss_deepsets_total'] = None
        else:
            _giou_loss = self.loss_giou.forward(pred_for_loss[gt_valid_sets > 0, :4], gt_box_per_set[gt_valid_sets > 0],
                                                self.giou_coef)
            _l1_loss = self.loss_l1.forward(pred_for_loss[gt_valid_sets > 0, :4], gt_box_per_set[gt_valid_sets > 0])
            # _ce_loss = self.loss_ce.forward(pred[:, 4], gt_valid_sets)
            # _l1_score_loss = self.loss_l1.forward(pred[:, 4], torch.tensor(score_target).cuda(device=device))
            # _ap_loss = self.loss_ap.forward(pred[:, 4].unsqueeze(0), gt_valid_sets.unsqueeze(0))

            ### full map
            _ap_loss_l = []
            set_labels = torch.FloatTensor(set_labels).cuda(device=device)
            for label in torch.unique(set_labels):
                _ap_loss_iou = []
                label_mask = set_labels == label
                for iou in range(50, 100, 5):
                    iou_mask = score_target > iou/100
                    if torch.sum(iou_mask[label_mask]) > 0:
                        _ap_loss_iou.append(self.loss_ap.forward(pred[label_mask, 4].unsqueeze(0), iou_mask[label_mask].unsqueeze(0)))
                    else:
                        _ap_loss_iou.append(torch.ones(1).cuda(device=device))
                _ap_loss_l.append(sum(_ap_loss_iou)/len(_ap_loss_iou))
            _ap_loss = sum(_ap_loss_l)/len(_ap_loss_l)

            # ap 0.5-0.95
            # _ap_loss_iou = []
            # for iou in range(50, 100, 5):
            #     iou_mask = score_target > iou/100
            #     if torch.sum(iou_mask) > 0:
            #         _ap_loss_iou.append(self.loss_ap.forward(pred[:, 4].unsqueeze(0), iou_mask.unsqueeze(0)))
            #     else:
            #         _ap_loss_iou.append(torch.ones(1).cuda(device=device))
            # _ap_loss = sum(_ap_loss_iou)/len(_ap_loss_iou)

            losses['loss_deepsets_total'] = lambda_iou * _giou_loss \
                                            + lambda_l1 * _l1_loss  \
                                            + 2*_ap_loss
            losses['giou'] = lambda_iou * _giou_loss
            losses['l1'] = lambda_l1 * _l1_loss  # / (np.mean(img_shape[0] + img_shape[1]))
            losses['std'] = 2*(torch.mean(ndps[:, 1]) + torch.mean(ndps[:, 3]) \
                            + torch.mean(ndps[:, 5]) + torch.mean(ndps[:, 7]))
            losses['bce'] = 2*_ap_loss
            # losses['bce'] = 0.5 * _l1_score_loss
            losses['valid_set_acc'] = torch.sum(torch.eq(pred[:, 4]>0.5, gt_valid_sets))/float(len(gt_valid_sets))
        return losses

