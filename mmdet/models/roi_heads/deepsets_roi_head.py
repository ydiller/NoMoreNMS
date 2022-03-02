# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
import numpy as np
import sys
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from tools.set_transformer import SetTransformer

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
      valid_mask = y > iou_thr
      # valid_inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
      # loss = -(y[valid_inds]*p[valid_inds].T).mean()
      # loss = -(p[valid_inds]).mean()
      # loss = -torch.log(torch.sum(p[valid_mask]))
      valid_mask = valid_mask.type(torch.float)/len(valid_mask.nonzero(as_tuple=False))
      loss = -torch.mean(valid_mask*p.T)
      # print(y[valid_inds])
      return loss

   # def forward(self, y_hat, y, iou_thr):
   #    p = F.log_softmax(y*y_hat.T, 1)
   #    loss = -p[torch.argmax(y_hat)]
   #    return loss


@HEADS.register_module()
class DeepsetsRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
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
                 indim=1041,
                 ds1=1000,  # previously 400
                 ds2=800,  # previously 150
                 ds3=600,  # previously 50
                 set_size=6,
                 reg=1,
                 include_ds4=1,
                 loss_mse=dict(
                     type='MSELoss',
                     loss_weight=1.0),
                 loss_ce=dict(
                     type='CrossEntropyLoss',
                     loss_weight=4.0)):
        super(DeepsetsRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.set_size = set_size
        self.reg = reg
        self.include_ds4 = include_ds4
        self.loss_mse = build_loss(loss_mse)
        self.loss_ce = build_loss(loss_ce)
        # self.ds1 = PermEqui2_mean(indim, ds1)
        # self.ds2 = PermEqui2_mean(ds1, ds2)
        # self.ds3 = PermEqui2_mean(ds2, ds3)
        # if not include_ds4:
        #     self.ds4 = PermEqui2_mean(ds3, 1)
        # else:
            # self.ds4 = PermEqui2_mean(ds3, int(ds3/2))
            # self.bn4 = torch.nn.BatchNorm1d(int(ds3/2))
        # self.ds5 = PermEqui2_mean(int(ds3/2), 1)

        self.ln1 = torch.nn.Linear(indim, 128)
        # self.ln2 = torch.nn.Linear(512, 256)
        # self.ln3 = torch.nn.Linear(256, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        # self.bn2 = torch.nn.BatchNorm1d(256)
        # self.bn3 = torch.nn.BatchNorm1d(128)
        # self.ds1 = PermEqui2_mean(128, 128)
        # self.ds2 = PermEqui2_mean(128, 128)
        # self.ds3 = PermEqui2_mean(128, 1)

        self.set_transformer = SetTransformer(dim_input=128, num_outputs=128, dim_output=1, num_inds=16, dim_hidden=128,
                                              num_heads=4, ln=False, mode="dense")
        self.ln4 = torch.nn.Linear(128, 1)
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
            outs = outs + (mask_results['mask_pred'], )
        return outs

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
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        ######## deepsets ######
        num_proposals_per_img = tuple(len(res.bboxes) for res in sampling_results)
        rois = bbox2roi([res.bboxes for res in sampling_results])
        feats = bbox_results["last_layer_feats"]
        rois = rois.split(num_proposals_per_img, 0)
        scores = bbox_results["cls_score"].split(num_proposals_per_img, 0)
        bboxes = bbox_results["bbox_pred"].split(num_proposals_per_img, 0)
        feats = feats.split(num_proposals_per_img, 0)
        device = bboxes[0].get_device()
        ds_cfg = self.train_cfg['deepsets_config']
        loss_deepsets = dict()
        loss_deepsets['loss_deepsets_ce'] = 0.
        loss_deepsets['ds_acc'] = 0.
        loss_deepsets['iou_error'] = 0.
        loss_deepsets['ds_pred_on_max'] = 0.
        for i in range(num_imgs):
            bbox, score = self.bbox_head.get_bboxes(
                rois[i],
                scores[i],
                bboxes[i],
                img_metas[i]['img_shape'],
                img_metas[i]['scale_factor'],
                rescale=False,
                cfg=None)
            sets, preds, set_labels, set_bboxes = self._forward_deepsets(bbox, score, feats[i],
                                                                             img_shape=img_metas[i]['img_shape'], gt_labels=gt_labels[i],
                                                                             ds_cfg=ds_cfg, device=device, score_thr=0.05)
            valid_preds, valid_set_bboxes, valid_set_scores, valid_ious, one_hot_targets, soft_targets = \
                self._get_target(sets, set_labels, set_bboxes, gt_bboxes[i], gt_labels[i], preds, device=device)
            loss_deepsets_i = self._loss(valid_preds, valid_ious, one_hot_targets, soft_targets,
                                                    valid_set_scores, device=device)
            if loss_deepsets_i['loss_deepsets_ce'] is not None:
                loss_deepsets['loss_deepsets_ce'] += loss_deepsets_i['loss_deepsets_ce']
                loss_deepsets['ds_acc'] += loss_deepsets_i['ds_acc']
                loss_deepsets['iou_error'] += loss_deepsets_i['iou_error']
                loss_deepsets['ds_pred_on_max'] += loss_deepsets_i['ds_pred_on_max']
        for key, value in loss_deepsets.items():
            loss_deepsets[key] = value/num_imgs
        losses.update(loss_deepsets)

        ### temp
        # p = [torch.argmax(x) for x in preds]
        # z = [torch.argmax(x[:, -1]) for x in sets]
        # eq = [p == z for p, z in zip(p, z)]
        # max_score_rate = float(sum(eq)) / (len(eq)+0.000001)
        ## wandb bounding box
        # pred_boxes_log = []
        # gt_boxes_log = []
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
        #             "box_caption": f'{_s} ({valid_set_scores[_s][_b][-1].item():.2f})',
        #             # "box_caption:": "%d (%.2f)" % (_s, sets[_s][_b][-1].item()),
        #             "scores": {"score": valid_set_scores[_s][_b][-1].item(),
        #                        "set id": _s}
        #         }
        #         pred_boxes_log.append(box_data)
        # for _b, b in enumerate(gt_bboxes[0]):
        #     box_data = {
        #         "position": {
        #             "minX": b[0].item(),
        #             "maxX": b[2].item(),
        #             "minY": b[1].item(),
        #             "maxY": b[3].item()},
        #         "class_id": gt_labels[0][_b].item(),
        #         "domain": "pixel",
        #         "box_caption": f'{gt_labels[0][_b].item()}'
        #     }
        #     gt_boxes_log.append(box_data)
        ###
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                wandb.log({"CE loss": loss_deepsets["loss_deepsets_ce"],
                           "ds_acc": loss_deepsets["ds_acc"],
                           "iou_error": loss_deepsets["iou_error"],
                           "max score predictions": loss_deepsets_i['ds_pred_on_max']
                           })
        else:
            if self.train_cfg.with_wandb:
                wandb.log({"CE loss": loss_deepsets["loss_deepsets_ce"],
                           "ds_acc": loss_deepsets["ds_acc"],
                           "iou_error": loss_deepsets["iou_error"],
                           "max score predictions": loss_deepsets_i['ds_pred_on_max']
                           })
        # "max score rate": max_score_rate})
        # "image": wandb.Image(img[0].cpu().permute(1, 2, 0).numpy(),
        #   boxes={"predictions": {"box_data": pred_boxes_log}, "gts": {"box_data": gt_boxes_log}})})

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

        det_bboxes, det_labels, sets = self.simple_test_deepsets(
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

    def _forward_deepsets(self, multi_bboxes, cls_score, last_layer_feats,
                mode='train', ds_cfg=None, img_shape=None, gt_labels=None, device=0, score_thr=0.05):
        """
        Args:
            labels: bbox labels (assigned by comparing to gt)
        """
        num_classes = 80
        preds = []
        inputs = []
        set_bboxes = []
        set_scores = []
        input_labels = []
        top_c = ds_cfg['top_c']
        max_num = ds_cfg['max_num']
        iou_threshold = ds_cfg['iou_thresh']
        if mode == 'train':
            classes = torch.unique(gt_labels)
        else:
            # classes = torch.unique(torch.topk(cls_score[:, :-1], top_c, dim=1)[1])  # top_classes_on_set
            classes = torch.range(0, num_classes-1, dtype=int).cuda(device=device)
        zero = torch.zeros(1).cuda(device=device)
        one = torch.ones(1).cuda(device=device)
        for c in classes:  # c in 0 to data length - 1
            sets = []
            bboxes = multi_bboxes[:, (c) * 4:((c) + 1) * 4]
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
                if is_clustered[j] == 1:
                    selected_indices = torch.nonzero((row > iou_threshold) * is_clustered, as_tuple=False).squeeze()
                    is_clustered *= torch.where(row > iou_threshold, zero, one)
                    if selected_indices.ndim == 0:
                        selected_indices = selected_indices.unsqueeze(0)
                    # x1 = bboxes[selected_indices, 0] / img_shape[1]
                    # x2 = bboxes[selected_indices, 2] / img_shape[1]
                    # y1 = bboxes[selected_indices, 1] / img_shape[0]
                    # y2 = bboxes[selected_indices, 3] / img_shape[0]
                    # width = (x2 - x1) / img_shape[1]
                    # height = (y2 - y1) / img_shape[0]
                    # aspect_ratio = torch.div(width, height + sys.float_info.epsilon)
                    # if (torch.max(aspect_ratio) - torch.min(aspect_ratio)) > 1.2:
                    #     selected_indices_wide = selected_indices[torch.nonzero(aspect_ratio >= 1, as_tuple=False)].squeeze(1)
                    #     selected_indices_tall = selected_indices[torch.nonzero(aspect_ratio < 1, as_tuple=False)].squeeze(1)
                    #     if len(selected_indices_wide > 0):
                    #         sets.append(selected_indices_wide)
                    #     if len(selected_indices_tall > 0):
                    #         sets.append(selected_indices_tall)
                    # else:
                    sets.append(selected_indices)

            for s, _set in enumerate(sets):
                if mode == 'train' and len(_set) == 1:  # were set includes only one object
                    continue
                # score_threshold = np.percentile(scores[sets[s]].detach().cpu().numpy(), 70)
                # score_mask = scores[sets[s]] >= score_threshold
                # sets[s] = sets[s][score_mask.nonzero(as_tuple=False).squeeze(1)]
                x1 = bboxes[sets[s], 0].unsqueeze(1)/img_shape[1]
                x2 = bboxes[sets[s], 2].unsqueeze(1)/img_shape[1]
                y1 = bboxes[sets[s], 1].unsqueeze(1)/img_shape[0]
                y2 = bboxes[sets[s], 3].unsqueeze(1)/img_shape[0]
                width = (x2 - x1)/img_shape[1]
                height = (y2 - y1)/img_shape[0]
                aspect_ratio = torch.div(width, height+sys.float_info.epsilon)
                area = width*height
                set_area_mean = area/torch.mean(area)
                set_area_sum = area/torch.sum(area)
                set_aspect_mean = aspect_ratio/torch.mean(aspect_ratio)
                set_aspect_sum = aspect_ratio/torch.sum(aspect_ratio)
                # one_hot_classes = torch.zeros((1, num_classes)).repeat(len(_set), 1).cuda(device=device)
                # one_hot_classes[:, c] = 1
                centroids = torch.cat([x2-x1, y2-1], dim=1).cuda(device=device)
                set_dist_mean = torch.mean(torch.cdist(centroids, centroids, p=2), 1).unsqueeze(1)
                set_dist_sum = torch.sum(torch.cdist(centroids, centroids, p=2), 1).unsqueeze(1)
                set_ious_mean = torch.mean(ious[sets[s]][:, sets[s]], 1).unsqueeze(1)
                set_ious_sum = torch.sum(ious[sets[s]][:, sets[s]], 1).unsqueeze(1)
                input = torch.cat([bboxes[sets[s]], feats[sets[s]], width, height, aspect_ratio, area, set_ious_mean,
                                   set_ious_sum, set_dist_mean, set_dist_sum, set_area_mean, set_area_sum,
                                   set_aspect_mean, set_aspect_sum, scores[sets[s]].unsqueeze(1)], dim=1)
                _set_bboxes = bboxes[sets[s]]
                # _set_scores = scores[sets[s]].unsqueeze(1)
                # if len(input) < self.set_size:
                #     # zero padding
                #     input = torch.cat([input, torch.zeros((self.set_size - input.shape[0]), input.shape[1]).cuda()], 0)
                #     _set_bboxes = torch.cat([_set_bboxes, torch.zeros((self.set_size - _set_bboxes.shape[0]), _set_bboxes.shape[1]).cuda()], 0)
                # else:
                # _, top_scores_idx = input[:, -1].sort(descending=True)
                # input = input[top_scores_idx]
                # _set_bboxes = _set_bboxes[top_scores_idx]
                randperm = torch.randperm(len(input))
                input = input[randperm]
                set_bboxes.append(_set_bboxes[randperm])
                # set_scores.append(_set_scores[randperm])
                inputs.append(input)
                input_labels.append(c)
                # pred = self.set_forward2(input)
                if input.shape[0] == 1:
                    pred = self.set_forward3(input).unsqueeze(0).unsqueeze(1)
                else:
                    pred = self.set_forward3(input)
                preds.append(pred)
        return inputs, preds, input_labels, set_bboxes

    def _get_target(self,
                   sets,
                   set_labels,
                   set_bboxes,
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
        one_hot_targets = []
        soft_targets = []
        valid_preds = []
        valid_set_bboxes = []  # for wandb log
        valid_set_scores = []  # for wandb log
        valid_ious = []  # for error viz
        for j, set in enumerate(sets):
            c = set_labels[j]
            # c = torch.argmax(set[0, -1231:]).item()
            if len(set) > 0 and c in gt_labels:
                # gt_class_inds = torch.tensor(np.where(gt_labels.cpu().numpy() == c)).cuda().squeeze()
                gt_class_inds = torch.where(gt_labels == c)[0].squeeze()
                # takes boxes of relevant gt
                class_boxes = torch.index_select(gt_bboxes, 0, gt_class_inds)
                gt_iou = bbox_overlaps(set_bboxes[j], class_boxes)
                vals, row_idx = gt_iou.max(0)  # row idx: indices of set elements with max ious with each GT (1, num_gts)
                col_idx = vals.argmax(0)  # col idx: index of GT with highest iou with any set element
                # min max norm
                # min_vals, min_row_idx = gt_iou[:, col_idx].min(0)
                # v_max = gt_iou[row_idx[col_idx], col_idx]+sys.float_info.epsilon
                # v_min = gt_iou[min_row_idx, col_idx]+sys.float_info.epsilon
                # if v_max == v_min:
                #     v_min = 0
                # if DS_TYPE == 'soft':
                    # targets.append((gt_iou[:, col_idx] - v_min) / (v_max - v_min) * (1.0 - 0.0) + 0.0 +sys.float_info.epsilon)
                # print(torch.max(gt_iou[:, col_idx]))
                if torch.max(gt_iou[:, col_idx]) < 0.8:
                    continue
                soft_targets.append(gt_iou[:, col_idx] / (gt_iou[row_idx[col_idx], col_idx]+sys.float_info.epsilon))
                # soft_targets.append(gt_iou[:, col_idx] / torch.sum(gt_iou[:, col_idx]))
                # soft_targets.append(torch.pow(gt_iou[:, col_idx] / (gt_iou[row_idx[col_idx], col_idx] + sys.float_info.epsilon), 3))
                # soft_targets.append(gt_iou[:, col_idx])
                # t = gt_iou[:, col_idx] / (gt_iou[row_idx[col_idx], col_idx]+sys.float_info.epsilon)
                # t -= t.min(0, keepdim=True)[0]
                # t /= t.max(0, keepdim=True)[0]
                # if torch.sum(torch.isnan(t)) == 0:
                #     soft_targets.append(t)
                # else:
                #     soft_targets.append(torch.ones_like(t))
                # else:
                trg = torch.zeros(len(set), dtype=torch.int64).cuda(device=device)
                trg[row_idx[col_idx]] = 1
                one_hot_targets.append(trg)
                valid_preds.append(preds[j])
                valid_set_bboxes.append(set_bboxes[j])
                valid_set_scores.append(sets[j][:, -1])
                valid_ious.append(gt_iou[:, col_idx])
                    # soft x-ent
                    # trg = gt_iou[:, col_idx] / (gt_iou[row_idx[col_idx], col_idx] + sys.float_info.epsilon)
                    # trg = torch.softmax(trg, 0)
                    # targets.append(trg)

            # else:
            #     trg = torch.zeros(len(set)).cuda()
            #     targets.append(trg)

        return valid_preds, valid_set_bboxes, valid_set_scores, valid_ious, one_hot_targets, soft_targets

    def _loss(self, preds, valid_ious, one_hot_targets, soft_targets, valid_set_scores, device=0):
        losses = dict()
        c = sys.float_info.epsilon
        skipped = 0
        if preds is not None:
            _mse_loss = torch.zeros(1).cuda(device=device)
            _ce_loss = torch.zeros(1).cuda(device=device)
            _soft_ce = torch.zeros(1).cuda(device=device)
            _error = torch.zeros(1).cuda(device=device)
            _ds_acc = torch.zeros(1).cuda(device=device)
            iou_error = torch.zeros(1).cuda(device=device)
            _ds_pred_on_max = torch.zeros(1).cuda(device=device)
            for i, pred in enumerate(preds):
                # if torch.sum(one_hot_targets[i]) == 0:
                #     skipped += 1
                #     print('prob')
                #     continue
                # if DS_TYPE == 'hard':
                    # weight = ((deepsets_targets[i]) * (len(deepsets_targets[i][deepsets_targets[i] == 0])-1) + 1).type(torch.float)
                    # weight = ((deepsets_targets[i]) * 9 + 1).type(torch.float)  # const wegiht
                    # _loss += self.loss_ce(  # self.loss_ms
                    #     pred,  # pred.squeeze()
                    #     deepsets_targets[i],
                    #     weight=weight)
                # _ce_loss += self.loss_ce(  # self.loss_ms
                #     pred.T,  # pred.squeeze()
                #     torch.argmax(one_hot_targets[i]).unsqueeze(0))
                _ds_acc += torch.argmax(pred) == torch.argmax(one_hot_targets[i])
                iou_error += torch.max(valid_ious[i]) - valid_ious[i][torch.argmax(pred)]
                _ds_pred_on_max += torch.argmax(pred) == torch.argmax(valid_set_scores[i])

                #nms testing:
                # _ds_acc += torch.argmax(valid_set_scores[i]) == torch.argmax(one_hot_targets[i])
                # iou_error += torch.max(valid_ious[i]) - valid_ious[i][torch.argmax(valid_set_scores[i])]

                # else:
                # _mse_loss += self.loss_mse(  # self.loss_ms
                #     pred,  # pred.squeeze()
                #     soft_targets[i])
                    # _ds_acc += torch.argmax(pred) == torch.argmax(deepsets_targets[i])
                    # _ds_pred_on_max += pred[torch.argmax(deepsets_targets[i])]
                # _error += torch.sum(torch.abs(deepsets_targets[i]-pred.squeeze()))/pred.shape[0]
                _ce_loss += SoftCrossEntropyLoss().forward(pred, soft_targets[i], 0.98)
                c += 1
            _ce_loss /= c
            # _mse_loss /= c
            # _soft_ce /= c
            _error /= c
            _ds_acc /= c
            iou_error /= c
            _ds_pred_on_max /= c
        if len(preds) == 0:
            # print('prob')
            # dl = torch.zeros(1).cuda(device=device)
            # dl.requires_grad = True
            # losses['loss_deepsets_ce'] = dl
            # # losses['loss_deepsets_mse'] = dl
            # losses['ds_acc'] = torch.zeros(1).cuda(device=device)
            # losses['iou_error'] = torch.zeros(1).cuda(device=device)
            losses['loss_deepsets_ce'] = None
            losses['ds_acc'] = None
            losses['iou_error'] = None
            losses['ds_pred_on_max'] = None

        else:
            losses['loss_deepsets_ce'] = _ce_loss + self.reg*iou_error
            # losses['loss_deepsets_mse'] = _mse_loss
            losses['ds_acc'] = _ds_acc
            losses['iou_error'] = iou_error
            losses['ds_pred_on_max'] = _ds_pred_on_max
        return losses
